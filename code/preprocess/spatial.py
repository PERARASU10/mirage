from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

try:
    import torch
except Exception:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mirage import HierarchicalABMIL, PatchProjector, PositionalMLP, ensure_directory, save_embedding_h5
from mirage.config import OUT_DIR, WSI_CUP, WSI_HYPO, WSI_LARYNX, WSI_LYMPH, WSI_ORAL, WSI_OROPH1, WSI_OROPH2


MAX_PATCHES = 2048


class RegionFusion(torch.nn.Module):
    def __init__(self, input_dim=1024, output_dim=512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.GELU(),
            torch.nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def extract_pid_from_filename(path: Path) -> str:
    match = re.search(r"HE_(\d+)", path.name)
    if match:
        return match.group(1).zfill(3)
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return digits[-3:].zfill(3)


def collect_wsi_h5_files(directories):
    files = []
    for directory in directories:
        files.extend(sorted(Path(directory).glob("*.h5")))
    return files


def read_wsi_h5(path: Path):
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["features"], dtype=np.float32), np.asarray(handle["coords"], dtype=np.float32)


def reduce_patches(features: np.ndarray, coords: np.ndarray):
    if len(features) <= MAX_PATCHES:
        return features, coords
    merged = np.concatenate([features, coords], axis=1)
    kmeans = MiniBatchKMeans(n_clusters=MAX_PATCHES, random_state=42, batch_size=256)
    kmeans.fit(merged)
    centers = kmeans.cluster_centers_
    return centers[:, : features.shape[1]], centers[:, features.shape[1] :]


def encode_region(paths, projector, pos_mlp, habmil, device):
    if not paths:
        return None, 0.0, 0

    feature_parts = []
    coord_parts = []
    for path in paths:
        features, coords = read_wsi_h5(path)
        feature_parts.append(features)
        coord_parts.append(coords)
    features = np.concatenate(feature_parts, axis=0)
    coords = np.concatenate(coord_parts, axis=0)
    features = normalize(features, axis=1)
    coord_min = coords.min(axis=0)
    coord_span = coords.max(axis=0) - coord_min
    coord_span[coord_span == 0] = 1.0
    coords = (coords - coord_min) / coord_span
    features, coords = reduce_patches(features, coords)

    cluster_ids = MiniBatchKMeans(n_clusters=max(1, min(32, len(features))), random_state=42).fit_predict(features)
    with torch.no_grad():
        patch_tensor = torch.from_numpy(features).float().to(device)
        coord_tensor = torch.from_numpy(coords).float().to(device)
        projected = projector(patch_tensor)
        spatial_bias = pos_mlp(coord_tensor)
        region_embedding = habmil.encode_bag(projected + spatial_bias, torch.from_numpy(cluster_ids).to(device))

    quality = float(min(len(features) / float(MAX_PATCHES), 1.0))
    return region_embedding.detach().cpu(), quality, len(features)


def parse_args():
    parser = argparse.ArgumentParser(description="MIRAGE-Net spatial preprocessing")
    parser.add_argument("--wsi_lymph", type=Path, default=WSI_LYMPH)
    parser.add_argument("--wsi_cup", type=Path, default=WSI_CUP)
    parser.add_argument("--wsi_hypo", type=Path, default=WSI_HYPO)
    parser.add_argument("--wsi_larynx", type=Path, default=WSI_LARYNX)
    parser.add_argument("--wsi_oral", type=Path, default=WSI_ORAL)
    parser.add_argument("--wsi_oroph1", type=Path, default=WSI_OROPH1)
    parser.add_argument("--wsi_oroph2", type=Path, default=WSI_OROPH2)
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    if torch is None:
        raise RuntimeError("spatial.py requires PyTorch")

    args = parse_args()
    ensure_directory(args.outdir)

    files = collect_wsi_h5_files([args.wsi_lymph, args.wsi_cup, args.wsi_hypo, args.wsi_larynx, args.wsi_oral, args.wsi_oroph1, args.wsi_oroph2])
    grouped = defaultdict(lambda: {"primary": [], "lymph": []})
    for path in files:
        bucket = "lymph" if "LymphNode" in str(path) else "primary"
        grouped[extract_pid_from_filename(path)][bucket].append(path)

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    projector = PatchProjector(in_dim=1024, model_dim=512).to(device)
    pos_mlp = PositionalMLP(model_dim=512).to(device)
    habmil = HierarchicalABMIL(d_model=512).to(device)
    fusion = RegionFusion().to(device)

    patient_ids = []
    embeddings = []
    embedding_var = []
    qualities = []
    n_patches = []
    tumor_site = []

    for patient_id, regions in sorted(grouped.items()):
        patient_ids.append(patient_id)
        samples = []
        primary_quality = 0.0
        lymph_quality = 0.0
        total_patches = 0
        for _ in range(8):
            primary_emb, primary_quality, primary_patches = encode_region(regions["primary"], projector, pos_mlp, habmil, device)
            lymph_emb, lymph_quality, lymph_patches = encode_region(regions["lymph"], projector, pos_mlp, habmil, device)
            total_patches = max(total_patches, primary_patches + lymph_patches)
            if primary_emb is None and lymph_emb is None:
                combined = torch.zeros(512, device=device)
            elif primary_emb is None:
                combined = lymph_emb.to(device)
            elif lymph_emb is None:
                combined = primary_emb.to(device)
            else:
                combined = fusion(torch.cat([primary_emb.to(device), lymph_emb.to(device)], dim=-1).unsqueeze(0)).squeeze(0)
            samples.append(combined.detach().cpu().numpy())
        sample_stack = np.stack(samples, axis=0)
        embeddings.append(sample_stack.mean(axis=0))
        embedding_var.append(sample_stack.var(axis=0))
        qualities.append(float(max(primary_quality, lymph_quality)))
        n_patches.append(total_patches)
        tumor_site.append("primary_and_lymph" if regions["primary"] and regions["lymph"] else "single_site")

    save_embedding_h5(
        args.outdir / "spatial_embedding_512.h5",
        patient_ids,
        np.asarray(embeddings, dtype=np.float32),
        quality_scores=np.asarray(qualities, dtype=np.float32),
        extra={
            "embedding_var_512": np.asarray(embedding_var, dtype=np.float32),
            "n_patches": np.asarray(n_patches, dtype=np.int32),
            "tumor_site": tumor_site,
        },
    )
    torch.save(projector.state_dict(), args.outdir / "patch_projector.pt")
    torch.save(pos_mlp.state_dict(), args.outdir / "positional_mlp.pt")
    torch.save(habmil.state_dict(), args.outdir / "habmil_model.pt")
    torch.save(fusion.state_dict(), args.outdir / "spatial_fusion.pt")
    joblib.dump({"max_patches": MAX_PATCHES}, args.outdir / "spatial_preproc_objects.joblib")

    print(f"spatial patients: {len(patient_ids)}")
    print(f"embedding shape: {(len(patient_ids), 512)}")


if __name__ == "__main__":
    main()
