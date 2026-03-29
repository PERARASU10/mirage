from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from mirage import HierarchicalABMIL, PatchProjector, PositionalMLP


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


def reduce_patches(features: np.ndarray, coords: np.ndarray, max_patches: int):
    if len(features) <= max_patches:
        return features, coords
    merged = np.concatenate([features, coords], axis=1)
    kmeans = MiniBatchKMeans(n_clusters=max_patches, random_state=42, batch_size=256)
    kmeans.fit(merged)
    centers = kmeans.cluster_centers_
    return centers[:, : features.shape[1]], centers[:, features.shape[1] :]


def encode_region(features, coords, projector, pos_mlp, habmil, device, max_patches):
    if features is None or len(features) == 0:
        return None, 0.0, 0
    features = normalize(features, axis=1)
    coord_min = coords.min(axis=0)
    coord_span = coords.max(axis=0) - coord_min
    coord_span[coord_span == 0] = 1.0
    coords = (coords - coord_min) / coord_span
    features, coords = reduce_patches(features, coords, max_patches)
    cluster_ids = MiniBatchKMeans(n_clusters=max(1, min(32, len(features))), random_state=42).fit_predict(features)
    with torch.no_grad():
        patch_tensor = torch.from_numpy(features).float().to(device)
        coord_tensor = torch.from_numpy(coords).float().to(device)
        projected = projector(patch_tensor)
        spatial_bias = pos_mlp(coord_tensor)
        region_embedding = habmil.encode_bag(projected + spatial_bias, torch.from_numpy(cluster_ids).to(device))
    quality = float(min(len(features) / float(max_patches), 1.0))
    return region_embedding.detach().cpu(), quality, len(features)


def get_spatial_embedding(primary_wsi, lymph_wsi, resources_path, device):
    resources_path = Path(resources_path)
    projector = PatchProjector(in_dim=1024, model_dim=512).to(device)
    pos_mlp = PositionalMLP(model_dim=512).to(device)
    habmil = HierarchicalABMIL(d_model=512).to(device)
    fusion = RegionFusion().to(device)

    projector.load_state_dict(torch.load(resources_path / "patch_projector.pt", map_location=device), strict=False)
    pos_mlp.load_state_dict(torch.load(resources_path / "positional_mlp.pt", map_location=device), strict=False)
    habmil.load_state_dict(torch.load(resources_path / "habmil_model.pt", map_location=device), strict=False)
    if (resources_path / "spatial_fusion.pt").exists():
        fusion.load_state_dict(torch.load(resources_path / "spatial_fusion.pt", map_location=device), strict=False)

    primary_features = np.asarray(primary_wsi.get("features", []), dtype=np.float32) if isinstance(primary_wsi, dict) else None
    primary_coords = np.asarray(primary_wsi.get("coords", []), dtype=np.float32) if isinstance(primary_wsi, dict) else None
    lymph_features = np.asarray(lymph_wsi.get("features", []), dtype=np.float32) if isinstance(lymph_wsi, dict) else None
    lymph_coords = np.asarray(lymph_wsi.get("coords", []), dtype=np.float32) if isinstance(lymph_wsi, dict) else None

    primary_embedding, primary_quality, _ = encode_region(primary_features, primary_coords, projector, pos_mlp, habmil, device, 2048)
    lymph_embedding, lymph_quality, _ = encode_region(lymph_features, lymph_coords, projector, pos_mlp, habmil, device, 2048)

    if primary_embedding is None and lymph_embedding is None:
        return {"embedding": torch.zeros(512, device=device), "quality": 0.0, "present": 0}
    if primary_embedding is None:
        embedding = lymph_embedding
    elif lymph_embedding is None:
        embedding = primary_embedding
    else:
        with torch.no_grad():
            embedding = fusion(torch.cat([primary_embedding.to(device), lymph_embedding.to(device)], dim=-1).unsqueeze(0)).squeeze(0).cpu()
    return {"embedding": embedding, "quality": float(max(primary_quality, lymph_quality)), "present": 1}

