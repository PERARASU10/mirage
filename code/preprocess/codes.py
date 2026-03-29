import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from sklearn.decomposition import PCA, TruncatedSVD

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mirage.common import ensure_directory, normalize_patient_id, save_embedding_h5, set_seed
from mirage.config import ICD_CODES_DIR, OPS_CODES_DIR, OUT_DIR
from mirage.models import HierarchicalCodeEmbedder
from mirage.preprocess_utils import decompose_icd, decompose_ops, list_text_files, parse_code_tokens, read_text


ICD_PATTERN = r"\[([A-Z]\d{2}[\.\d]*)"
OPS_PATTERN = r"\[(\d-\d{3}[\.\d]*)"


def extract_patient_id(path: Path) -> str:
    match = re.search(r"(\d{1,5})", path.stem)
    if match:
        return str(int(match.group(1))).zfill(3)
    return normalize_patient_id(path.stem)


def build_token_bags(icd_dir: Path, ops_dir: Path):
    bags = defaultdict(list)
    for path in list_text_files(icd_dir):
        bags[extract_patient_id(path)].extend(parse_code_tokens(read_text(path), ICD_PATTERN, decompose_icd))
    for path in list_text_files(ops_dir):
        bags[extract_patient_id(path)].extend(parse_code_tokens(read_text(path), OPS_PATTERN, decompose_ops))
    return bags


def build_token_matrix(bags):
    patient_ids = sorted(bags)
    vocabulary = sorted({token for tokens in bags.values() for token in tokens})
    token_to_idx = {token: index for index, token in enumerate(vocabulary)}
    matrix = np.zeros((len(patient_ids), len(vocabulary)), dtype=np.float32)
    max_tokens = 1
    for row_index, patient_id in enumerate(patient_ids):
        tokens = bags[patient_id]
        max_tokens = max(max_tokens, len(tokens))
        for token in tokens:
            matrix[row_index, token_to_idx[token]] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    matrix = matrix / row_sums
    return patient_ids, vocabulary, token_to_idx, matrix, max_tokens


def train_embedding_model(matrix: np.ndarray, args):
    if not TORCH_AVAILABLE:
        if matrix.shape[0] <= 1 or matrix.shape[1] <= 1 or not matrix.size:
            embeddings = matrix
        else:
            n_components = min(512, max(2, min(matrix.shape[0], matrix.shape[1])))
            try:
                reducer = PCA(n_components=min(n_components, matrix.shape[1]), random_state=42)
                embeddings = reducer.fit_transform(matrix)
            except Exception:
                reducer = TruncatedSVD(n_components=min(n_components, matrix.shape[1]), random_state=42)
                embeddings = reducer.fit_transform(matrix)
        if embeddings.shape[1] < 512:
            embeddings = np.pad(embeddings, ((0, 0), (0, 512 - embeddings.shape[1])), mode="constant")
        elif embeddings.shape[1] > 512:
            embeddings = embeddings[:, :512]
        return None, embeddings.astype(np.float32)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = HierarchicalCodeEmbedder(input_dim=matrix.shape[1], embed_dim=512, dropout=0.1).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(matrix).float()), batch_size=args.bs, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    for _ in range(args.epochs):
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction, _ = model(batch)
            loss = criterion(reconstruction, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(torch.from_numpy(matrix).float().to(device)).cpu().numpy().astype(np.float32)
    return model, embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="MIRAGE-Net ICD/OPS code preprocessing")
    parser.add_argument("--icd-dir", type=Path, default=ICD_CODES_DIR)
    parser.add_argument("--ops-dir", type=Path, default=OPS_CODES_DIR)
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_directory(args.outdir)
    set_seed(args.seed)

    bags = build_token_bags(args.icd_dir, args.ops_dir)
    patient_ids, vocabulary, token_to_idx, matrix, max_tokens = build_token_matrix(bags)
    quality = (matrix.sum(axis=1) > 0).astype(np.float32)

    model, embeddings = train_embedding_model(matrix, args)

    save_embedding_h5(
        args.outdir / "codes_embedding_512.h5",
        patient_ids,
        embeddings,
        quality_scores=quality,
        extra={
            "token_count": matrix.sum(axis=1),
        },
    )

    joblib.dump(
        {
            "vocabulary": vocabulary,
            "token_to_idx": token_to_idx,
            "max_tokens": max_tokens,
            "icd_pattern": ICD_PATTERN,
            "ops_pattern": OPS_PATTERN,
            "patient_ids": patient_ids,
        },
        args.outdir / "codes_preproc_objects.joblib",
    )

    if model is not None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": matrix.shape[1],
                "embed_dim": 512,
            },
            args.outdir / "codes_hierarchical_embedder.pt",
        )

    print(f"codes patients: {len(patient_ids)}")
    print(f"vocab size: {len(vocabulary)}")
    print(f"embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
