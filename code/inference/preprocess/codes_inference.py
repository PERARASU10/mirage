from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from mirage import HierarchicalCodeEmbedder
from mirage.preprocess_utils import decompose_icd, decompose_ops, parse_code_tokens


def collect_text(surgery_text_data):
    if not isinstance(surgery_text_data, dict):
        return ""
    return "\n".join(
        [
            str(surgery_text_data.get("icd_codes", "") or ""),
            str(surgery_text_data.get("ops_codes", "") or ""),
            str(surgery_text_data.get("description", "") or ""),
        ]
    )


def vectorize_tokens(tokens, objects):
    token_to_idx = objects.get("token_to_idx", {})
    if not token_to_idx:
        return None
    dense = np.zeros((1, len(token_to_idx)), dtype=np.float32)
    for token in tokens:
        idx = token_to_idx.get(token)
        if idx is not None:
            dense[0, idx] += 1.0
    row_sum = dense.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return dense / row_sum


def get_codes_embedding(surgery_text_data, resources_path, device):
    resources_path = Path(resources_path)
    preproc = joblib.load(resources_path / "codes_preproc_objects.joblib")
    text = collect_text(surgery_text_data)
    tokens = []
    tokens.extend(parse_code_tokens(text, preproc["icd_pattern"], decompose_icd))
    tokens.extend(parse_code_tokens(text, preproc["ops_pattern"], decompose_ops))
    dense = vectorize_tokens(tokens, preproc)
    if dense is None or not np.any(dense):
        return {"embedding": torch.zeros(512, device=device), "quality": 0.0, "present": 0}

    checkpoint = torch.load(resources_path / "codes_hierarchical_embedder.pt", map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    input_dim = int(checkpoint.get("input_dim", dense.shape[1])) if isinstance(checkpoint, dict) else dense.shape[1]
    embed_dim = int(checkpoint.get("embed_dim", 512)) if isinstance(checkpoint, dict) else 512

    model = HierarchicalCodeEmbedder(input_dim=input_dim, embed_dim=embed_dim).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    with torch.no_grad():
        embedding = model.encode(torch.from_numpy(dense).float().to(device)).squeeze(0)
    quality = float(min(len(tokens) / 32.0, 1.0))
    return {"embedding": embedding.detach().cpu(), "quality": quality, "present": 1}
