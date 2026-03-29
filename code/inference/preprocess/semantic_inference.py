from __future__ import annotations

from pathlib import Path
import re

import joblib
import numpy as np
import torch

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None


def clean_text(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value)).strip()


def merge_text_fields(text_data):
    if not isinstance(text_data, dict):
        return ""
    parts = [clean_text(text_data.get(key, "")) for key in ("history", "report", "description")]
    return "\n".join([part for part in parts if part]).strip()


def transformer_embedding(text, resources_path, device):
    model_path = Path(resources_path) / "Bio_ClinicalBERT"
    if AutoTokenizer is None or AutoModel is None or not model_path.exists():
        return None
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()
    with torch.no_grad():
        encoded = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = model(**encoded).last_hidden_state.squeeze(0)
        mask = encoded["attention_mask"].squeeze(0).unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=0) / mask.sum(dim=0).clamp(min=1.0)
    return pooled.detach().cpu().numpy()


def project_embedding(raw_embedding, preproc):
    vector = np.asarray(raw_embedding, dtype=np.float32).reshape(1, -1)
    projector = preproc.get("projector")
    if projector is not None and hasattr(projector, "transform"):
        try:
            projected = projector.transform(vector)
            vector = np.asarray(projected, dtype=np.float32)
        except Exception:
            pass
    vector = vector.reshape(-1)
    if vector.size >= 512:
        return vector[:512]
    return np.pad(vector, (0, 512 - vector.size))


def get_semantic_embedding(text_data, resources_path, device):
    resources_path = Path(resources_path)
    merged = merge_text_fields(text_data)
    if not merged:
        return {"embedding": torch.zeros(512, device=device), "quality": 0.0, "present": 0}

    preproc = {}
    preproc_path = resources_path / "text_semantic_preproc_objects.joblib"
    if preproc_path.exists():
        preproc = joblib.load(preproc_path)

    raw_embedding = transformer_embedding(merged, resources_path, device) if preproc.get("mode") == "transformer" else None
    if raw_embedding is None:
        vectorizer = preproc.get("tfidf_vectorizer", preproc.get("vectorizer"))
        if vectorizer is None:
            return {"embedding": torch.zeros(512, device=device), "quality": 0.0, "present": 0}
        transformed = vectorizer.transform([merged])
        if preproc.get("projector") is not None and hasattr(preproc["projector"], "transform"):
            transformed = preproc["projector"].transform(transformed)
            raw_embedding = np.asarray(transformed, dtype=np.float32).reshape(-1)
        else:
            raw_embedding = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed).reshape(-1)

    embedding = project_embedding(raw_embedding, preproc)
    quality = float(min(len(merged.split()) / 256.0, 1.0))
    return {"embedding": torch.from_numpy(embedding.astype(np.float32)).to(device), "quality": quality, "present": 1}
