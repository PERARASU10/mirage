from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from mirage import TabularKANAutoencoder


def clean_margin(value):
    if value is None:
        return np.nan
    text = str(value).strip()
    if text.startswith("<"):
        try:
            return float(text[1:]) / 2.0
        except ValueError:
            return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _coerce_scalar(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def build_feature_row(data, feature_names, frequency_maps):
    row = {}
    for feature in feature_names:
        if feature.startswith("miss__"):
            base_feature = feature.replace("miss__", "", 1)
            row[feature] = 0.0 if _coerce_scalar(data.get(base_feature)) is not None else 1.0
        elif feature in frequency_maps:
            raw = _coerce_scalar(data.get(feature))
            row[feature] = frequency_maps[feature].get(str(raw), 0.0) if raw is not None else np.nan
        elif feature == "closest_resection_margin_in_cm":
            row[feature] = clean_margin(_coerce_scalar(data.get(feature)))
        else:
            raw = _coerce_scalar(data.get(feature))
            if feature in {
                "lymphovascular_invasion_L",
                "vascular_invasion_V",
                "perineural_invasion_Pn",
                "carcinoma_in_situ",
            }:
                row[feature] = {"yes": 1.0, "no": 0.0}.get(str(raw).lower(), np.nan) if raw is not None else np.nan
            else:
                row[feature] = pd.to_numeric(raw, errors="coerce")
    return pd.DataFrame([row], columns=feature_names)


def get_pathological_embedding(pathological_data, resources_path, device):
    resources_path = Path(resources_path)
    if not pathological_data:
        return {"embedding": torch.zeros(512, device=device), "quality": 0.0, "present": 0}

    objects = joblib.load(resources_path / "pathological_preproc_objects.joblib")
    feature_names = objects["feature_names"]
    frequency_maps = objects.get("frequency_maps", objects.get("cat_encoders", {}))
    frame = build_feature_row(pathological_data, feature_names, frequency_maps)
    raw = frame.to_numpy(dtype=np.float32)
    quality = float(np.isfinite(raw).mean()) if raw.size else 0.0
    imputer = objects.get("median_imputer", objects["imputer"])
    scaled = objects["scaler"].transform(imputer.transform(raw)).astype(np.float32)

    checkpoint = torch.load(resources_path / "pathological_kan_encoder.pt", map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    input_dim = int(checkpoint.get("input_dim", scaled.shape[1])) if isinstance(checkpoint, dict) else scaled.shape[1]
    embed_dim = int(checkpoint.get("embed_dim", 512)) if isinstance(checkpoint, dict) else 512

    model = TabularKANAutoencoder(input_dim=input_dim, embed_dim=embed_dim).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    with torch.no_grad():
        embedding = model.encode(torch.from_numpy(scaled).float().to(device)).squeeze(0)
    return {"embedding": embedding.detach().cpu(), "quality": quality, "present": 1}
