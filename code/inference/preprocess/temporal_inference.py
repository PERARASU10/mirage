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

from mirage import LSTMTemporalEncoder, RWKVTemporalEncoder


def _as_records(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return value["records"]
    return []


def make_time_bins(seq_len, window_days):
    return np.linspace(0, window_days, seq_len + 1)


def patient_to_matrix(frame, analytes, edges, seq_len):
    matrix = np.full((len(analytes), seq_len), np.nan, dtype=np.float32)
    for idx, analyte in enumerate(analytes):
        rows = frame[frame["analyte_name"] == analyte]
        if rows.empty:
            continue
        bins = np.digitize(rows["days_before_first_treatment"].to_numpy(), edges, right=False) - 1
        bins = np.clip(bins, 0, seq_len - 1)
        for bucket in range(seq_len):
            values = rows.loc[bins == bucket, "value"]
            if not values.empty:
                matrix[idx, bucket] = float(values.mean())
    return matrix


def physiology_fill(matrix, analytes, reference_ranges):
    filled = matrix.copy()
    for idx, analyte in enumerate(analytes):
        row = filled[idx]
        if np.isnan(row).all():
            ref = reference_ranges.get(analyte, {})
            mins = [item for item in [ref.get("male_min"), ref.get("female_min")] if item is not None]
            maxs = [item for item in [ref.get("male_max"), ref.get("female_max")] if item is not None]
            if mins and maxs:
                filled[idx, :] = (float(np.mean(mins)) + float(np.mean(maxs))) / 2.0
        else:
            missing = np.isnan(row)
            if missing.any() and (~missing).any():
                axis = np.arange(row.size)
                row[missing] = np.interp(axis[missing], axis[~missing], row[~missing])
                filled[idx] = row
    return filled


def get_temporal_embedding(blood_data, resources_path, device):
    records = _as_records(blood_data)
    resources_path = Path(resources_path)
    if not records:
        return {"embedding": torch.zeros(512, device=device), "quality": 0.0, "present": 0}

    objects = joblib.load(resources_path / "temporal_preproc_objects.joblib")
    analytes = objects["analytes"]
    seq_len = int(objects.get("seq_length", objects.get("seq_len", 16)))
    window_days = int(objects.get("window_days", 14))
    reference_ranges = objects.get("reference_ranges", {})

    frame = pd.DataFrame(records)
    for column in ["analyte_name", "value", "days_before_first_treatment"]:
        if column not in frame.columns:
            frame[column] = np.nan
    frame["analyte_name"] = frame["analyte_name"].astype(str).str.strip()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["days_before_first_treatment"] = pd.to_numeric(frame["days_before_first_treatment"], errors="coerce").fillna(0)

    matrix = patient_to_matrix(frame, analytes, make_time_bins(seq_len, window_days), seq_len)
    matrix = physiology_fill(matrix, analytes, reference_ranges)
    quality = float(np.isfinite(matrix).mean()) if matrix.size else 0.0

    flat = matrix.reshape(1, -1)
    flat = np.nan_to_num(flat, nan=0.0)
    flat = objects["knn_imputer"].transform(flat)
    flat = objects["scaler"].transform(flat).astype(np.float32)
    sequence = flat.reshape(1, len(analytes), seq_len).transpose(0, 2, 1)

    checkpoint_path = resources_path / "temporal_rwkv_encoder.pt"
    encoder_cls = RWKVTemporalEncoder
    if not checkpoint_path.exists():
        checkpoint_path = resources_path / "temporal_lstm_encoder.pt"
        encoder_cls = LSTMTemporalEncoder
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    input_dim = int(checkpoint.get("input_dim", len(analytes))) if isinstance(checkpoint, dict) else len(analytes)
    mode = checkpoint.get("mode", objects.get("mode", "rwkv")) if isinstance(checkpoint, dict) else objects.get("mode", "rwkv")
    if mode == "lstm":
        encoder_cls = LSTMTemporalEncoder

    model = encoder_cls(input_dim=input_dim, embed_dim=512).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    with torch.no_grad():
        embedding = model(torch.from_numpy(sequence).float().to(device)).squeeze(0)
    return {"embedding": embedding.detach().cpu(), "quality": quality, "present": 1}
