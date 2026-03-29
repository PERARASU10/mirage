import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mirage.common import ensure_directory, load_json_file, save_embedding_h5, set_seed, write_json_file
from mirage.config import BLOOD_JSON, BLOOD_REF_JSON, OUT_DIR
from mirage.models import LSTMTemporalEncoder, RWKVTemporalEncoder


TIME_WINDOW_DAYS = 14
SEQ_LENGTH = 16
MIN_ANALYTE_PATIENTS = 3
KNN_NEIGHBORS = 8


def load_reference_ranges(records):
    reference = {}
    if isinstance(records, dict):
        records = list(records.values())
    for record in records or []:
        name = record.get("analyte_name")
        if name:
            reference[str(name).strip()] = {
                "male_min": record.get("normal_male_min", record.get("male_min")),
                "male_max": record.get("normal_male_max", record.get("male_max")),
                "female_min": record.get("normal_female_min", record.get("female_min")),
                "female_max": record.get("normal_female_max", record.get("female_max")),
                "unit": record.get("unit"),
                "group": record.get("group"),
            }
    return reference


def normalize_blood_dataframe(records):
    df = pd.DataFrame(records or [])
    if df.empty:
        return df
    for column in ["patient_id", "analyte_name", "value", "days_before_first_treatment"]:
        if column not in df.columns:
            df[column] = np.nan
    df["patient_id"] = df["patient_id"].apply(lambda value: str(value).zfill(3))
    df["analyte_name"] = df["analyte_name"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["days_before_first_treatment"] = pd.to_numeric(df["days_before_first_treatment"], errors="coerce")
    df["days_before_first_treatment"] = df["days_before_first_treatment"].fillna(0).clip(0, TIME_WINDOW_DAYS).astype(int)
    return df


def make_time_bins(seq_len=SEQ_LENGTH, window_days=TIME_WINDOW_DAYS):
    edges = np.linspace(0, window_days, seq_len + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers


def patient_to_matrix(patient_df, analytes, edges, seq_len):
    matrix = np.full((len(analytes), seq_len), np.nan, dtype=np.float32)
    for analyte_index, analyte in enumerate(analytes):
        analyte_df = patient_df[patient_df["analyte_name"] == analyte]
        if analyte_df.empty:
            continue
        bins = np.digitize(analyte_df["days_before_first_treatment"].values, edges, right=False) - 1
        bins = np.clip(bins, 0, seq_len - 1)
        for bin_index in range(seq_len):
            values = analyte_df.loc[bins == bin_index, "value"]
            if not values.empty:
                matrix[analyte_index, bin_index] = float(values.mean())
    return matrix


def physiology_fill(matrix, analytes, reference_ranges):
    filled = matrix.copy()
    for row_index, analyte in enumerate(analytes):
        row = filled[row_index]
        if np.isnan(row).all():
            ref = reference_ranges.get(analyte)
            if ref:
                mins = [value for value in [ref.get("male_min"), ref.get("female_min")] if value is not None]
                maxs = [value for value in [ref.get("male_max"), ref.get("female_max")] if value is not None]
                if mins and maxs:
                    midpoint = (float(np.mean(mins)) + float(np.mean(maxs))) / 2.0
                    filled[row_index, :] = midpoint
                elif mins:
                    filled[row_index, :] = float(np.mean(mins))
                elif maxs:
                    filled[row_index, :] = float(np.mean(maxs))
        else:
            nan_mask = np.isnan(row)
            if nan_mask.any() and (~nan_mask).any():
                x = np.arange(row.size)
                row[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], row[~nan_mask])
                filled[row_index] = row
    return filled


def _pad_or_trim(array, target_dim=512):
    if array.shape[1] < target_dim:
        return np.pad(array, ((0, 0), (0, target_dim - array.shape[1])), mode="constant")
    if array.shape[1] > target_dim:
        return array[:, :target_dim]
    return array


def train_encoder(train_inputs, mode, device, epochs, batch_size, lr, input_dim):
    if not TORCH_AVAILABLE:
        return None, None

    if mode == "rwkv":
        model = RWKVTemporalEncoder(input_dim=input_dim, d_model=64, n_layers=3, embed_dim=512, dropout=0.1).to(device)
    else:
        model = LSTMTemporalEncoder(input_dim=input_dim, hidden_dim=128, n_layers=2, embed_dim=512, dropout=0.1).to(device)

    class _Dataset(Dataset):
        def __init__(self, tensor):
            self.tensor = tensor

        def __len__(self):
            return self.tensor.size(0)

        def __getitem__(self, index):
            return self.tensor[index]

    tensor = torch.from_numpy(train_inputs).float().to(device)
    loader = DataLoader(_Dataset(tensor), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for batch in loader:
            reconstruction = model(batch)
            batch_target = batch.reshape(batch.size(0), -1)
            batch_target = torch.from_numpy(_pad_or_trim(batch_target.detach().cpu().numpy())).float().to(device)
            loss = criterion(reconstruction, batch_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        embeddings = model(tensor).cpu().numpy().astype(np.float32)
    return model, embeddings


def fallback_temporal_embedding(flattened_inputs):
    n_samples, n_features = flattened_inputs.shape
    n_components = min(512, max(2, min(n_samples, n_features)))
    if n_components >= 2 and n_samples > 1 and n_features > 1:
        try:
            reducer = PCA(n_components=n_components, random_state=42)
            embeddings = reducer.fit_transform(flattened_inputs)
        except Exception:
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
            embeddings = reducer.fit_transform(flattened_inputs)
    else:
        embeddings = flattened_inputs[:, : min(512, n_features)]
    return _pad_or_trim(embeddings.astype(np.float32))


def parse_args():
    parser = argparse.ArgumentParser(description="MIRAGE-Net temporal preprocessing")
    parser.add_argument("--blood-json", type=Path, default=BLOOD_JSON)
    parser.add_argument("--ref-json", type=Path, default=BLOOD_REF_JSON)
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    parser.add_argument("--mode", choices=["rwkv", "lstm"], default="rwkv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_directory(args.outdir)
    set_seed(args.seed)

    records = load_json_file(args.blood_json, default=[])
    ref_records = load_json_file(args.ref_json, default=[])
    df = normalize_blood_dataframe(records)
    reference_ranges = load_reference_ranges(ref_records)

    if df.empty:
        raise ValueError("blood_data.json did not contain any records")

    patient_ids = sorted(df["patient_id"].unique().tolist())
    analyte_counts = df.groupby("analyte_name")["patient_id"].nunique().sort_values(ascending=False)
    analytes = analyte_counts[analyte_counts >= MIN_ANALYTE_PATIENTS].index.tolist()
    for analyte in reference_ranges.keys():
        if analyte not in analytes:
            analytes.append(analyte)
    analytes = sorted(dict.fromkeys(analytes))

    edges, centers = make_time_bins()
    matrices = np.full((len(patient_ids), len(analytes), SEQ_LENGTH), np.nan, dtype=np.float32)
    for patient_index, patient_id in enumerate(patient_ids):
        patient_df = df[df["patient_id"] == patient_id]
        matrices[patient_index] = patient_to_matrix(patient_df, analytes, edges, SEQ_LENGTH)

    quality = ((~np.isnan(matrices)).sum(axis=(1, 2)) / float(len(analytes) * SEQ_LENGTH)).astype(np.float32)

    filled = np.stack([physiology_fill(matrix, analytes, reference_ranges) for matrix in matrices], axis=0)
    flattened = filled.reshape(len(patient_ids), -1)

    knn_imputer = KNNImputer(n_neighbors=min(KNN_NEIGHBORS, max(2, len(patient_ids) - 1)))
    imputed = knn_imputer.fit_transform(flattened)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed).astype(np.float32)
    sequence_inputs = scaled.reshape(len(patient_ids), len(analytes), SEQ_LENGTH).transpose(0, 2, 1)

    device = args.device
    if device == "auto":
        device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

    if TORCH_AVAILABLE:
        model, embeddings = train_encoder(
            sequence_inputs,
            args.mode,
            device,
            args.epochs,
            args.bs,
            args.lr,
            len(analytes),
        )
        if model is not None:
            model_name = "temporal_rwkv_encoder.pt" if args.mode == "rwkv" else "temporal_lstm_encoder.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mode": args.mode,
                    "input_dim": len(analytes),
                },
                args.outdir / model_name,
            )
        else:
            embeddings = fallback_temporal_embedding(sequence_inputs.reshape(len(patient_ids), -1))
    else:
        embeddings = fallback_temporal_embedding(sequence_inputs.reshape(len(patient_ids), -1))

    save_embedding_h5(
        args.outdir / "temporal_embedding_512.h5",
        patient_ids,
        embeddings,
        quality_scores=quality,
        extra={
            "analytes": analytes,
            "time_centers": centers,
        },
    )

    joblib.dump(
        {
            "analytes": analytes,
            "time_edges": edges,
            "time_centers": centers,
            "seq_len": SEQ_LENGTH,
            "seq_length": SEQ_LENGTH,
            "window_days": TIME_WINDOW_DAYS,
            "mode": args.mode,
            "knn_imputer": knn_imputer,
            "scaler": scaler,
            "reference_ranges": reference_ranges,
        },
        args.outdir / "temporal_preproc_objects.joblib",
    )

    write_json_file(
        args.outdir / "temporal_preproc_summary.json",
        {
            "n_patients": len(patient_ids),
            "n_analytes": len(analytes),
            "seq_len": SEQ_LENGTH,
            "mode": args.mode,
            "embedding_shape": list(embeddings.shape),
        },
    )

    print(f"temporal patients: {len(patient_ids)}")
    print(f"n_analytes total: {len(analytes)}")
    print(f"embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
