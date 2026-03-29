import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mirage.common import (
    build_knn_graph_matrix,
    ensure_directory,
    save_embedding_h5,
    save_preprocessed_h5,
    set_seed,
)
from mirage.config import CLINICAL_JSON, OUT_DIR
from mirage.models import TabularKANAutoencoder
from mirage.preprocess_utils import (
    apply_frequency_maps,
    build_frequency_maps,
    build_missing_indicators,
    derive_labels,
    fit_tabular_pipeline,
)


DROP_COLS = [
    "survival_status",
    "survival_status_with_cause",
    "days_to_last_information",
    "days_to_recurrence",
    "recurrence",
    "patient_id",
]

NUMERIC_COLS = [
    "year_of_initial_diagnosis",
    "age_at_initial_diagnosis",
    "days_to_first_treatment",
    "days_to_progress_1",
    "days_to_progress_2",
    "days_to_metastasis_1",
    "days_to_metastasis_2",
    "days_to_metastasis_3",
    "days_to_metastasis_4",
]

CATEGORICAL_COLS = [
    "sex",
    "smoking_status",
    "primarily_metastasis",
    "first_treatment_intent",
    "first_treatment_modality",
    "adjuvant_treatment_intent",
    "adjuvant_radiotherapy",
    "adjuvant_radiotherapy_modality",
    "adjuvant_systemic_therapy",
    "adjuvant_systemic_therapy_modality",
    "adjuvant_radiochemotherapy",
    "progress_1",
    "progress_2",
    "metastasis_1_locations",
    "metastasis_2_locations",
    "metastasis_3_locations",
    "metastasis_4_locations",
]


def _series_or_nan(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(np.nan, index=df.index)


def build_feature_frame(df: pd.DataFrame):
    numeric = pd.DataFrame(index=df.index)
    for column in NUMERIC_COLS:
        numeric[column] = pd.to_numeric(_series_or_nan(df, column), errors="coerce")

    frequency_maps = build_frequency_maps(df, CATEGORICAL_COLS)
    categorical = apply_frequency_maps(df, frequency_maps)
    observed = pd.concat([numeric, df[CATEGORICAL_COLS].copy() if set(CATEGORICAL_COLS).issubset(df.columns) else df.reindex(columns=CATEGORICAL_COLS)], axis=1)
    base_features = pd.concat([numeric, categorical], axis=1)
    missing = build_missing_indicators(observed, list(observed.columns))
    features = pd.concat([base_features, missing], axis=1)
    return features, base_features, missing, frequency_maps, observed


def train_embedding_model(features: np.ndarray, args):
    if not TORCH_AVAILABLE:
        return None, features.astype(np.float32)

    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = TabularKANAutoencoder(input_dim=features.shape[1], embed_dim=512, dropout=0.1).to(device)
    dataset = TensorDataset(torch.from_numpy(features).float())
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
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
        embeddings = model.encode(torch.from_numpy(features).float().to(device)).cpu().numpy().astype(np.float32)
    return model, embeddings


def normalize_embedding_dim(embeddings: np.ndarray, target_dim: int = 512) -> np.ndarray:
    if embeddings.shape[1] == target_dim:
        return embeddings.astype(np.float32)
    if embeddings.shape[1] > target_dim:
        return embeddings[:, :target_dim].astype(np.float32)
    pad_width = target_dim - embeddings.shape[1]
    return np.pad(embeddings.astype(np.float32), ((0, 0), (0, pad_width)), mode="constant")


def parse_args():
    parser = argparse.ArgumentParser(description="MIRAGE-Net clinical preprocessing")
    parser.add_argument("--input-json", type=Path, default=CLINICAL_JSON)
    parser.add_argument("--outdir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--graph-k", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_directory(args.outdir)
    set_seed(args.seed)

    df = pd.read_json(args.input_json)
    if "patient_id" not in df.columns:
        raise ValueError("clinical_data.json must contain patient_id")

    df["patient_id"] = df["patient_id"].apply(lambda value: str(int(value)).zfill(3))
    df = df.drop_duplicates(subset=["patient_id"]).set_index("patient_id").sort_index()

    surv_label, rec_label = derive_labels(df.reset_index())
    features, base_features, missing, frequency_maps, observed = build_feature_frame(df)
    pipeline = fit_tabular_pipeline(features)
    graph_adjacency = build_knn_graph_matrix(pipeline["scaled"], k=args.graph_k)
    quality = (1.0 - observed.isna().mean(axis=1).values).astype(np.float32)

    model, embeddings = train_embedding_model(pipeline["scaled"], args)
    embeddings = normalize_embedding_dim(embeddings)

    save_preprocessed_h5(
        args.outdir / "clinical_preprocessed.h5",
        df.index.tolist(),
        features.columns.tolist(),
        pipeline["raw"],
        pipeline["imputed"],
        pipeline["mask"],
        extra={
            "graph_adjacency": graph_adjacency,
            "surv_5yr_label": surv_label,
            "rec_2yr_label": rec_label,
            "quality_score": quality,
            "base_feature_names": base_features.columns.tolist(),
        },
    )

    save_embedding_h5(
        args.outdir / "clinical_embedding_512.h5",
        df.index.tolist(),
        embeddings,
        quality_scores=quality,
        extra={
            "surv_5yr_label": surv_label,
            "rec_2yr_label": rec_label,
        },
    )

    joblib.dump(
        {
            "feature_names": features.columns.tolist(),
            "base_feature_names": base_features.columns.tolist(),
            "frequency_maps": frequency_maps,
            "cat_encoders": frequency_maps,
            "imputer": pipeline["imputer"],
            "median_imputer": pipeline["imputer"],
            "scaler": pipeline["scaler"],
            "graph_k": args.graph_k,
            "drop_cols": DROP_COLS,
            "numeric_cols": NUMERIC_COLS,
            "categorical_cols": CATEGORICAL_COLS,
        },
        args.outdir / "clinical_preproc_objects.joblib",
    )

    if model is not None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": pipeline["scaled"].shape[1],
                "embed_dim": 512,
            },
            args.outdir / "clinical_kan_encoder.pt",
        )

    print(f"clinical patients: {len(df)}")
    print(f"n_features total: {features.shape[1]}")
    print(f"embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
