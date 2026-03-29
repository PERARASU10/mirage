import argparse
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
from mirage.config import OUT_DIR, PATHOLOGICAL_JSON
from mirage.models import TabularKANAutoencoder
from mirage.preprocess_utils import (
    apply_frequency_maps,
    build_frequency_maps,
    build_missing_indicators,
    fit_tabular_pipeline,
)


BINARY_COLUMNS = [
    "lymphovascular_invasion_L",
    "vascular_invasion_V",
    "perineural_invasion_Pn",
    "carcinoma_in_situ",
]

CATEGORICAL_COLUMNS = [
    "primary_tumor_site",
    "pT_stage",
    "pN_stage",
    "grading",
    "hpv_association_p16",
    "perinodal_invasion",
    "resection_status",
    "resection_status_carcinoma_in_situ",
    "histologic_type",
]

NUMERIC_COLUMNS = [
    "number_of_positive_lymph_nodes",
    "number_of_resected_lymph_nodes",
    "closest_resection_margin_in_cm",
    "infiltration_depth_in_mm",
]


def clean_margin(value):
    if pd.isna(value):
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


def _series_or_nan(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series(np.nan, index=df.index)


def build_feature_frame(df: pd.DataFrame):
    base = pd.DataFrame(index=df.index)
    base["closest_resection_margin_in_cm"] = df["closest_resection_margin_in_cm"].apply(clean_margin)
    for column in NUMERIC_COLUMNS:
        if column == "closest_resection_margin_in_cm":
            continue
        base[column] = pd.to_numeric(_series_or_nan(df, column), errors="coerce")

    for column in BINARY_COLUMNS:
        base[column] = (
            _series_or_nan(df, column)
            .astype(str)
            .str.lower()
            .map({"yes": 1.0, "no": 0.0})
        )

    frequency_maps = build_frequency_maps(df, CATEGORICAL_COLUMNS)
    categorical = apply_frequency_maps(df, frequency_maps)
    observed = pd.concat(
        [
            base,
            df[CATEGORICAL_COLUMNS].copy() if set(CATEGORICAL_COLUMNS).issubset(df.columns) else df.reindex(columns=CATEGORICAL_COLUMNS),
        ],
        axis=1,
    )
    base_features = pd.concat([base, categorical], axis=1)
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
    loader = DataLoader(TensorDataset(torch.from_numpy(features).float()), batch_size=args.bs, shuffle=True)
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
    parser = argparse.ArgumentParser(description="MIRAGE-Net pathological preprocessing")
    parser.add_argument("--input-json", type=Path, default=PATHOLOGICAL_JSON)
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
        raise ValueError("pathological_data.json must contain patient_id")
    df["patient_id"] = df["patient_id"].apply(lambda value: str(int(value)).zfill(3))
    df = df.drop_duplicates(subset=["patient_id"]).set_index("patient_id").sort_index()

    features, base_features, missing, frequency_maps, observed = build_feature_frame(df)
    pipeline = fit_tabular_pipeline(features)
    graph_adjacency = build_knn_graph_matrix(pipeline["scaled"], k=args.graph_k)
    quality = (1.0 - observed.isna().mean(axis=1).values).astype(np.float32)

    model, embeddings = train_embedding_model(pipeline["scaled"], args)
    embeddings = normalize_embedding_dim(embeddings)

    save_preprocessed_h5(
        args.outdir / "pathological_preprocessed.h5",
        df.index.tolist(),
        features.columns.tolist(),
        pipeline["raw"],
        pipeline["imputed"],
        pipeline["mask"],
        extra={
            "graph_adjacency": graph_adjacency,
            "quality_score": quality,
            "base_feature_names": base_features.columns.tolist(),
        },
    )

    save_embedding_h5(
        args.outdir / "pathological_embedding_512.h5",
        df.index.tolist(),
        embeddings,
        quality_scores=quality,
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
            "binary_columns": BINARY_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
            "numeric_columns": NUMERIC_COLUMNS,
        },
        args.outdir / "pathological_preproc_objects.joblib",
    )

    if model is not None:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": pipeline["scaled"].shape[1],
                "embed_dim": 512,
            },
            args.outdir / "pathological_kan_encoder.pt",
        )

    print(f"pathological patients: {len(df)}")
    print(f"n_features total: {features.shape[1]}")
    print(f"embedding shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
