from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .common import (
    apply_frequency_encoder,
    build_knn_graph_matrix,
    make_frequency_encoder,
    normalize_patient_id,
)


def read_text(path) -> str:
    with open(Path(path), "r", encoding="utf-8", errors="ignore") as handle:
        return handle.read()


def list_text_files(directory) -> List[Path]:
    return sorted(Path(directory).rglob("*.txt"))


def build_frequency_maps(df: pd.DataFrame, columns: Sequence[str]) -> Dict[str, Dict[str, float]]:
    maps: Dict[str, Dict[str, float]] = {}
    for column in columns:
        if column in df.columns:
            maps[column] = make_frequency_encoder(df[column])
        else:
            maps[column] = {}
    return maps


def apply_frequency_maps(df: pd.DataFrame, maps: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for column, encoder in maps.items():
        out[column] = apply_frequency_encoder(df[column], encoder) if column in df.columns else 0.0
    return out


def build_missing_indicators(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    missing = pd.DataFrame(index=df.index)
    for column in columns:
        missing[f"miss__{column}"] = df[column].isna().astype(float)
    return missing


def derive_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    followup = pd.to_numeric(df.get("days_to_last_information"), errors="coerce")
    cause = df.get("survival_status_with_cause", pd.Series([""] * len(df))).astype(str).str.lower()
    recurrence = df.get("recurrence", pd.Series([""] * len(df))).astype(str).str.lower()
    days_rec = pd.to_numeric(df.get("days_to_recurrence"), errors="coerce")

    surv = np.full(len(df), -1, dtype=np.int8)
    rec = np.full(len(df), -1, dtype=np.int8)
    surv[(cause.str.contains("tumor", na=False)) & (followup <= 1825)] = 0
    surv[(~cause.str.contains("tumor", na=False)) & (followup >= 1825)] = 1
    rec[(recurrence == "yes") & (days_rec <= 730)] = 1
    rec[(recurrence == "no") & (followup >= 730)] = 0
    return surv, rec


def derive_outcome_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    return derive_labels(df)


def fit_tabular_pipeline(features: pd.DataFrame) -> Dict[str, np.ndarray]:
    raw = features.astype(np.float32).copy()
    missing_mask = (~raw.isna()).astype(np.float32).values
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(raw.values)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed).astype(np.float32)
    return {
        "raw": raw.values.astype(np.float32),
        "imputed": imputed.astype(np.float32),
        "scaled": scaled,
        "mask": missing_mask.astype(np.float32),
        "imputer": imputer,
        "scaler": scaler,
    }


def build_patient_graph_features(
    clinical_features: np.ndarray,
    pathological_features: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    merged = np.concatenate([clinical_features, pathological_features], axis=1)
    return build_knn_graph_matrix(merged, k=k)


def decompose_icd(code: str) -> List[str]:
    code = code.strip().upper()
    if not code:
        return []
    code = code.replace(" ", "")
    parts = [code[:1]]
    if len(code) >= 3:
        parts.append(code[:3])
    if "." in code:
        parts.append(code)
    elif len(code) > 3:
        parts.append(code[:4])
    return list(dict.fromkeys(parts))


def decompose_ops(code: str) -> List[str]:
    code = code.strip().upper()
    if not code:
        return []
    code = code.replace(" ", "")
    parts = [code[:1]]
    if len(code) >= 5:
        parts.append(code[:5])
    if "." in code:
        parts.append(code)
    return list(dict.fromkeys(parts))


def parse_code_tokens(text: str, pattern: str, splitter: Callable[[str], List[str]]) -> List[str]:
    tokens: List[str] = []
    for match in re.findall(pattern, text or ""):
        tokens.extend(splitter(match))
    return tokens
