from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

try:
    import torch
except Exception:
    torch = None


MODALITY_ORDER = [
    "clinical",
    "pathological",
    "temporal",
    "semantic",
    "codes",
    "spatial",
]
MODALITY_INDEX = {name: idx for idx, name in enumerate(MODALITY_ORDER)}


def ensure_directory(path):
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def normalize_patient_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip()
    if not text:
        return ""
    match = re.search(r"(\d+)", text)
    if match:
        return str(int(match.group(1))).zfill(3)
    return text


def extract_patient_id_from_name(value: str) -> str:
    return normalize_patient_id(Path(value).stem)


def load_json_file(path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return [] if default is None else default
    with open(target, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(path, content: Any) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(to_serializable(content), handle, indent=2)


def make_frequency_encoder(series: pd.Series) -> Dict[str, float]:
    clean = series.fillna("<<MISSING>>").astype(str)
    counts = clean.value_counts(normalize=True)
    return {str(key): float(value) for key, value in counts.items()}


def apply_frequency_encoder(series: pd.Series, encoder: Dict[str, float]) -> pd.Series:
    clean = series.fillna("<<MISSING>>").astype(str)
    return clean.map(lambda value: encoder.get(value, 0.0)).astype(float)


def _write_string_dataset(handle: h5py.File, key: str, values: Sequence[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    handle.create_dataset(key, data=np.asarray(list(values), dtype=object), dtype=dtype)


def save_embedding_h5(
    path,
    patient_ids: Sequence[str],
    embeddings: np.ndarray,
    quality_scores: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    quality_scores = (
        np.asarray(quality_scores, dtype=np.float32)
        if quality_scores is not None
        else np.ones((len(patient_ids),), dtype=np.float32)
    )
    extra = extra or {}
    with h5py.File(target, "w") as handle:
        _write_string_dataset(handle, "patient_id", [normalize_patient_id(pid) for pid in patient_ids])
        handle.create_dataset("embedding_512", data=embeddings)
        handle.create_dataset("quality_score", data=quality_scores)
        for key, value in extra.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
                _write_string_dataset(handle, key, [str(item) for item in value])
            else:
                handle.create_dataset(key, data=np.asarray(value))


def save_preprocessed_h5(
    path,
    patient_ids: Sequence[str],
    feature_names: Sequence[str],
    features_raw: np.ndarray,
    features_imputed: np.ndarray,
    missing_mask: np.ndarray,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    target = Path(path)
    ensure_directory(target.parent)
    extra = extra or {}
    with h5py.File(target, "w") as handle:
        _write_string_dataset(handle, "patient_id", [normalize_patient_id(pid) for pid in patient_ids])
        _write_string_dataset(handle, "feature_names", [str(name) for name in feature_names])
        handle.create_dataset("features_raw", data=np.asarray(features_raw, dtype=np.float32))
        handle.create_dataset("features_imputed", data=np.asarray(features_imputed, dtype=np.float32))
        handle.create_dataset("missing_mask", data=np.asarray(missing_mask, dtype=np.float32))
        for key, value in extra.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
                _write_string_dataset(handle, key, [str(item) for item in value])
            else:
                handle.create_dataset(key, data=np.asarray(value))


def _load_h5_dataset(handle: h5py.File, candidates: Sequence[str]) -> Optional[np.ndarray]:
    for name in candidates:
        if name in handle:
            return np.asarray(handle[name])
    return None


def load_embedding_h5(path) -> Dict[str, Any]:
    target = Path(path)
    with h5py.File(target, "r") as handle:
        raw_patient_ids = _load_h5_dataset(handle, ["patient_id", "patient_ids", "patient_index_map"])
        if raw_patient_ids is None:
            raise RuntimeError(f"{target} is missing patient_id.")
        patient_ids = [normalize_patient_id(item) for item in raw_patient_ids.tolist()]
        embeddings = _load_h5_dataset(
            handle,
            [
                "embedding_512",
                "embedding_mean_512",
                "text_combined_embedding_512",
                "features_mean",
                "embedding",
                "embeddings",
            ],
        )
        if embeddings is None:
            raise RuntimeError(f"{target} is missing an embedding dataset.")
        quality = _load_h5_dataset(handle, ["quality_score", "quality", "quality_scores"])
        if quality is None:
            quality = np.ones((len(patient_ids),), dtype=np.float32)
        quality = np.asarray(quality, dtype=np.float32).reshape(-1)
        result = {
            "patient_ids": patient_ids,
            "embeddings": np.asarray(embeddings, dtype=np.float32),
            "quality": quality,
        }
        for key in ["surv_5yr_label", "rec_2yr_label", "n_patches"]:
            if key in handle:
                result[key] = np.asarray(handle[key])
        return result


def align_modalities(h5_paths: Dict[str, Any]) -> Dict[str, Any]:
    clinical = load_embedding_h5(h5_paths["clinical"])
    master_ids = clinical["patient_ids"]
    n_patients = len(master_ids)
    emb_stack = np.zeros((n_patients, len(MODALITY_ORDER), 512), dtype=np.float32)
    quality = np.zeros((n_patients, len(MODALITY_ORDER)), dtype=np.float32)
    present_mask = np.zeros((n_patients, len(MODALITY_ORDER)), dtype=np.int8)

    id_to_row = {pid: idx for idx, pid in enumerate(master_ids)}
    for modality in MODALITY_ORDER:
        loaded = load_embedding_h5(h5_paths[modality])
        modality_index = MODALITY_INDEX[modality]
        for pid, emb, q in zip(loaded["patient_ids"], loaded["embeddings"], loaded["quality"]):
            row = id_to_row.get(pid)
            if row is None:
                continue
            emb_stack[row, modality_index] = emb.astype(np.float32)
            quality[row, modality_index] = float(q)
            present_mask[row, modality_index] = 1

    return {
        "patient_ids": master_ids,
        "emb_stack": emb_stack,
        "quality": quality,
        "present_mask": present_mask,
        "surv_5yr_label": np.asarray(clinical.get("surv_5yr_label", np.full(n_patients, -1)), dtype=np.int8),
        "rec_2yr_label": np.asarray(clinical.get("rec_2yr_label", np.full(n_patients, -1)), dtype=np.int8),
    }


def make_stratification_labels(surv_labels: np.ndarray, rec_labels: np.ndarray) -> np.ndarray:
    combined = np.full(len(surv_labels), -1, dtype=np.int16)
    for idx, (surv_value, rec_value) in enumerate(zip(surv_labels, rec_labels)):
        if surv_value != -1 and rec_value != -1:
            combined[idx] = int(surv_value) * 2 + int(rec_value)
        elif surv_value != -1:
            combined[idx] = int(surv_value) + 4
        elif rec_value != -1:
            combined[idx] = int(rec_value) + 6
    return combined


def build_cv_splits(
    surv_labels: np.ndarray,
    rec_labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    labels = make_stratification_labels(surv_labels, rec_labels)
    indices = np.arange(len(labels))
    fold_assignment = np.full(len(labels), -1, dtype=np.int16)

    known_mask = labels != -1
    known_indices = indices[known_mask]
    known_labels = labels[known_mask].copy()
    if known_indices.size:
        counts = pd.Series(known_labels).value_counts()
        rare_labels = counts[counts < n_splits].index.tolist()
        for rare_label in rare_labels:
            label_mask = known_labels == rare_label
            if rare_label in (4, 5):
                known_labels[label_mask] = 0 if rare_label == 4 else 2
            elif rare_label in (6, 7):
                known_labels[label_mask] = 0 if rare_label == 6 else 1

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (_, val_pos) in enumerate(splitter.split(known_indices, known_labels)):
            fold_assignment[known_indices[val_pos]] = fold_idx

    unknown_indices = indices[~known_mask].copy()
    if unknown_indices.size:
        rng = np.random.default_rng(seed)
        rng.shuffle(unknown_indices)
        for offset, idx in enumerate(unknown_indices):
            fold_assignment[idx] = offset % n_splits

    splits = []
    for fold_idx in range(n_splits):
        val_idx = indices[fold_assignment == fold_idx]
        train_idx = indices[fold_assignment != fold_idx]
        splits.append((train_idx, val_idx))
    return splits


def build_knn_graph_matrix(features: np.ndarray, k: int = 10) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    n_samples = features.shape[0]
    if n_samples == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if n_samples == 1:
        return np.ones((1, 1), dtype=np.float32)

    model = NearestNeighbors(n_neighbors=min(k + 1, n_samples), metric="euclidean")
    model.fit(features)
    distances, indices = model.kneighbors(features)
    sigma = float(np.median(distances[:, 1:])) if distances.shape[1] > 1 else 1.0
    sigma = max(sigma, 1e-3)

    adjacency = np.eye(n_samples, dtype=np.float32)
    for row, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for distance, neighbor in zip(row_distances[1:], row_indices[1:]):
            weight = float(np.exp(-distance / sigma))
            adjacency[row, neighbor] = max(adjacency[row, neighbor], weight)
            adjacency[neighbor, row] = max(adjacency[neighbor, row], weight)

    row_sums = adjacency.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return adjacency / row_sums


def compute_binary_metrics(
    targets: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    targets_array = np.asarray(targets, dtype=np.int8)
    probs_array = np.asarray(probabilities, dtype=np.float32)
    if targets_array.size == 0:
        return {
            "f1": 0.0,
            "auc": 0.0,
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "brier": 0.0,
            "confusion_matrix": np.zeros((2, 2), dtype=np.int64),
            "roc_curve": (
                np.asarray([0.0, 1.0], dtype=np.float32),
                np.asarray([0.0, 1.0], dtype=np.float32),
                np.asarray([np.inf, 0.0], dtype=np.float32),
            ),
            "calibration_curve": (
                np.asarray([0.0], dtype=np.float32),
                np.asarray([0.0], dtype=np.float32),
            ),
        }

    predictions = (probs_array >= threshold).astype(np.int8)
    if len(np.unique(targets_array)) > 1:
        auc = float(roc_auc_score(targets_array, probs_array))
        fpr, tpr, roc_thresholds = roc_curve(targets_array, probs_array)
    else:
        auc = 0.0
        fpr = np.asarray([0.0, 1.0], dtype=np.float32)
        tpr = np.asarray([0.0, 1.0], dtype=np.float32)
        roc_thresholds = np.asarray([np.inf, 0.0], dtype=np.float32)
    prob_true, prob_pred = calibration_curve(
        targets_array,
        probs_array,
        n_bins=min(10, max(2, len(targets_array))),
        strategy="uniform",
    )
    return {
        "f1": float(f1_score(targets_array, predictions, zero_division=0)),
        "auc": auc,
        "acc": float(accuracy_score(targets_array, predictions)),
        "precision": float(precision_score(targets_array, predictions, zero_division=0)),
        "recall": float(recall_score(targets_array, predictions, zero_division=0)),
        "brier": float(brier_score_loss(targets_array, probs_array)),
        "confusion_matrix": confusion_matrix(targets_array, predictions, labels=[0, 1]).astype(np.int64),
        "roc_curve": (
            np.asarray(fpr, dtype=np.float32),
            np.asarray(tpr, dtype=np.float32),
            np.asarray(roc_thresholds, dtype=np.float32),
        ),
        "calibration_curve": (
            np.asarray(prob_pred, dtype=np.float32),
            np.asarray(prob_true, dtype=np.float32),
        ),
    }


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return _to_python_scalar(value)
