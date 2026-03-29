from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mirage import ensure_directory, extract_patient_id_from_name, save_embedding_h5  # noqa: E402

# ============================================================
# UPDATE THESE PATHS BEFORE RUNNING
# ============================================================
DATASET_ROOT = "/kaggle/input/datasets/sudharsananh/hancothon-2025-complete"
HISTORIES_DIR = f"{DATASET_ROOT}/TextData/TextData/histories_english"
REPORTS_DIR = f"{DATASET_ROOT}/TextData/TextData/reports_english"
SURGERY_DIR = f"{DATASET_ROOT}/TextData/TextData/surgery_descriptions_english"
OUT_DIR = "/kaggle/working/outputs"


def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_text_folder(path: Path) -> Dict[str, List[str]]:
    patient_text = defaultdict(list)
    for file_path in sorted(path.glob("*.txt")):
        patient_id = extract_patient_id_from_name(file_path.name)
        patient_text[patient_id].append(clean_text(file_path.read_text(encoding="utf-8", errors="ignore")))
    return patient_text


def combine_modalities(histories: Dict[str, List[str]], reports: Dict[str, List[str]], surgery: Dict[str, List[str]]) -> Tuple[List[str], List[str], np.ndarray]:
    patient_ids = sorted(set(histories) | set(reports) | set(surgery))
    documents = []
    quality_scores = []
    for patient_id in patient_ids:
        parts = []
        present_modalities = 0
        for title, bucket in (("history", histories), ("report", reports), ("surgery", surgery)):
            texts = [text for text in bucket.get(patient_id, []) if text]
            if texts:
                present_modalities += 1
                parts.append(f"{title}: {' '.join(texts)}")
        document = "\n".join(parts).strip()
        quality = present_modalities / 3.0
        if document:
            quality += min(len(document.split()) / 256.0, 1.0)
        documents.append(document)
        quality_scores.append(min(quality / 2.0, 1.0))
    return patient_ids, documents, np.asarray(quality_scores, dtype=np.float32)


def transformer_embeddings(documents: List[str], model_name_or_path: str, batch_size: int, device: str) -> Tuple[np.ndarray, Dict[str, object]]:
    if not HAS_TRANSFORMERS:
        raise RuntimeError("transformers is not available.")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()
    outputs = []
    for start in range(0, len(documents), batch_size):
        batch_docs = [doc if doc else "[EMPTY]" for doc in documents[start : start + batch_size]]
        encoded = tokenizer(batch_docs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            outputs.append(pooled.cpu().numpy())
    stacked = np.concatenate(outputs, axis=0).astype(np.float32)
    projector = TruncatedSVD(n_components=min(512, stacked.shape[1] - 1), random_state=42)
    reduced = projector.fit_transform(stacked)
    if reduced.shape[1] < 512:
        reduced = np.pad(reduced, ((0, 0), (0, 512 - reduced.shape[1])))
    return reduced.astype(np.float32), {
        "mode": "transformer",
        "projector": projector,
        "transformer_model_path": model_name_or_path,
    }


def tfidf_embeddings(documents: List[str]) -> Tuple[np.ndarray, Dict[str, object]]:
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform([doc if doc else "[EMPTY]" for doc in documents])
    n_components = max(2, min(512, matrix.shape[1] - 1, matrix.shape[0] - 1))
    if n_components >= 2:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = svd.fit_transform(matrix)
    else:
        svd = None
        reduced = matrix.toarray()
    if reduced.shape[1] < 512:
        reduced = np.pad(reduced, ((0, 0), (0, 512 - reduced.shape[1])))
    return reduced[:, :512].astype(np.float32), {
        "mode": "tfidf",
        "vectorizer": vectorizer,
        "tfidf_vectorizer": vectorizer,
        "svd": svd,
        "projector": svd,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MIRAGE-Net semantic embeddings.")
    parser.add_argument("--histories", default=HISTORIES_DIR)
    parser.add_argument("--reports", default=REPORTS_DIR)
    parser.add_argument("--surgery", default=SURGERY_DIR)
    parser.add_argument("--outdir", default=OUT_DIR)
    parser.add_argument("--use_transformer", action="store_true")
    parser.add_argument("--transformer_model", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outdir = ensure_directory(args.outdir)
    histories = read_text_folder(Path(args.histories))
    reports = read_text_folder(Path(args.reports))
    surgery = read_text_folder(Path(args.surgery))
    patient_ids, documents, quality_scores = combine_modalities(histories, reports, surgery)

    if args.use_transformer and HAS_TRANSFORMERS:
        device = args.device if torch.cuda.is_available() else "cpu"
        embeddings, preproc_objects = transformer_embeddings(documents, args.transformer_model, args.batch_size, device)
    else:
        embeddings, preproc_objects = tfidf_embeddings(documents)
        preproc_objects["transformer_model_path"] = args.transformer_model

    save_embedding_h5(
        outdir / "text_semantic_embedding_512.h5",
        patient_ids,
        embeddings,
        quality_scores=quality_scores,
        extra={"document": documents},
    )
    joblib.dump(preproc_objects, outdir / "text_semantic_preproc_objects.joblib")
    print(f"semantic patients: {len(patient_ids)}")
    print(f"embedding shape: {embeddings.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
