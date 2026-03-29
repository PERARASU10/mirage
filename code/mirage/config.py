from __future__ import annotations

import os
from pathlib import Path


DATASET_ROOT = Path(
    os.environ.get(
        "MIRAGE_DATASET_ROOT",
        "/kaggle/input/datasets/sudharsananh/hancothon-2025-complete",
    )
)

WORKING_ROOT = Path(os.environ.get("MIRAGE_WORKING_ROOT", "/kaggle/working"))

CLINICAL_JSON = DATASET_ROOT / "StructuredData/StructuredData/clinical_data.json"
PATHOLOGICAL_JSON = DATASET_ROOT / "StructuredData/StructuredData/pathological_data.json"
BLOOD_JSON = DATASET_ROOT / "StructuredData/StructuredData/blood_data.json"
BLOOD_REF_JSON = DATASET_ROOT / "StructuredData/StructuredData/blood_data_reference_ranges.json"

HISTORIES_DIR = DATASET_ROOT / "TextData/TextData/histories_english"
REPORTS_DIR = DATASET_ROOT / "TextData/TextData/reports_english"
SURGERY_DIR = DATASET_ROOT / "TextData/TextData/surgery_descriptions_english"
ICD_CODES_DIR = DATASET_ROOT / "TextData/TextData/icd_codes"
OPS_CODES_DIR = DATASET_ROOT / "TextData/TextData/ops_codes"

WSI_LYMPH = DATASET_ROOT / "WSI_UNI_encodings/WSI_LymphNode/h5_files"
WSI_CUP = DATASET_ROOT / "WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_CUP/h5_files"
WSI_HYPO = DATASET_ROOT / "WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_Hypopharynx/h5_files"
WSI_LARYNX = DATASET_ROOT / "WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_Larynx/h5_files"
WSI_ORAL = DATASET_ROOT / "WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_OralCavity/h5_files"
WSI_OROPH1 = DATASET_ROOT / "WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_Oropharynx_Part1/h5_files"
WSI_OROPH2 = DATASET_ROOT / "WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_Oropharynx_Part2/h5_files"

OUT_DIR = Path(os.environ.get("MIRAGE_OUT_DIR", str(WORKING_ROOT / "outputs")))
CHECKPOINT_DIR = Path(os.environ.get("MIRAGE_CHECKPOINT_DIR", str(WORKING_ROOT / "checkpoints")))
INFERENCE_RESOURCES_DIR = Path(
    os.environ.get(
        "MIRAGE_INFERENCE_RESOURCES_DIR",
        str(WORKING_ROOT / "inference_resources"),
    )
)
