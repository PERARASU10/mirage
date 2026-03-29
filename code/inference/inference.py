from pathlib import Path
import json
import os
import sys

import torch

BASE_DIR = Path(__file__).resolve().parent
ROOT = BASE_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from preprocess.clinical_inference import get_clinical_embedding
from preprocess.codes_inference import get_codes_embedding
from preprocess.mirage_model import load_mirage_model
from preprocess.pathological_inference import get_pathological_embedding
from preprocess.semantic_inference import get_semantic_embedding
from preprocess.spatial_inference import get_spatial_embedding
from preprocess.temporal_inference import get_temporal_embedding


PREDICTION_TARGET_SLUG = "5-year-survival"


def default_resource_path() -> Path:
    configured = os.environ.get("MIRAGE_INFERENCE_RESOURCES_DIR")
    if configured:
        return Path(configured)
    kaggle_resources = Path("/kaggle/working/inference_resources")
    if kaggle_resources.parent.exists():
        return kaggle_resources
    return BASE_DIR / "resources"

if BASE_DIR == Path("/opt/app"):
    INPUT_PATH = Path("/input")
    OUTPUT_PATH = Path("/output")
    RESOURCE_PATH = default_resource_path()
else:
    INPUT_PATH = BASE_DIR / "test" / "input" / "interf0"
    OUTPUT_PATH = BASE_DIR / "test" / "output" / "interf0"
    RESOURCE_PATH = default_resource_path()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def load_json_file(location: Path):
    try:
        with open(location, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json_file(location: Path, content):
    with open(location, "w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2)


def output_name(task_slug: str) -> str:
    return "2-year-recurrence.json" if task_slug == "2-year-recurrence-after-diagnosis" else "5-year-survival.json"


def prediction_to_string(probability: float, task_slug: str) -> str:
    if task_slug == "2-year-recurrence-after-diagnosis":
        return "recurrence" if probability >= 0.5 else "no recurrence"
    return "deceased" if probability >= 0.5 else "living"


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {
        "clinical": load_json_file(INPUT_PATH / "hancock-clinical-data.json"),
        "pathological": load_json_file(INPUT_PATH / "hancock-pathological-data.json"),
        "blood": load_json_file(INPUT_PATH / "hancock-blood-data.json"),
        "surgery_text": load_json_file(INPUT_PATH / "hancock-surgery-text-data.json"),
        "primary_wsi": load_json_file(INPUT_PATH / "hancock-primary-tumor-wsi-embeddings.json"),
        "lymph_wsi": load_json_file(INPUT_PATH / "hancock-lymph-node-wsi-embeddings.json"),
    }

    modality_outputs = [
        get_clinical_embedding(inputs["clinical"], RESOURCE_PATH, device),
        get_pathological_embedding(inputs["pathological"], RESOURCE_PATH, device),
        get_temporal_embedding(inputs["blood"], RESOURCE_PATH, device),
        get_semantic_embedding(inputs["surgery_text"], RESOURCE_PATH, device),
        get_codes_embedding(inputs["surgery_text"], RESOURCE_PATH, device),
        get_spatial_embedding(inputs["primary_wsi"], inputs["lymph_wsi"], RESOURCE_PATH, device),
    ]

    emb_stack = torch.stack([output["embedding"] for output in modality_outputs], dim=0).unsqueeze(0).to(device)
    quality_tensor = torch.tensor([[output["quality"] for output in modality_outputs]], dtype=torch.float32, device=device)
    present_tensor = torch.tensor([[output["present"] for output in modality_outputs]], dtype=torch.int64, device=device)

    model = load_mirage_model(RESOURCE_PATH / "mirage_net_best_avg.pt", device)
    with torch.no_grad():
        outputs = model(emb_stack, quality_tensor, present_tensor, patient_graph=None)

    logit = outputs["logit_rec"] if PREDICTION_TARGET_SLUG == "2-year-recurrence-after-diagnosis" else outputs["logit_surv"]
    probability = torch.sigmoid(logit).item()
    write_json_file(OUTPUT_PATH / output_name(PREDICTION_TARGET_SLUG), prediction_to_string(probability, PREDICTION_TARGET_SLUG))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
