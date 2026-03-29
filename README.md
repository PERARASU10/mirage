# MIRAGE-Net

MIRAGE-Net is the upgraded HANCOCK multimodal pipeline for 5-year survival and 2-year recurrence prediction. The repository now uses 6 modalities in a single architecture:

- clinical tabular
- pathological tabular
- blood temporal
- semantic text
- ICD/OPS codes
- WSI spatial

The repository root `train.py` is the single entrypoint. It runs preprocessing, training, and inference resource export with default Kaggle-friendly paths. The underlying implementation lives in `code/train.py`, preprocessing stages live under `code/preprocess/`, and inference lives under `code/inference/`.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- h5py
- joblib
- Optional: `transformers` for semantic preprocessing with Bio_ClinicalBERT

## Training

Update dataset paths or set environment variables in [code/mirage/config.py](/C:/Users/PERARASU%20M/Desktop/Final%20Year%20Project/Project/Hcat-FusionNet/code/mirage/config.py), then run:

```bash
python train.py
```

This single command:

- runs all preprocessing stages as classes from `code/mirage/pipeline.py`
- writes preprocessed embeddings to `/kaggle/working/outputs`
- writes fold checkpoints and `training_summary.json` to `/kaggle/working/checkpoints`
- exports inference resources to `/kaggle/working/inference_resources`

You can still override defaults with CLI flags such as:

```bash
python train.py --force-preprocess --epochs 60 --device cuda
```

This Windows workspace keeps the legacy filename `Temporal.py`; it is imported by the integrated preprocessing pipeline.

## Inference

The inference stack is under [code/inference/inference.py](/C:/Users/PERARASU%20M/Desktop/Final%20Year%20Project/Project/Hcat-FusionNet/code/inference/inference.py). By default it reads resources from `/kaggle/working/inference_resources`, or from `MIRAGE_INFERENCE_RESOURCES_DIR` when that environment variable is set. Training exports the required files automatically, including:

- `clinical_preproc_objects.joblib`
- `clinical_kan_encoder.pt`
- `pathological_preproc_objects.joblib`
- `pathological_kan_encoder.pt`
- `temporal_preproc_objects.joblib`
- `temporal_rwkv_encoder.pt` or `temporal_lstm_encoder.pt`
- `text_semantic_preproc_objects.joblib`
- `codes_preproc_objects.joblib`
- `codes_hierarchical_embedder.pt`
- `spatial_preproc_objects.joblib`
- `patch_projector.pt`
- `positional_mlp.pt`
- `spatial_fusion.pt`
- `habmil_model.pt`
- `mirage_net_best_avg.pt`

Set `PREDICTION_TARGET_SLUG` in `code/inference/inference.py` to either `5-year-survival` or `2-year-recurrence-after-diagnosis`.
