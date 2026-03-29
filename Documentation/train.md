# Training Pipeline

The shareable entrypoint is the repository root `train.py`. It delegates to `code/train.py`, runs the full MIRAGE-Net preprocessing pipeline, trains the model, and exports inference resources in one command.

## Default Flow

```bash
python train.py
```

By default this pipeline:

* preprocesses all six modalities:
  * clinical
  * pathological
  * temporal blood
  * semantic text
  * ICD/OPS codes
  * spatial WSI
* saves modality embeddings to `/kaggle/working/outputs`
* saves checkpoints and metrics to `/kaggle/working/checkpoints`
* exports inference assets to `/kaggle/working/inference_resources`

## Main Components

* `code/mirage/pipeline.py`
  * wraps each preprocessing stage in a class
  * allows `train.py` to build missing artifacts automatically
* `code/train.py`
  * aligns modality embeddings
  * runs cross-validation
  * trains `MIRAGENet`
  * writes `training_summary.json`
  * exports the final inference bundle
* `code/inference/inference.py`
  * loads the saved preprocessing objects and model checkpoint
  * runs inference from the exported resource directory

## Important CLI Flags

```bash
python train.py --force-preprocess --epochs 60 --device cuda
```

Common options:

* `--force-preprocess`: rebuild preprocessing artifacts even if they already exist
* `--skip-preprocess`: train from existing HDF5 embeddings
* `--preprocess-outdir`: override the embedding output directory
* `--outdir`: override the checkpoint directory
* `--resources-dir`: override the inference resource export directory
* `--device`: set `cuda` or `cpu`

## Outputs

Training produces:

* fold checkpoints such as `fold_0_best_model.pt`
* `mirage_net_best_avg.pt`
* ROC, calibration, and confusion-matrix artifacts
* `training_summary.json`

The exported inference bundle includes preprocessing joblib files, modality encoder weights, spatial modules, and the final model checkpoint.
