# Pathological Preprocessing

Pathological preprocessing lives in `code/preprocess/pathological.py` and is invoked automatically by the integrated MIRAGE pipeline.

## Inputs

* `pathological_data.json`

## Processing Steps

* normalizes and deduplicates patient records
* builds numeric and categorical feature sets
* applies frequency encoding for categorical columns
* imputes and scales the tabular matrix
* builds a graph adjacency matrix for downstream training support
* trains a 512-dimensional tabular encoder when PyTorch is available

## Outputs

* `pathological_preprocessed.h5`
* `pathological_embedding_512.h5`
* `pathological_preproc_objects.joblib`
* `pathological_kan_encoder.pt`

## Inference Support

`code/inference/preprocess/pathological_inference.py` reuses the saved schema, encoders, and scaler from the exported resource directory.
