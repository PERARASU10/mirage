# Spatial Preprocessing

Spatial preprocessing is implemented in `code/preprocess/spatial.py`.

## Inputs

* primary tumor WSI feature directories
* lymph node WSI feature directory
* per-slide `.h5` files containing patch features and coordinates

## Processing Steps

* discovers WSI feature files across the configured directories
* groups slides by patient and by region type
* projects patch features into the shared 512-dimensional space
* injects positional information with `PositionalMLP`
* aggregates patches with hierarchical attention
* fuses primary and lymph representations into one patient embedding
* estimates quality from patch availability and region coverage

## Outputs

* `spatial_embedding_512.h5`
* `patch_projector.pt`
* `positional_mlp.pt`
* `habmil_model.pt`
* `spatial_fusion.pt`
* `spatial_preproc_objects.joblib`

## Inference Support

These saved modules are loaded by `code/inference/preprocess/spatial_inference.py` to reproduce the same spatial embedding path at prediction time.
