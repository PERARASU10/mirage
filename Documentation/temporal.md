# Temporal Preprocessing

Temporal preprocessing is implemented in `code/preprocess/Temporal.py`. The repository keeps the capitalized filename for compatibility, but the stage is now called automatically by the integrated training pipeline.

## Inputs

* `blood_data.json`
* `blood_data_reference_ranges.json`

## Processing Steps

* normalizes blood records into a patient/analyte/time dataframe
* selects analytes with enough cohort support
* bins measurements into a fixed sequence window
* fills missing values using physiology-aware defaults and cohort-level imputation
* scales the resulting sequence tensors
* trains either:
  * an RWKV temporal encoder
  * an LSTM temporal encoder
  * or a deterministic fallback embedding path when PyTorch is unavailable

## Outputs

* `temporal_embedding_512.h5`
* `temporal_preproc_objects.joblib`
* `temporal_preproc_summary.json`
* `temporal_rwkv_encoder.pt` or `temporal_lstm_encoder.pt`

## Inference Support

The saved preprocessing object includes analytes, time bins, reference ranges, the imputer, and the scaler used by `code/inference/preprocess/temporal_inference.py`.
