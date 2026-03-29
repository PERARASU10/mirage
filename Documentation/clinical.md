# Clinical Preprocessing

Clinical preprocessing is implemented in `code/preprocess/clinical.py` and is invoked automatically from `code/mirage/pipeline.py` when you run `python train.py`.

## Inputs

* `clinical_data.json`

## Processing Steps

* normalizes patient identifiers
* derives survival and recurrence labels
* builds numeric and categorical feature tables
* applies frequency encoding for categorical values
* imputes missing values and scales the feature matrix
* builds a patient similarity graph
* trains a tabular autoencoder when PyTorch is available
* falls back to deterministic preprocessing where needed

## Outputs

Saved under the preprocessing output directory:

* `clinical_preprocessed.h5`
* `clinical_embedding_512.h5`
* `clinical_preproc_objects.joblib`
* `clinical_kan_encoder.pt`

## Inference Support

The exported preprocessing objects include the feature schema, encoders, imputer, and scaler needed by `code/inference/preprocess/clinical_inference.py`.
