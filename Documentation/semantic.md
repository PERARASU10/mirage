# Semantic Preprocessing

Semantic preprocessing is implemented in `code/preprocess/semantic.py`.

## Inputs

* histories text directory
* reports text directory
* surgery description text directory

## Processing Steps

* reads text files and maps them to patient IDs
* cleans and merges document text across the three sources
* computes modality-aware quality scores
* builds embeddings using either:
  * TF-IDF plus TruncatedSVD
  * Bio_ClinicalBERT when `--semantic-use-transformer` is enabled and dependencies are available

## Outputs

* `text_semantic_embedding_512.h5`
* `text_semantic_preproc_objects.joblib`

## Inference Support

The saved preprocessing object stores the vectorizer or transformer projection information used by `code/inference/preprocess/semantic_inference.py`.
