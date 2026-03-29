# Repository Guidelines

## Project Structure & Module Organization
`code/` contains the runnable Python pipeline. Use `code/preprocess/` for modality-specific embedding builders (`clinical.py`, `Temporal.py`, `semantic.py`, `pathological.py`, `spatial.py`), `code/train.py` for multimodal training, and `code/inference/` for prediction packaging and local inference. `Documentation/` holds per-module notes and CLI examples. `img/` stores figures referenced by the docs and README.

## Build, Test, and Development Commands
Create a Python 3.8+ environment with PyTorch, NumPy, Pandas, scikit-learn, h5py, and joblib installed.

```bash
python code/preprocess/clinical.py --epochs 50 --bs 128
python code/preprocess/Temporal.py --mode lstm --epochs 50
python code/preprocess/semantic.py --histories <dir> --reports <dir> --surgery <dir> --outdir <outdir>
python code/train.py --clinical <clinical.h5> --semantic <semantic.h5> --temporal <temporal.h5> --pathological <pathological.h5> --spatial <spatial.h5> --outdir checkpoints
python code/inference/inference.py
```

Run preprocessors first, then train on the generated HDF5 embeddings. Inference expects the resource/test directory layout defined in `code/inference/inference.py`.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and snake_case for functions, variables, and CLI flags. Preserve the repository’s script-oriented style: keep entrypoints under `if __name__ == "__main__":`, keep constants near the top of each file, and prefer explicit argparse options over hard-coded paths. Match existing file naming when extending a modality pipeline, even where legacy capitalization exists such as `Temporal.py`.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Treat each change as requiring a smoke test:
1. run the affected preprocessing or training command on a small sample;
2. confirm the expected `.h5`, `.json`, or `.pt` outputs are created;
3. review console metrics or warnings for regressions.
Document any skipped validation in the PR.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects such as `Update README.md` and `Delete img/images.txt`. Keep commits focused and concise: `Add temporal preprocessing summary`, `Fix inference resource path`. PRs should include scope, changed data paths or artifacts, validation performed, and screenshots only when documentation or visual outputs changed. Link the related issue or experiment note when available.

## Data & Configuration Notes
Do not commit patient data, generated checkpoints, or large HDF5 artifacts. Keep local paths configurable through CLI arguments, and store reusable model weights or preprocessed files in external artifact storage rather than the repository.
