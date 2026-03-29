#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "code"
MODULE_PATH = CODE_DIR / "train.py"


def load_train_module():
    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    spec = importlib.util.spec_from_file_location("repo_code_train", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load training entrypoint from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_train_module()
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
