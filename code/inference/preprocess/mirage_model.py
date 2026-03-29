from pathlib import Path
import sys

import torch

CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from mirage import MIRAGENet, QualityGate


def load_mirage_model(checkpoint_path, device):
    model = MIRAGENet(
        d_model=512,
        n_modalities=6,
        n_latents=64,
        n_perceiver_layers=4,
        dropout=0.0,
        use_graph_vae=True,
        use_kan_heads=True,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


__all__ = ["MIRAGENet", "QualityGate", "load_mirage_model"]
