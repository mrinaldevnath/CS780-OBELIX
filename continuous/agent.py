import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

_NET = None


class ReLUPolicyNetwork(nn.Module):
    """Policy network used by TD3 and SAC."""

    def __init__(self, in_dim, out_dim, h_dim=[64, 64]):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = in_dim
        for h in h_dim:
            self.layers.append(nn.Linear(current_dim, h))
            current_dim = h
        self.out_layer = nn.Linear(current_dim, out_dim)

    def forward(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        for layer in self.layers:
            s = F.relu(layer(s))
        return self.out_layer(s)


ALGO = "TD3"  # Options: "TD3", "SAC", "PPO"
WEIGHT_FILE = "weightsPth/TD3_v1.pth"


def _load_network():
    global _NET
    if _NET is not None:
        return

    in_dim = 18
    out_dim = len(ACTIONS)

    if ALGO in ["TD3", "SAC"]:
        _NET = ReLUPolicyNetwork(in_dim, out_dim)
    elif ALGO == "PPO":
        _NET = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim),
        )
    else:
        raise ValueError(f"Unknown algorithm: {ALGO}")

    submission_dir = os.path.dirname(__file__)
    pth_path = os.path.join(submission_dir, WEIGHT_FILE)

    _NET.load_state_dict(torch.load(pth_path, map_location=torch.device("cpu")))
    _NET.eval()


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_network()

    state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = _NET(state_t)

    return ACTIONS[torch.argmax(logits, dim=1).item()]
