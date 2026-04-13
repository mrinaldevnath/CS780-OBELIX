import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

_NET = None


def _create_value_network(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, out_dim),
    )


class DuelingNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


ALGO = "DQN"  # Options: "NFQ", "DQN", "DDQN", "D3QN", "D3QN_PER"
WEIGHT_FILE = "weightsPth/DQN_v1.pth"


def _load_network():
    global _NET
    if _NET is not None:
        return

    in_dim = 18
    out_dim = len(ACTIONS)

    if ALGO in ["NFQ", "DQN", "DDQN"]:
        _NET = _create_value_network(in_dim, out_dim)
    elif ALGO in ["D3QN", "D3QN_PER"]:
        _NET = DuelingNetwork(in_dim, out_dim)
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
        q_values = _NET(state_t)

    return ACTIONS[torch.argmax(q_values, dim=1).item()]
