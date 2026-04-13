import os
import pickle
import numpy as np
from typing import Sequence

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

_POLICY = None


def _load_policy():
    global _POLICY
    if _POLICY is not None:
        return

    submission_dir = os.path.dirname(__file__)
    pkl_path = os.path.join(submission_dir, "weightsTab/qlearning_v1.pkl")

    with open(pkl_path, "rb") as f:
        _POLICY = pickle.load(f)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    _load_policy()

    state = tuple(obs.astype(int))

    action_data = _POLICY[state]

    if isinstance(action_data, np.ndarray):
        return ACTIONS[int(np.argmax(action_data))]

    else:
        return action_data
