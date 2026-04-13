"""
Submission template (USES trained weights).

Use this template if your agent depends on a trained neural network.
Place your saved model file (weights.pth) inside the submission folder.

The policy loads the model and uses it to predict the best action
from the observation.

The evaluator will import this file and call `policy(obs, rng)`.
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None


def get_obelix_action(continuous_action):
    val = np.clip(continuous_action[0], -1.0, 1.0)
    if val < -0.6:
        return "L45"
    elif val < -0.2:
        return "L22"
    elif val < 0.2:
        return "FW"
    elif val < 0.6:
        return "R22"
    else:
        return "R45"


def _load_once():
    """Load the trained model and weights."""
    global _MODEL
    if _MODEL is not None:
        return

    current_file_path = os.path.abspath(__file__)
    submission_dir = os.path.dirname(current_file_path)

    wpath = os.path.join(submission_dir, "SAC_v3.pth")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal

    class PolicyNetwork(nn.Module):
        def __init__(self, stateDim, actionRange, hiddenDims):
            super(PolicyNetwork, self).__init__()
            self.action_low = torch.tensor(actionRange[0], dtype=torch.float32)
            self.action_high = torch.tensor(actionRange[1], dtype=torch.float32)
            actionDim = len(self.action_low)

            self.action_scale = (self.action_high - self.action_low) / 2.0
            self.action_bias = (self.action_high + self.action_low) / 2.0

            self.layers = nn.ModuleList()
            current_dim = stateDim
            for h in hiddenDims:
                self.layers.append(nn.Linear(current_dim, h))
                current_dim = h

            self.mean_layer = nn.Linear(current_dim, actionDim)
            self.log_std_layer = nn.Linear(current_dim, actionDim)
            self.activation = F.relu
            self.LOG_SIG_MAX = 2
            self.LOG_SIG_MIN = -20

        def forward(self, state):
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            s = state
            for layer in self.layers:
                s = self.activation(layer(s))

            mu = self.mean_layer(s)
            log_std = torch.clamp(self.log_std_layer(s), min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
            std = torch.exp(log_std)

            dist = Normal(mu, std)
            u = dist.rsample()
            a_tanh = torch.tanh(u)
            action_env = self.rescale(a_tanh)

            log_prob = dist.log_prob(u) - torch.log(self.action_scale * (1 - a_tanh.pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            action_greedy = self.rescale(torch.tanh(mu))

            return action_env, action_greedy, log_prob

        def rescale(self, action):
            scale = self.action_scale.to(action.device)
            bias = self.action_bias.to(action.device)
            return action * scale + bias

    net = PolicyNetwork(stateDim=18, actionRange=([-1.0], [1.0]), hiddenDims=[64, 64])
    net.load_state_dict(torch.load(wpath, map_location="cpu"))
    net.eval()

    _MODEL = net


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Use the trained model to choose the best action."""
    _load_once()

    import torch

    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        _, action_greedy, _ = _MODEL(obs_tensor)

    cont_action = action_greedy.squeeze(0).numpy()

    return get_obelix_action(cont_action)
