import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from tqdm import tqdm

from obelix import OBELIX

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

CONFIG = {
    "scaling_factor": 5,
    "arena_size": 500,
    "max_steps": 2000,
    "wall_obstacles": True,
    "difficulty": 3,
    "box_speed": 2,
    "seed": 200,
    "gamma": 0.99,
    "tau": 0.01,
    "buffer_size": 100000,
    "batch_size": 128,
    "update_frequency": 1,
    "total_timesteps": 2000000,
    "min_samples": 1000,
    "max_eval_episodes": 5,
    "eval_freq": 10000,
    "max_grad_norm": 1.0,
    "learning_rate": 3e-4,
    "hDims": [64, 64],
    "policy_file": "weightsPth/SAC.pth",
}


class ActionSpace:
    def __init__(self):
        self.low = np.array([-1.0], dtype=np.float32)
        self.high = np.array([1.0], dtype=np.float32)
        self.shape = (1,)
        self._rng = np.random.default_rng()

    def sample(self):
        return self._rng.uniform(self.low, self.high).astype(np.float32)


class ObservationSpace:
    def __init__(self, dim):
        self.shape = (dim,)


def continuous_to_discrete(continuous_action):
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


class OBELIXContinuousWrapper:
    def __init__(self, **kwargs):
        self._env = OBELIX(**kwargs)
        self.observation_space = ObservationSpace(18)
        self.action_space = ActionSpace()
        self.max_steps = self._env.max_steps

    def reset(self, seed=None):
        obs = self._env.reset(seed=seed)
        return obs.copy(), {}

    def step(self, continuous_action):
        discrete_action = continuous_to_discrete(continuous_action)
        obs, reward, done = self._env.step(discrete_action, render=False)

        terminated = done and (self._env.current_step < self._env.max_steps)
        truncated = done and (self._env.current_step >= self._env.max_steps)

        return obs.copy(), float(reward), terminated, truncated, {}


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        random.seed(seed)
        np.random.seed(seed)
        self.pos = 0
        self.size = 0
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

    def store(self, state, action, reward, next_state, done):
        if self.states is None:
            state_shape = np.shape(state)
            action_shape = np.shape(action)
            self.states = np.zeros((self.buffer_size, *state_shape), dtype=np.float32)
            self.actions = np.zeros((self.buffer_size, *action_shape), dtype=np.float32)
            self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
            self.next_states = np.zeros((self.buffer_size, *state_shape), dtype=np.float32)
            self.dones = np.zeros((self.buffer_size,), dtype=bool)
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def length(self):
        return self.size


class ValueNetwork(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenDims):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = stateDim + actionDim
        for h in hiddenDims:
            self.layers.append(nn.Linear(current_dim, h))
            current_dim = h
        self.out_layer = nn.Linear(current_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out_layer(x)


class PolicyNetwork(nn.Module):
    def __init__(self, stateDim, actionRange, hiddenDims):
        super().__init__()
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
        self.LOG_SIG_MAX = 2
        self.LOG_SIG_MIN = -20

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        s = state
        for layer in self.layers:
            s = F.relu(layer(s))

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


class SAC:
    def __init__(self, config):
        self.config = config
        self.seed = config.get("seed", 200)

        self.env = OBELIXContinuousWrapper(
            scaling_factor=config.get("scaling_factor", 5),
            arena_size=config.get("arena_size", 500),
            max_steps=config.get("max_steps", 2000),
            wall_obstacles=config.get("wall_obstacles", True),
            difficulty=config.get("difficulty", 3),
            box_speed=config.get("box_speed", 2),
            seed=config.get("seed", 200),
        )

        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.bufferSize = config.get("buffer_size", 1000000)
        self.batch_size = config.get("batch_size", 256)
        self.updateFrequency = config.get("update_frequency", 1)
        self.total_timesteps = config.get("total_timesteps", 1_000_000)
        self.minSamples = config.get("min_samples", 1000)
        self.MAX_EVAL_EPISODE = config.get("max_eval_episodes", 5)
        self.eval_freq = config.get("eval_freq", 10000)
        self.policy_file = config.get("policy_file", "SAC_continuous_v1.pth")
        self.maxGradNorm = config.get("max_grad_norm", 1.0)

        lr = config.get("learning_rate", 3e-4)
        hDims = config.get("hDims", [256, 256])

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.stateDim = self.env.observation_space.shape[0]
        self.actionDim = self.env.action_space.shape[0]
        self.envActionRange = (self.env.action_space.low, self.env.action_space.high)

        self.policyNetwork = PolicyNetwork(self.stateDim, self.envActionRange, hDims)
        self.q1_online = ValueNetwork(self.stateDim, self.actionDim, hDims)
        self.q2_online = ValueNetwork(self.stateDim, self.actionDim, hDims)
        self.q1_target = ValueNetwork(self.stateDim, self.actionDim, hDims)
        self.q2_target = ValueNetwork(self.stateDim, self.actionDim, hDims)

        self.updateTargetNetworks(tau=1.0)

        self.policyOptimizer = Adam(self.policyNetwork.parameters(), lr=lr)
        self.q1_Optimizer = Adam(self.q1_online.parameters(), lr=lr)
        self.q2_Optimizer = Adam(self.q2_online.parameters(), lr=lr)

        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alphaOptimizer = Adam([self.log_alpha], lr=lr)

        self.rBuffer = ReplayBuffer(self.bufferSize, self.batch_size, self.seed)
        self.best_eval_reward = -float("inf")

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def updateTargetNetworks(self, tau=None):
        if tau is None:
            tau = self.tau
        for tParam, oParam in zip(self.q1_target.parameters(), self.q1_online.parameters()):
            tParam.data.copy_((1 - tau) * tParam.data + tau * oParam.data)
        for tParam, oParam in zip(self.q2_target.parameters(), self.q2_online.parameters()):
            tParam.data.copy_((1 - tau) * tParam.data + tau * oParam.data)

    def selectAction(self, state, evaluate=False):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_env, action_greedy, _ = self.policyNetwork(state_t)
        return action_greedy.squeeze(0).numpy() if evaluate else action_env.squeeze(0).numpy()

    def _save_best(self, eval_reward):
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            if self.policy_file is not None:
                save_dir = os.path.dirname(self.policy_file)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                torch.save(self.policyNetwork.state_dict(), self.policy_file)

    def _save_last(self):
        if self.policy_file is not None:
            last_path = self.policy_file.replace(".pth", "_last.pth")
            save_dir = os.path.dirname(last_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(self.policyNetwork.state_dict(), last_path)

    def train(self):
        state, _ = self.env.reset(seed=self.seed)

        pbar = tqdm(range(self.total_timesteps), desc="Training SAC (Continuous)", leave=False)

        for step in pbar:
            if self.rBuffer.length() < self.minSamples:
                action = self.env.action_space.sample()
            else:
                action = self.selectAction(state, evaluate=False)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.rBuffer.store(state, action, reward, next_state, terminated)

            if self.rBuffer.length() > self.minSamples:
                experiences = self.rBuffer.sample(self.batch_size)
                self.trainNetwork(experiences, step)

            state = next_state

            if done:
                state, _ = self.env.reset()

            if (step + 1) % self.eval_freq == 0:
                eval_mean = np.mean(self.evaluateAgent())
                self._save_best(eval_mean)
                pbar.set_postfix(
                    {
                        "Best Eval": f"{self.best_eval_reward:.1f}",
                        "Curr Eval": f"{eval_mean:.1f}",
                    }
                )

        self._save_last()
        return self.best_eval_reward

    def trainNetwork(self, experiences, step):
        states, actions, rewards, next_states, dones = experiences

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            next_actions, _, next_log_probs = self.policyNetwork(next_states)
            q1_next_target = self.q1_target(next_states, next_actions)
            q2_next_target = self.q2_target(next_states, next_actions)
            min_q_next_target = (
                torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_probs
            )
            target_q = rewards + self.gamma * (1.0 - dones) * min_q_next_target

        q1_pred = self.q1_online(states, actions)
        q2_pred = self.q2_online(states, actions)

        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_Optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1_online.parameters(), self.maxGradNorm)
        self.q1_Optimizer.step()

        self.q2_Optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2_online.parameters(), self.maxGradNorm)
        self.q2_Optimizer.step()

        if step % self.updateFrequency == 0:
            curr_actions, _, curr_log_probs = self.policyNetwork(states)
            q1_curr = self.q1_online(states, curr_actions)
            q2_curr = self.q2_online(states, curr_actions)
            min_q_curr = torch.min(q1_curr, q2_curr)

            policy_loss = (self.alpha.detach() * curr_log_probs - min_q_curr).mean()

            self.policyOptimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.maxGradNorm)
            self.policyOptimizer.step()

            alpha_loss = -(self.log_alpha * (curr_log_probs + self.target_entropy).detach()).mean()

            self.alphaOptimizer.zero_grad()
            alpha_loss.backward()
            self.alphaOptimizer.step()

            self.updateTargetNetworks()

    def evaluateAgent(self):
        eval_rewards = []
        for i in range(self.MAX_EVAL_EPISODE):
            state, _ = self.env.reset(seed=self.seed + 1000 + i)
            done = False
            ep_reward = 0
            while not done:
                action = self.selectAction(state, evaluate=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
            eval_rewards.append(ep_reward)
        return eval_rewards


if __name__ == "__main__":
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG["seed"])
        torch.cuda.manual_seed_all(CONFIG["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    agent = SAC(CONFIG)

    print("Starting SAC (Continuous) Training...")
    best_reward = agent.train()

    final_eval = np.mean(agent.evaluateAgent())
    agent._save_best(final_eval)
    agent._save_last()

    print(f"\nTraining complete!")
    print(f"Best Eval Reward : {agent.best_eval_reward:.2f} -> {CONFIG['policy_file']}")
    print(f"Last Weights     : {CONFIG['policy_file'].replace('.pth', '_last.pth')}")
