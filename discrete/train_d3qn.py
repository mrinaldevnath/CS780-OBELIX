import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from obelix import OBELIX

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

CONFIG = {
    "scaling_factor": 5,
    "arena_size": 500,
    "max_steps": 1000,
    "wall_obstacles": True,
    "difficulty": 0,
    "box_speed": 2,
    "seed": 200,
    "gamma": 0.99,
    "epsilon_init": 1.0,
    "min_epsilon": 0.05,
    "epsilon_decay": 0.99,
    "max_train_eps": 1000,
    "max_eval_eps": 1,
    "tau": 0.05,
    "updateFrequency": 1,
    "bufferSize": 100000,
    "batchSize": 256,
    "optimizerLR": 1e-3,
    "policy_file": "weightsPth/D3QN_v1.pth",
}


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def decay_epsilon(e, initial_epsilon, min_epsilon, decay_rate):
    return max(min_epsilon, initial_epsilon * (decay_rate**e))


class DuelingNetwork(nn.Module):
    def __init__(self, inDim, outDim):
        super(DuelingNetwork, self).__init__()
        self.feature_layer = nn.Sequential(nn.Linear(inDim, 64), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, outDim))

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


def epsilon_greedy_train(net, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(ACTIONS))
    else:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = net(state_t)
        return torch.argmax(q_values, dim=1).item()


def greedy_eval(net, state):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = net(state_t)
    return torch.argmax(q_values, dim=1).item()


class ReplayBuffer:
    def __init__(self, bufferSize):
        self.bufferSize = bufferSize
        self.buffer = [None] * bufferSize
        self.pos = 0
        self.size = 0

    def store(self, experience):
        self.buffer[self.pos] = experience
        self.pos = (self.pos + 1) % self.bufferSize
        self.size = min(self.size + 1, self.bufferSize)

    def sample(self, batchSize):
        indices = np.random.choice(self.size, batchSize, replace=False)
        return [self.buffer[idx] for idx in indices]

    def splitExperiences(self, experiences):
        states, actions, rewards, nextStates, dones = map(np.array, zip(*experiences))
        return states, actions, rewards, nextStates, dones

    def length(self):
        return self.size


class D3QN:
    def __init__(
        self,
        env,
        seed,
        gamma,
        tau,
        bufferSize,
        batchSize,
        optimizerLR,
        updateFrequency,
        max_train_eps,
        max_eval_eps,
    ):
        self.env = env
        self.seed = seed
        self.gamma = gamma
        self.tau = tau
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.updateFrequency = updateFrequency
        self.MAX_TRAIN_EPISODES = max_train_eps
        self.MAX_EVAL_EPISODES = max_eval_eps

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        inDim = 18
        outDim = len(ACTIONS)

        self.onlineNet = DuelingNetwork(inDim, outDim)
        self.targetNet = DuelingNetwork(inDim, outDim)
        self.targetNet.load_state_dict(self.onlineNet.state_dict())

        self.optimizer = Adam(self.onlineNet.parameters(), lr=optimizerLR)
        self.rBuffer = ReplayBuffer(self.bufferSize)

        self.initBookKeeping()

    def initBookKeeping(self):
        self.trainRewardsList = []
        self.evalRewardsList = []
        self.current_ep_train_reward = 0
        self.current_ep_eval_reward = 0

    def performBookKeeping(self, train=True):
        if train:
            self.trainRewardsList.append(self.current_ep_train_reward)
            self.evalRewardsList.append(self.current_ep_eval_reward)

    def trainAgent(self):
        pbar = tqdm(range(self.MAX_TRAIN_EPISODES), desc="Training D3QN", unit="ep")
        for episode in pbar:
            epsilon = decay_epsilon(
                episode, CONFIG["epsilon_init"], CONFIG["min_epsilon"], CONFIG["epsilon_decay"]
            )

            state = self.env.reset(seed=self.seed + episode)
            done = False
            ep_reward = 0

            while not done:
                action_idx = epsilon_greedy_train(self.onlineNet, state, epsilon)
                action_str = ACTIONS[action_idx]

                next_state, reward, done = self.env.step(action_str, render=False)

                self.rBuffer.store((state, action_idx, reward, next_state, done))
                state = next_state
                ep_reward += reward

            if self.rBuffer.length() >= self.batchSize:
                experiences = self.rBuffer.sample(self.batchSize)
                self.trainNetwork(experiences)

            if episode % self.updateFrequency == 0:
                self.updateNetwork()

            eval_rewards = self.evaluateAgent()
            self.current_ep_train_reward = ep_reward
            self.current_ep_eval_reward = np.mean(eval_rewards)
            self.performBookKeeping(train=True)

            pbar.set_postfix({"Eval Rwd": f"{self.current_ep_eval_reward:.1f}"})

        return self.trainRewardsList, self.evalRewardsList

    def trainNetwork(self, experiences):
        states, actions, rewards, next_states, dones = self.rBuffer.splitExperiences(experiences)

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            argmax_a_qs = self.onlineNet(next_states_t).max(1)[1].unsqueeze(1)
            max_next_q = self.targetNet(next_states_t).gather(1, argmax_a_qs)
            targets = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        q_values = self.onlineNet(states_t).gather(1, actions_t)
        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def updateNetwork(self):
        # Soft Polyak Averaging
        for target_param, online_param in zip(
            self.targetNet.parameters(), self.onlineNet.parameters()
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def evaluateAgent(self):
        finalEvalRewardsList = []
        for eval_ep in range(self.MAX_EVAL_EPISODES):
            state = self.env.reset(seed=self.seed + 1000 + eval_ep)
            done = False
            ep_reward = 0

            self.onlineNet.eval()
            while not done:
                action_idx = greedy_eval(self.onlineNet, state)
                action_str = ACTIONS[action_idx]
                next_state, reward, done = self.env.step(action_str, render=False)
                state = next_state
                ep_reward += reward
            self.onlineNet.train()

            finalEvalRewardsList.append(ep_reward)
        return finalEvalRewardsList

    def runD3QN(self):
        trainRewardsList, evalRewardsList = self.trainAgent()
        final_eval_rewards = self.evaluateAgent()
        return np.mean(final_eval_rewards)


if __name__ == "__main__":
    set_global_seeds(CONFIG["seed"])

    env = OBELIX(
        scaling_factor=CONFIG["scaling_factor"],
        arena_size=CONFIG["arena_size"],
        max_steps=CONFIG["max_steps"],
        wall_obstacles=CONFIG["wall_obstacles"],
        difficulty=CONFIG["difficulty"],
        box_speed=CONFIG["box_speed"],
        seed=CONFIG["seed"],
    )

    agent = D3QN(
        env=env,
        seed=CONFIG["seed"],
        gamma=CONFIG["gamma"],
        tau=CONFIG["tau"],
        bufferSize=CONFIG["bufferSize"],
        batchSize=CONFIG["batchSize"],
        optimizerLR=CONFIG["optimizerLR"],
        updateFrequency=CONFIG["updateFrequency"],
        max_train_eps=CONFIG["max_train_eps"],
        max_eval_eps=CONFIG["max_eval_eps"],
    )

    print("Starting Dueling Double Deep Q-Learning (D3QN)...")
    final_reward = agent.runD3QN()
    print(f"\nTraining complete! Final Evaluation Reward: {final_reward:.2f}")

    save_path = CONFIG["policy_file"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(agent.onlineNet.state_dict(), save_path)
    print(f"Model weights saved to '{save_path}'.")
