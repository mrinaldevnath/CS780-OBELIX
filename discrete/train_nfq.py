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
    "epochs": 5,
    "bufferSize": 100000,
    "batchSize": 256,
    "optimizerLR": 1e-3,
    "policy_file": "weightsPth/NFQ_v1.pth",
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


def createValueNetwork(inDim, outDim):
    return nn.Sequential(
        nn.Linear(inDim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, outDim)
    )


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
    def __init__(self, bufferSize, bufferType="NFQ"):
        self.bufferSize = bufferSize
        self.bufferType = bufferType
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


class NFQ:
    def __init__(
        self,
        env,
        seed,
        gamma,
        epochs,
        bufferSize,
        batchSize,
        optimizerLR,
        max_train_eps,
        max_eval_eps,
    ):
        self.env = env
        self.seed = seed
        self.gamma = gamma
        self.epochs = epochs
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.MAX_TRAIN_EPISODES = max_train_eps
        self.MAX_EVAL_EPISODES = max_eval_eps

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        inDim = 18
        outDim = len(ACTIONS)

        self.q_network = createValueNetwork(inDim, outDim)
        self.optimizer = Adam(self.q_network.parameters(), lr=optimizerLR)
        self.rBuffer = ReplayBuffer(self.bufferSize, bufferType="NFQ")

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
        self.start_wall_time = time.time()
        self.total_train_time = 0.0

        pbar = tqdm(range(self.MAX_TRAIN_EPISODES), desc="Training NFQ", unit="ep")
        for episode in pbar:
            epsilon = decay_epsilon(
                episode, CONFIG["epsilon_init"], CONFIG["min_epsilon"], CONFIG["epsilon_decay"]
            )

            state = self.env.reset(seed=self.seed + episode)
            done = False
            ep_reward = 0

            while not done:
                action_idx = epsilon_greedy_train(self.q_network, state, epsilon)
                action_str = ACTIONS[action_idx]

                next_state, reward, done = self.env.step(action_str, render=False)

                self.rBuffer.store((state, action_idx, reward, next_state, done))
                state = next_state
                ep_reward += reward

            if self.rBuffer.length() >= self.batchSize:
                experiences = self.rBuffer.sample(self.batchSize)
                self.trainNetwork(experiences, self.epochs)

            eval_rewards = self.evaluateAgent()
            self.current_ep_train_reward = ep_reward
            self.current_ep_eval_reward = np.mean(eval_rewards)
            self.performBookKeeping(train=True)

            pbar.set_postfix({"Eval Rwd": f"{self.current_ep_eval_reward:.1f}"})

        return self.trainRewardsList, self.evalRewardsList

    def trainNetwork(self, experiences, epochs):
        states, actions, rewards, next_states, dones = self.rBuffer.splitExperiences(experiences)

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            max_next_q = self.q_network(next_states_t).max(1, keepdim=True)[0]
            targets = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        for _ in range(epochs):
            q_values = self.q_network(states_t).gather(1, actions_t)
            loss = F.mse_loss(q_values, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluateAgent(self):
        finalEvalRewardsList = []
        for eval_ep in range(self.MAX_EVAL_EPISODES):
            state = self.env.reset(seed=self.seed + 1000 + eval_ep)
            done = False
            ep_reward = 0

            while not done:
                action_idx = greedy_eval(self.q_network, state)
                action_str = ACTIONS[action_idx]

                next_state, reward, done = self.env.step(action_str, render=False)
                state = next_state
                ep_reward += reward

            finalEvalRewardsList.append(ep_reward)
        return finalEvalRewardsList

    def runNFQ(self):
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

    agent = NFQ(
        env=env,
        seed=CONFIG["seed"],
        gamma=CONFIG["gamma"],
        epochs=CONFIG["epochs"],
        bufferSize=CONFIG["bufferSize"],
        batchSize=CONFIG["batchSize"],
        optimizerLR=CONFIG["optimizerLR"],
        max_train_eps=CONFIG["max_train_eps"],
        max_eval_eps=CONFIG["max_eval_eps"],
    )

    print("Starting Neural Fitted Q Iteration...")
    final_reward = agent.runNFQ()
    print(f"\nTraining complete! Final Evaluation Reward: {final_reward:.2f}")

    save_path = CONFIG["policy_file"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(agent.q_network.state_dict(), save_path)
    print(f"Model weights saved to '{save_path}'.")
