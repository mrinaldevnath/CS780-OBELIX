import os
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
    "tau": 0.005,
    "max_train_eps": 1000,
    "max_eval_eps": 1,
    "bufferSize": 100000,
    "batchSize": 256,
    "minSamples": 1000,
    "policyOptimizerLR": 1e-3,
    "valueOptimizerLR": 1e-3,
    "epsilon_init": 1.0,
    "min_epsilon": 0.05,
    "epsilon_decay": 0.99,
    "trainPolicyFrequency": 2,
    "updateFrequencyPolicy": 2,
    "updateFrequencyValue": 2,
    "maxGradNorm": 1.0,
    "gumbel_temp_init": 1.0,
    "gumbel_temp_min": 0.1,
    "gumbel_temp_decay": 0.99,
    "policy_file": "weightsPth/TD3_v1.pth",
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


def decay_temperature(e, temp_init, temp_min, temp_decay):
    return max(temp_min, temp_init * (temp_decay**e))


def gumbel_softmax_sample(logits, temperature):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-20)))
    return F.softmax((logits + gumbel_noise) / temperature, dim=-1)


def gumbel_softmax_straight_through(logits, temperature):
    soft = gumbel_softmax_sample(logits, temperature)
    hard = torch.zeros_like(soft).scatter_(1, soft.argmax(dim=1, keepdim=True), 1.0)
    return (hard - soft).detach() + soft


class ValueNetwork(nn.Module):
    def __init__(self, inDim, outDim=1, hDim=[64, 64]):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = inDim
        for h in hDim:
            self.layers.append(nn.Linear(current_dim, h))
            current_dim = h
        self.out_layer = nn.Linear(current_dim, outDim)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out_layer(x)


class PolicyNetwork(nn.Module):
    def __init__(self, inDim, outDim, hDim=[64, 64]):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = inDim
        for h in hDim:
            self.layers.append(nn.Linear(current_dim, h))
            current_dim = h
        self.out_layer = nn.Linear(current_dim, outDim)

    def forward(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        for layer in self.layers:
            s = F.relu(layer(s))
        return self.out_layer(s)


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
            self.states = np.zeros((self.buffer_size, *np.shape(state)), dtype=np.float32)
            self.actions = np.zeros((self.buffer_size,), dtype=np.int64)
            self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
            self.next_states = np.zeros((self.buffer_size, *np.shape(next_state)), dtype=np.float32)
            self.dones = np.zeros((self.buffer_size,), dtype=bool)
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batchSize):
        indices = np.random.choice(self.size, batchSize, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def length(self):
        return self.size


def epsilon_greedy_train(net, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(len(ACTIONS))
    with torch.no_grad():
        logits = net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
    return torch.argmax(logits, dim=1).item()


def greedy_eval(net, state):
    with torch.no_grad():
        logits = net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
    return torch.argmax(logits, dim=1).item()


class TD3:
    def __init__(
        self,
        env,
        seed,
        gamma,
        tau,
        bufferSize,
        batchSize,
        minSamples,
        updateFrequencyPolicy,
        updateFrequencyValue,
        trainPolicyFrequency,
        policyOptimizerLR,
        valueOptimizerLR,
        maxGradNorm,
        max_train_eps,
        max_eval_eps,
        epsilon_init=1.0,
        min_epsilon=0.05,
        epsilon_decay=0.99,
        gumbel_temp_init=1.0,
        gumbel_temp_min=0.1,
        gumbel_temp_decay=0.995,
        save_path=None,
    ):
        self.env = env
        self.seed = seed
        self.gamma = gamma
        self.tau = tau
        self.bufferSize = bufferSize
        self.batchSize = batchSize
        self.minSamples = minSamples
        self.updateFrequencyPolicy = updateFrequencyPolicy
        self.updateFrequencyValue = updateFrequencyValue
        self.trainPolicyFrequency = trainPolicyFrequency
        self.maxGradNorm = maxGradNorm
        self.MAX_TRAIN_EPISODES = max_train_eps
        self.MAX_EVAL_EPISODES = max_eval_eps
        self.save_path = save_path
        self.epsilon_init = epsilon_init
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gumbel_temp_init = gumbel_temp_init
        self.gumbel_temp_min = gumbel_temp_min
        self.gumbel_temp_decay = gumbel_temp_decay

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.stateDim = 18
        self.numActions = len(ACTIONS)

        self.q1_online = ValueNetwork(self.stateDim + self.numActions, 1)
        self.q1_target = ValueNetwork(self.stateDim + self.numActions, 1)
        self.q2_online = ValueNetwork(self.stateDim + self.numActions, 1)
        self.q2_target = ValueNetwork(self.stateDim + self.numActions, 1)

        self.onlinePolicyNetwork = PolicyNetwork(self.stateDim, self.numActions)
        self.targetPolicyNetwork = PolicyNetwork(self.stateDim, self.numActions)

        self._hardUpdate(self.q1_online, self.q1_target)
        self._hardUpdate(self.q2_online, self.q2_target)
        self._hardUpdate(self.onlinePolicyNetwork, self.targetPolicyNetwork)

        self.q1_optimizer = Adam(self.q1_online.parameters(), lr=valueOptimizerLR)
        self.q2_optimizer = Adam(self.q2_online.parameters(), lr=valueOptimizerLR)
        self.policyOptimizer = Adam(self.onlinePolicyNetwork.parameters(), lr=policyOptimizerLR)

        self.rBuffer = ReplayBuffer(self.bufferSize, self.batchSize, self.seed)
        self.initBookKeeping()

    def initBookKeeping(self):
        self.trainRewardsList = []
        self.evalRewardsList = []
        self.current_ep_train_reward = 0
        self.current_ep_eval_reward = 0
        self.best_eval_reward = -np.inf

    def performBookKeeping(self, train=True):
        if train:
            self.trainRewardsList.append(self.current_ep_train_reward)
            self.evalRewardsList.append(self.current_ep_eval_reward)

    def _hardUpdate(self, onlineNet, targetNet):
        for tParam, oParam in zip(targetNet.parameters(), onlineNet.parameters()):
            tParam.data.copy_(oParam.data)

    def _softUpdate(self, onlineNet, targetNet):
        for tParam, oParam in zip(targetNet.parameters(), onlineNet.parameters()):
            tParam.data.copy_((1 - self.tau) * tParam.data + self.tau * oParam.data)

    def _save_best(self, eval_reward):
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            if self.save_path is not None:
                save_dir = os.path.dirname(self.save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                torch.save(self.onlinePolicyNetwork.state_dict(), self.save_path)

    def _save_last(self):
        if self.save_path is not None:
            last_path = self.save_path.replace(".pth", "_last.pth")
            save_dir = os.path.dirname(last_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(self.onlinePolicyNetwork.state_dict(), last_path)

    def trainNetwork(self, experiences, global_step, temperature):
        states, actions, rewards, next_states, dones = experiences

        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            target_next_logits = self.targetPolicyNetwork(next_states_t)
            soft_next_actions = gumbel_softmax_sample(target_next_logits, temperature)
            sa_next = torch.cat([next_states_t, soft_next_actions], dim=1)

            q1_next = self.q1_target(sa_next)
            q2_next = self.q2_target(sa_next)
            min_target_q = torch.min(q1_next, q2_next)
            td_targets = rewards_t + self.gamma * min_target_q * (1.0 - dones_t)

        one_hot = torch.zeros(states_t.size(0), self.numActions, dtype=torch.float32).scatter_(
            1, actions_t, 1.0
        )
        sa_input = torch.cat([states_t, one_hot], dim=1)

        q1_pred = self.q1_online(sa_input)
        q2_pred = self.q2_online(sa_input)

        q1_loss = F.mse_loss(q1_pred, td_targets)
        q2_loss = F.mse_loss(q2_pred, td_targets)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q1_online.parameters(), self.maxGradNorm)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q2_online.parameters(), self.maxGradNorm)
        self.q2_optimizer.step()

        if global_step % self.trainPolicyFrequency == 0:
            online_logits = self.onlinePolicyNetwork(states_t)
            st_actions = gumbel_softmax_straight_through(online_logits, temperature)
            sa_policy = torch.cat([states_t, st_actions], dim=1)

            policyLoss = -self.q1_online(sa_policy).mean()

            self.policyOptimizer.zero_grad()
            policyLoss.backward()
            torch.nn.utils.clip_grad_norm_(self.onlinePolicyNetwork.parameters(), self.maxGradNorm)
            self.policyOptimizer.step()

        if global_step % self.updateFrequencyValue == 0:
            self._softUpdate(self.q1_online, self.q1_target)
            self._softUpdate(self.q2_online, self.q2_target)
        if global_step % self.updateFrequencyPolicy == 0:
            self._softUpdate(self.onlinePolicyNetwork, self.targetPolicyNetwork)

    def eval_frequency(self, episode):
        if episode < 100:
            return 5
        elif episode < 300:
            return 10
        elif episode < 600:
            return 20
        else:
            return 30

    def evaluateAgent(self, num_eval_instances=3):
        all_scores = []
        for instance in range(num_eval_instances):
            instance_seed = self.seed + 1000 + instance * 100
            for eval_ep in range(self.MAX_EVAL_EPISODES):
                state = self.env.reset(seed=instance_seed + eval_ep)
                done = False
                ep_reward = 0
                while not done:
                    action_idx = greedy_eval(self.onlinePolicyNetwork, state)
                    action_str = ACTIONS[action_idx]
                    next_state, reward, done = self.env.step(action_str, render=False)
                    state = next_state
                    ep_reward += reward
                all_scores.append(ep_reward)
        return all_scores

    def trainAgent(self):
        global_step = 0
        eval_rewards = [0.0]

        pbar = tqdm(range(self.MAX_TRAIN_EPISODES), desc="Training TD3", unit="ep")
        for episode in pbar:
            epsilon = decay_epsilon(
                episode, self.epsilon_init, self.min_epsilon, self.epsilon_decay
            )
            temperature = decay_temperature(
                episode, self.gumbel_temp_init, self.gumbel_temp_min, self.gumbel_temp_decay
            )

            state = self.env.reset(seed=self.seed + episode)
            done = False
            ep_reward = 0

            while not done:
                action_idx = epsilon_greedy_train(self.onlinePolicyNetwork, state, epsilon)
                action_str = ACTIONS[action_idx]
                next_state, reward, done = self.env.step(action_str, render=False)
                self.rBuffer.store(state, action_idx, reward, next_state, done)
                state = next_state
                ep_reward += reward
                global_step += 1

                if self.rBuffer.length() >= self.minSamples:
                    self.trainNetwork(self.rBuffer.sample(self.batchSize), global_step, temperature)

            if episode % self.eval_frequency(episode) == 0:
                eval_rewards = self.evaluateAgent(num_eval_instances=3)
                self.current_ep_eval_reward = np.mean(eval_rewards)
                self._save_best(self.current_ep_eval_reward)

            self.current_ep_train_reward = ep_reward
            self.performBookKeeping(train=True)

            pbar.set_postfix(
                {
                    "Eps": f"{epsilon:.3f}",
                    "Temp": f"{temperature:.3f}",
                    "Train Rwd": f"{ep_reward:.1f}",
                    "Eval Rwd": f"{self.current_ep_eval_reward:.1f}",
                    "Best": f"{self.best_eval_reward:.1f}",
                }
            )

        self._save_last()
        return self.trainRewardsList, self.evalRewardsList

    def runTD3(self):
        self.trainAgent()
        final_eval_scores = self.evaluateAgent(num_eval_instances=3)
        final_eval_mean = np.mean(final_eval_scores)
        self._save_best(final_eval_mean)
        self._save_last()
        return self.best_eval_reward


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

    agent = TD3(
        env=env,
        seed=CONFIG["seed"],
        gamma=CONFIG["gamma"],
        tau=CONFIG["tau"],
        bufferSize=CONFIG["bufferSize"],
        batchSize=CONFIG["batchSize"],
        minSamples=CONFIG["minSamples"],
        updateFrequencyPolicy=CONFIG["updateFrequencyPolicy"],
        updateFrequencyValue=CONFIG["updateFrequencyValue"],
        trainPolicyFrequency=CONFIG["trainPolicyFrequency"],
        policyOptimizerLR=CONFIG["policyOptimizerLR"],
        valueOptimizerLR=CONFIG["valueOptimizerLR"],
        maxGradNorm=CONFIG["maxGradNorm"],
        max_train_eps=CONFIG["max_train_eps"],
        max_eval_eps=CONFIG["max_eval_eps"],
        epsilon_init=CONFIG["epsilon_init"],
        min_epsilon=CONFIG["min_epsilon"],
        epsilon_decay=CONFIG["epsilon_decay"],
        gumbel_temp_init=CONFIG["gumbel_temp_init"],
        gumbel_temp_min=CONFIG["gumbel_temp_min"],
        gumbel_temp_decay=CONFIG["gumbel_temp_decay"],
        save_path=CONFIG["policy_file"],
    )

    print("Starting TD3 Training...")
    best_reward = agent.runTD3()
    print(f"\nTraining complete!")
    print(f"Best Eval Reward : {best_reward:.2f}  {CONFIG['policy_file']}")
    print(f"Last Weights     : {CONFIG['policy_file'].replace('.pth', '_last.pth')}")
