import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
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
    "gae_lambda": 0.95,
    "max_train_eps": 1000,
    "max_eval_eps": 1,
    "optimizerLR": 3e-4,
    "maxGradNorm": 0.5,
    "clip_coef": 0.2,
    "clip_vloss": True,
    "entropy_coeff": 0.01,
    "vf_coeff": 0.5,
    "ppo_epochs": 4,
    "num_minibatches": 4,
    "rollout_steps": 2048,
    "anneal_lr": True,
    "target_kl": None,
    "norm_adv": True,
    "policy_file": "weightsPth/PPO_v1.pth",
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


class PPOAgent(nn.Module):
    """Single module containing both actor and critic, matching reference."""

    def __init__(self, stateDim, numActions, hDim=[64, 64]):
        super().__init__()

        critic_layers = []
        current_dim = stateDim
        for h in hDim:
            critic_layers.append(nn.Linear(current_dim, h))
            critic_layers.append(nn.Tanh())
            current_dim = h
        critic_layers.append(nn.Linear(current_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        actor_layers = []
        current_dim = stateDim
        for h in hDim:
            actor_layers.append(nn.Linear(current_dim, h))
            actor_layers.append(nn.Tanh())
            current_dim = h
        actor_layers.append(nn.Linear(current_dim, numActions))
        self.actor = nn.Sequential(*actor_layers)

    def get_value(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        value = self.critic(x)
        logits = self.actor(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value


class RolloutBuffer:
    def __init__(self, num_steps, obs_dim):
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.pos = 0

        self.states = torch.zeros((num_steps, obs_dim), dtype=torch.float32)
        self.actions = torch.zeros((num_steps,), dtype=torch.int64)
        self.log_probs = torch.zeros((num_steps,), dtype=torch.float32)
        self.rewards = torch.zeros((num_steps,), dtype=torch.float32)
        self.dones = torch.zeros((num_steps,), dtype=torch.float32)
        self.values = torch.zeros((num_steps,), dtype=torch.float32)

    def store(self, state, action, log_prob, reward, done, value):
        self.states[self.pos] = torch.tensor(state, dtype=torch.float32)
        self.actions[self.pos] = action
        self.log_probs[self.pos] = log_prob
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)
        self.values[self.pos] = value
        self.pos += 1

    def is_full(self):
        return self.pos >= self.num_steps

    def reset(self):
        self.pos = 0


class PPO:
    def __init__(
        self,
        env,
        seed,
        gamma,
        gae_lambda,
        clip_coef,
        clip_vloss,
        entropy_coeff,
        vf_coeff,
        ppo_epochs,
        num_minibatches,
        rollout_steps,
        optimizerLR,
        maxGradNorm,
        max_train_eps,
        max_eval_eps,
        anneal_lr=True,
        target_kl=None,
        norm_adv=True,
        save_path=None,
    ):
        self.env = env
        self.seed = seed
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.entropy_coeff = entropy_coeff
        self.vf_coeff = vf_coeff
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.rollout_steps = rollout_steps
        self.base_lr = optimizerLR
        self.maxGradNorm = maxGradNorm
        self.MAX_TRAIN_EPISODES = max_train_eps
        self.MAX_EVAL_EPISODES = max_eval_eps
        self.anneal_lr = anneal_lr
        self.target_kl = target_kl
        self.norm_adv = norm_adv
        self.save_path = save_path

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.stateDim = 18
        self.numActions = len(ACTIONS)
        self.batch_size = self.rollout_steps
        self.minibatch_size = self.batch_size // self.num_minibatches

        self.agent = PPOAgent(self.stateDim, self.numActions)
        self.optimizer = Adam(self.agent.parameters(), lr=optimizerLR, eps=1e-5)

        self.rolloutBuffer = RolloutBuffer(self.rollout_steps, self.stateDim)
        self.initBookKeeping()

    def initBookKeeping(self):
        self.trainRewardsList = []
        self.evalRewardsList = []
        self.current_ep_eval_reward = 0
        self.best_eval_reward = -np.inf

    def performBookKeeping(self, ep_reward):
        self.trainRewardsList.append(ep_reward)
        self.evalRewardsList.append(self.current_ep_eval_reward)

    def _save_best(self, eval_reward):
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            if self.save_path is not None:
                save_dir = os.path.dirname(self.save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                torch.save(self.agent.actor.state_dict(), self.save_path)

    def _save_last(self):
        if self.save_path is not None:
            last_path = self.save_path.replace(".pth", "_last.pth")
            save_dir = os.path.dirname(last_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(self.agent.actor.state_dict(), last_path)

    def compute_gae(self, next_obs, next_done):
        with torch.no_grad():
            next_value = self.agent.get_value(
                torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            ).squeeze()

            advantages = torch.zeros(self.rollout_steps)
            lastgaelam = 0.0

            for t in reversed(range(self.rollout_steps)):
                if t == self.rollout_steps - 1:
                    next_non_terminal = 1.0 - next_done
                    next_val = next_value
                else:
                    next_non_terminal = 1.0 - self.rolloutBuffer.dones[t + 1]
                    next_val = self.rolloutBuffer.values[t + 1]

                delta = (
                    self.rolloutBuffer.rewards[t]
                    + self.gamma * next_val * next_non_terminal
                    - self.rolloutBuffer.values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + self.gamma * self.gae_lambda * next_non_terminal * lastgaelam
                )

            returns = advantages + self.rolloutBuffer.values

        return advantages, returns

    def update(self, advantages, returns, update_idx, num_updates):
        if self.anneal_lr:
            frac = 1.0 - (update_idx / num_updates)
            lr_now = frac * self.base_lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr_now

        b_obs = self.rolloutBuffer.states
        b_actions = self.rolloutBuffer.actions
        b_logprobs = self.rolloutBuffer.log_probs
        b_advantages = advantages
        b_returns = returns
        b_values = self.rolloutBuffer.values

        if self.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        b_inds = np.arange(self.batch_size)

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.squeeze()
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - self.entropy_coeff * entropy_loss + self.vf_coeff * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.maxGradNorm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

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
        self.agent.eval()
        for instance in range(num_eval_instances):
            instance_seed = self.seed + 1000 + instance * 100
            for eval_ep in range(self.MAX_EVAL_EPISODES):
                state = self.env.reset(seed=instance_seed + eval_ep)
                done = False
                ep_reward = 0
                while not done:
                    with torch.no_grad():
                        logits = self.agent.actor(
                            torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        )
                    action_idx = torch.argmax(logits, dim=1).item()
                    action_str = ACTIONS[action_idx]
                    next_state, reward, done = self.env.step(action_str, render=False)
                    state = next_state
                    ep_reward += reward
                all_scores.append(ep_reward)
        self.agent.train()
        return all_scores

    def trainAgent(self):
        total_timesteps_approx = self.MAX_TRAIN_EPISODES * self.env.max_steps
        num_updates = total_timesteps_approx // self.rollout_steps
        update_count = 0

        state = self.env.reset(seed=self.seed)
        done = False
        ep_reward = 0
        episode = 0

        pbar = tqdm(range(self.MAX_TRAIN_EPISODES), desc="Training PPO", unit="ep")

        while episode < self.MAX_TRAIN_EPISODES:
            self.rolloutBuffer.reset()

            for step in range(self.rollout_steps):
                with torch.no_grad():
                    obs_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action_t, logprob, _, value = self.agent.get_action_and_value(obs_t)

                action_idx = action_t.item()
                action_str = ACTIONS[action_idx]
                next_state, reward, done = self.env.step(action_str, render=False)

                self.rolloutBuffer.store(
                    state, action_idx, logprob.item(), reward, done, value.squeeze().item()
                )

                state = next_state
                ep_reward += reward

                if done:
                    self.performBookKeeping(ep_reward)
                    episode += 1
                    pbar.update(1)

                    if episode % self.eval_frequency(episode) == 0:
                        eval_rewards = self.evaluateAgent(num_eval_instances=3)
                        self.current_ep_eval_reward = np.mean(eval_rewards)
                        self._save_best(self.current_ep_eval_reward)

                    pbar.set_postfix(
                        {
                            "Train Rwd": f"{ep_reward:.1f}",
                            "Eval Rwd": f"{self.current_ep_eval_reward:.1f}",
                            "Best": f"{self.best_eval_reward:.1f}",
                        }
                    )

                    ep_reward = 0

                    if episode >= self.MAX_TRAIN_EPISODES:
                        break

                    state = self.env.reset(seed=self.seed + episode)
                    done = False

            next_done = 1.0 if done else 0.0
            advantages, returns = self.compute_gae(state, next_done)

            self.update(advantages, returns, update_count, max(num_updates, 1))
            update_count += 1

        pbar.close()
        self._save_last()
        return self.trainRewardsList, self.evalRewardsList

    def runPPO(self):
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

    agent = PPO(
        env=env,
        seed=CONFIG["seed"],
        gamma=CONFIG["gamma"],
        gae_lambda=CONFIG["gae_lambda"],
        clip_coef=CONFIG["clip_coef"],
        clip_vloss=CONFIG["clip_vloss"],
        entropy_coeff=CONFIG["entropy_coeff"],
        vf_coeff=CONFIG["vf_coeff"],
        ppo_epochs=CONFIG["ppo_epochs"],
        num_minibatches=CONFIG["num_minibatches"],
        rollout_steps=CONFIG["rollout_steps"],
        optimizerLR=CONFIG["optimizerLR"],
        maxGradNorm=CONFIG["maxGradNorm"],
        max_train_eps=CONFIG["max_train_eps"],
        max_eval_eps=CONFIG["max_eval_eps"],
        anneal_lr=CONFIG["anneal_lr"],
        target_kl=CONFIG["target_kl"],
        norm_adv=CONFIG["norm_adv"],
        save_path=CONFIG["policy_file"],
    )

    print("Starting PPO Training...")
    best_reward = agent.runPPO()
    print(f"\nTraining complete!")
    print(f"Best Eval Reward : {best_reward:.2f} → {CONFIG['policy_file']}")
    print(f"Last Weights     : {CONFIG['policy_file'].replace('.pth', '_last.pth')}")
