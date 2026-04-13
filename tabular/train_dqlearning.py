import numpy as np
import pickle
from collections import defaultdict
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
    "alpha_init": 0.1,
    "min_alpha": 0.01,
    "alpha_decay": 0.99,
    "epsilon_init": 1.0,
    "min_epsilon": 0.05,
    "epsilon_decay": 0.99,
    "max_episodes": 500,
    "policy_file": "weights/dqlearning_v1.pkl",
}


def decay_learning_rate(e, initial_alpha, min_alpha, decay_rate):
    return max(min_alpha, initial_alpha * (decay_rate**e))


def decay_epsilon(e, initial_epsilon, min_epsilon, decay_rate):
    return max(min_epsilon, initial_epsilon * (decay_rate**e))


def argmax_tie_break(q_values, rng):
    max_val = np.max(q_values)
    best_actions = np.where(q_values == max_val)[0]
    return rng.choice(best_actions)


def action_select(s, Q1, Q2, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(0, len(ACTIONS))
    else:
        combined_q = Q1[s] + Q2[s]
        return argmax_tie_break(combined_q, rng)


def double_q_learning(env, config):
    Q1 = defaultdict(lambda: np.zeros(len(ACTIONS)))
    Q2 = defaultdict(lambda: np.zeros(len(ACTIONS)))

    rng = np.random.default_rng(config["seed"])

    for e in tqdm(range(config["max_episodes"]), desc="Double Q-Learning", unit="ep"):
        alpha = decay_learning_rate(
            e, config["alpha_init"], config["min_alpha"], config["alpha_decay"]
        )
        epsilon = decay_epsilon(
            e, config["epsilon_init"], config["min_epsilon"], config["epsilon_decay"]
        )

        obs = env.reset(seed=config["seed"] + e)
        s = tuple(obs.astype(int))
        done = False

        while not done:
            a_idx = action_select(s, Q1, Q2, epsilon, rng)
            action_str = ACTIONS[a_idx]

            next_obs, r, done = env.step(action_str, render=False)
            s_prime = tuple(next_obs.astype(int))

            if rng.integers(2) == 0:
                # Update Q1
                a_Q1 = argmax_tie_break(Q1[s_prime], rng)
                td_target = r
                if not done:
                    td_target += config["gamma"] * Q2[s_prime][a_Q1]

                td_error = td_target - Q1[s][a_idx]
                Q1[s][a_idx] += alpha * td_error
            else:
                # Update Q2
                a_Q2 = argmax_tie_break(Q2[s_prime], rng)
                td_target = r
                if not done:
                    td_target += config["gamma"] * Q1[s_prime][a_Q2]

                td_error = td_target - Q2[s][a_idx]
                Q2[s][a_idx] += alpha * td_error

            s = s_prime

    all_states = set(Q1.keys()).union(set(Q2.keys()))
    pi = {state: ACTIONS[np.argmax(Q1[state] + Q2[state])] for state in all_states}

    return Q1, Q2, pi


if __name__ == "__main__":
    env = OBELIX(
        scaling_factor=CONFIG["scaling_factor"],
        arena_size=CONFIG["arena_size"],
        max_steps=CONFIG["max_steps"],
        wall_obstacles=CONFIG["wall_obstacles"],
        difficulty=CONFIG["difficulty"],
        box_speed=CONFIG["box_speed"],
    )

    Q1, Q2, pi = double_q_learning(env, CONFIG)

    with open(CONFIG["policy_file"], "wb") as f:
        pickle.dump(pi, f)

    print(f"\nTraining complete! Policy saved to '{CONFIG['policy_file']}'.")
