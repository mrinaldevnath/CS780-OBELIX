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
    "alpha_init": 0.5,
    "min_alpha": 0.01,
    "alpha_decay": 0.99,
    "epsilon_init": 1.0,
    "min_epsilon": 0.05,
    "epsilon_decay": 0.99,
    "max_episodes": 500,
    "replace_trace": False,
    "policy_file": "weights/qlambda_v1.pkl",
}


def decay_learning_rate(e, initial_alpha, min_alpha, decay_rate):
    return max(min_alpha, initial_alpha * (decay_rate**e))


def decay_epsilon(e, initial_epsilon, min_epsilon, decay_rate):
    return max(min_epsilon, initial_epsilon * (decay_rate**e))


def action_select(s, Q, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(0, len(ACTIONS))
    else:
        q_values = Q[s]
        max_val = np.max(q_values)
        best_actions = np.where(q_values == max_val)[0]
        return rng.choice(best_actions)


def q_learning_lambda(env, config):
    Q = defaultdict(lambda: np.zeros(len(ACTIONS)))
    rng = np.random.default_rng(config["seed"])

    trace_threshold = 1e-4

    for e in tqdm(range(config["max_episodes"]), desc="Q(λ) Training", unit="ep"):
        alpha = decay_learning_rate(
            e, config["alpha_init"], config["min_alpha"], config["alpha_decay"]
        )
        epsilon = decay_epsilon(
            e, config["epsilon_init"], config["min_epsilon"], config["epsilon_decay"]
        )

        E = defaultdict(lambda: np.zeros(len(ACTIONS)))

        obs = env.reset(seed=config["seed"] + e)
        s = tuple(obs.astype(int))
        done = False

        a_idx = action_select(s, Q, epsilon, rng)

        while not done:
            action_str = ACTIONS[a_idx]
            next_obs, r, done = env.step(action_str, render=False)
            s_prime = tuple(next_obs.astype(int))

            a_prime_idx = action_select(s_prime, Q, epsilon, rng)

            max_q_val = np.max(Q[s_prime])
            s_prime_greedy = Q[s_prime][a_prime_idx] == max_q_val

            td_target = r
            if not done:
                td_target += config["gamma"] * max_q_val

            td_error = td_target - Q[s][a_idx]

            if config.get("replace_trace", True):
                E[s][:] = 0
            E[s][a_idx] += 1

            for state in list(E.keys()):
                Q[state] += alpha * td_error * E[state]

                if s_prime_greedy:
                    E[state] *= config["gamma"] * config.get("lambda_param", 0.9)

                    if np.max(E[state]) < trace_threshold:
                        del E[state]
                else:
                    E[state][:] = 0

            if not s_prime_greedy:
                E.clear()

            s = s_prime
            a_idx = a_prime_idx

    pi = {state: ACTIONS[np.argmax(values)] for state, values in Q.items()}
    return Q, pi


if __name__ == "__main__":
    env = OBELIX(
        scaling_factor=CONFIG["scaling_factor"],
        arena_size=CONFIG["arena_size"],
        max_steps=CONFIG["max_steps"],
        wall_obstacles=CONFIG["wall_obstacles"],
        difficulty=CONFIG["difficulty"],
        box_speed=CONFIG["box_speed"],
    )

    Q, pi = q_learning_lambda(env, CONFIG)

    with open(CONFIG["policy_file"], "wb") as f:
        pickle.dump(pi, f)

    print(f"\nTraining complete! Policy saved to '{CONFIG['policy_file']}'.")
