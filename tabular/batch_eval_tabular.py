import os
import glob
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
import concurrent.futures
from typing import Sequence
from obelix import OBELIX

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("batch_evaluation_tabular.log", mode="a"),
        logging.StreamHandler(),
    ],
)


def make_policy_fn(policy_dict):
    def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
        state = tuple(obs.astype(int))
        if state in policy_dict:
            action_data = policy_dict[state]
            if isinstance(action_data, np.ndarray):
                return ACTIONS[int(np.argmax(action_data))]
            else:
                return action_data
        else:
            probs = np.array([0.05, 0.10, 0.70, 0.10, 0.05], dtype=float)
            return ACTIONS[int(rng.choice(len(ACTIONS), p=probs))]

    return policy


def get_raw_scores(policy_fn, env_config, runs, base_seed):
    scores = []
    env = OBELIX(**env_config)

    for i in range(runs):
        seed = base_seed + i
        obs = env.reset(seed=seed)
        rng = np.random.default_rng(seed)

        total_reward = 0.0
        done = False

        while not done:
            action = policy_fn(obs, rng)
            obs, reward, done = env.step(action, render=False)
            total_reward += float(reward)

        scores.append(total_reward)

    return scores


def evaluate_single_file(pkl_path, config_with_obs, config_no_obs, runs, base_seed):
    filename = os.path.basename(pkl_path)
    try:
        with open(pkl_path, "rb") as f:
            policy_dict = pickle.load(f)

        policy_fn = make_policy_fn(policy_dict)

        scores_with = get_raw_scores(policy_fn, config_with_obs, runs=runs, base_seed=base_seed)
        scores_no = get_raw_scores(policy_fn, config_no_obs, runs=runs, base_seed=base_seed)
        scores_combined = scores_with + scores_no

        return {
            "Algorithm": filename,
            "WallObs Mean": np.mean(scores_with),
            "WallObs Std": np.std(scores_with),
            "NoWall Mean": np.mean(scores_no),
            "NoWall Std": np.std(scores_no),
            "Comb Mean": np.mean(scores_combined),
            "Comb Std": np.std(scores_combined),
        }
    except Exception as e:
        return {"Algorithm": filename, "Error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate tabular RL policies in parallel.")
    parser.add_argument(
        "--weights_dir", type=str, default="weights", help="Directory containing .pkl policy files"
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of runs per condition (with/without obstacles)"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of parallel processes to run"
    )
    parser.add_argument("--seed", type=int, default=200, help="Base seed for evaluation")
    args = parser.parse_args()

    config_with_obs = {
        "scaling_factor": 5,
        "arena_size": 500,
        "max_steps": 1000,
        "wall_obstacles": True,
        "difficulty": 0,
        "box_speed": 2,
    }

    config_no_obs = {
        "scaling_factor": 5,
        "arena_size": 500,
        "max_steps": 1000,
        "wall_obstacles": False,
        "difficulty": 0,
        "box_speed": 2,
    }

    search_pattern = os.path.join(args.weights_dir, "**", "*.pkl")
    pkl_files = glob.glob(search_pattern, recursive=True)

    if not pkl_files:
        logging.error(f"No .pkl files found in '{args.weights_dir}'.")
        return

    logging.info(
        f"Found {len(pkl_files)} policy files. Starting batch evaluation with {args.workers} workers..."
    )

    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_file, pkl_path, config_with_obs, config_no_obs, args.runs, args.seed
            ): pkl_path
            for pkl_path in pkl_files
        }

        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            res = future.result()
            if "Error" in res:
                logging.error(
                    f"[{count}/{len(pkl_files)}] FAILED: {res['Algorithm']} -> {res['Error']}"
                )
            else:
                results.append(res)
                logging.info(
                    f"[{count}/{len(pkl_files)}] Evaluated: {res['Algorithm']} (Comb Mean: {res['Comb Mean']:.2f})"
                )

    if not results:
        logging.warning("No successful evaluations to display.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by="Comb Mean", ascending=False).reset_index(drop=True)
    df = df.round(2)

    output_csv = "eval_results_tabular.csv"
    df.to_csv(output_csv, index=False)

    logging.info("\n" + "=" * 115)
    logging.info("FINAL EVALUATION RESULTS (Sorted by Combined Mean)")
    logging.info("=" * 115)
    logging.info("\n" + df.to_string(index=False))
    logging.info("=" * 115)
    logging.info(f"Results saved to '{output_csv}'.")


if __name__ == "__main__":
    main()
