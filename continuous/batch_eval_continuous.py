import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2

cv2.setNumThreads(0)

import glob
import argparse
import logging
import pandas as pd
import numpy as np
import concurrent.futures
import torch
import torch.nn as nn
import torch.nn.functional as F

from obelix import OBELIX

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

ALGO_NAMES = ["TD3", "SAC", "PPO"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("batch_evaluation_actorcritic.log", mode="a"),
        logging.StreamHandler(),
    ],
)


class ReLUPolicyNetwork(nn.Module):
    """Policy network used by TD3 and SAC (ReLU activations)."""

    def __init__(self, in_dim, out_dim, h_dim=[64, 64]):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = in_dim
        for h in h_dim:
            self.layers.append(nn.Linear(current_dim, h))
            current_dim = h
        self.out_layer = nn.Linear(current_dim, out_dim)

    def forward(self, s):
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        for layer in self.layers:
            s = F.relu(layer(s))
        return self.out_layer(s)


def get_policy_network(algo_name, in_dim, out_dim):
    if algo_name in ["TD3", "SAC"]:
        return ReLUPolicyNetwork(in_dim, out_dim)
    elif algo_name == "PPO":
        return nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, out_dim),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def infer_algo_name(filename):
    stem = os.path.splitext(filename)[0]
    for name in ALGO_NAMES:
        if stem.startswith(name):
            return name
    return None


def greedy_action(net, state):
    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = net(state_t)
    return torch.argmax(logits, dim=1).item()


def run_episodes(net, env_config, runs, base_seed):
    env = OBELIX(**env_config)
    scores = []
    for i in range(runs):
        state = env.reset(seed=base_seed + i)
        done = False
        ep_reward = 0.0
        while not done:
            a_idx = greedy_action(net, state)
            state, reward, done = env.step(ACTIONS[a_idx], render=False)
            ep_reward += float(reward)
        scores.append(ep_reward)
    return scores


def eval_worker(task_spec):
    pth_path = task_spec["pth_path"]
    algo_name = task_spec["algo_name"]
    runs = task_spec["runs"]
    base_seed = task_spec["base_seed"]
    config_with_obs = task_spec["config_with_obs"]
    config_no_obs = task_spec["config_no_obs"]

    filename = os.path.basename(pth_path)
    in_dim = 18
    out_dim = len(ACTIONS)

    try:
        net = get_policy_network(algo_name, in_dim, out_dim)
        state_dict = torch.load(pth_path, map_location=torch.device("cpu"))
        net.load_state_dict(state_dict)
        net.eval()

        scores_with = run_episodes(net, config_with_obs, runs, base_seed)
        scores_no = run_episodes(net, config_no_obs, runs, base_seed)
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
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(
        description="Batch evaluate TD3, SAC, and PPO policies concurrently."
    )
    parser.add_argument(
        "--weights_dir", type=str, default="weightsPth", help="Directory containing .pth files"
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of parallel processes"
    )
    parser.add_argument("--runs", type=int, default=10, help="Number of episodes per condition")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for evaluation")
    args = parser.parse_args()

    config_with_obs = {
        "scaling_factor": 5,
        "arena_size": 500,
        "max_steps": 1000,
        "wall_obstacles": True,
        "difficulty": 3,
        "box_speed": 2,
    }

    config_no_obs = {
        "scaling_factor": 5,
        "arena_size": 500,
        "max_steps": 1000,
        "wall_obstacles": False,
        "difficulty": 3,
        "box_speed": 2,
    }

    search_pattern = os.path.join(args.weights_dir, "**", "*.pth")
    pth_files = glob.glob(search_pattern, recursive=True)

    if not pth_files:
        logging.error(f"No .pth files found in '{args.weights_dir}'.")
        return

    tasks = []
    skipped = 0
    for pth_path in pth_files:
        filename = os.path.basename(pth_path)
        algo_name = infer_algo_name(filename)
        if algo_name is None:
            logging.warning(f"Skipping '{filename}': could not infer algorithm name.")
            skipped += 1
            continue
        tasks.append(
            {
                "pth_path": pth_path,
                "algo_name": algo_name,
                "runs": args.runs,
                "base_seed": args.seed,
                "config_with_obs": config_with_obs,
                "config_no_obs": config_no_obs,
            }
        )

    logging.info(
        f"Found {len(pth_files)} .pth files ({skipped} skipped, {len(tasks)} to evaluate)."
    )
    logging.info(f"Starting parallel evaluation with {args.workers} workers...")

    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(eval_worker, task): task for task in tasks}

        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            res = future.result()
            if "Error" in res:
                logging.error(
                    f"[{count}/{len(tasks)}] FAILED: {res['Algorithm']} -> {res['Error']}"
                )
            else:
                results.append(res)
                logging.info(
                    f"[{count}/{len(tasks)}] Evaluated: {res['Algorithm']} "
                    f"(Comb Mean: {res['Comb Mean']:.2f})"
                )

    if not results:
        logging.warning("No successful evaluations to display.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by="Comb Mean", ascending=False).reset_index(drop=True)
    df = df.round(2)

    output_csv = "eval_results_actorcritic.csv"
    df.to_csv(output_csv, index=False)

    logging.info("\n" + "=" * 115)
    logging.info("FINAL EVALUATION RESULTS (Sorted by Combined Mean)")
    logging.info("=" * 115)
    logging.info("\n" + df.to_string(index=False))
    logging.info("=" * 115)
    logging.info(f"Results saved to '{output_csv}'.")


if __name__ == "__main__":
    main()
