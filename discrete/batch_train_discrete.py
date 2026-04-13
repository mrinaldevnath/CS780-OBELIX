import os
import copy

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2

cv2.setNumThreads(0)

import itertools
import argparse
import logging
import pandas as pd
import concurrent.futures
import torch

from obelix import OBELIX

from train_nfq import NFQ
from train_dqn import DQN
from train_ddqn import DDQN
from train_d3qn import D3QN
from train_d3qn_per import D3QN_PER


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("batch_train_discrete.log", mode="a"),
        logging.StreamHandler(),
    ],
)


def worker_task(task_spec):
    """Executes a single training run for a Deep RL algorithm."""
    algo_name = task_spec["algorithm"]
    config = task_spec["config"]
    version = task_spec["version"]

    episodes = config["max_train_eps"]
    save_dir = os.path.join("weightsPth", str(episodes))
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{algo_name}_v{version}.pth")

    env = OBELIX(
        scaling_factor=config["scaling_factor"],
        arena_size=config["arena_size"],
        max_steps=config["max_steps"],
        wall_obstacles=config["wall_obstacles"],
        difficulty=config["difficulty"],
        box_speed=config["box_speed"],
        seed=config["seed"],
    )

    try:
        base_kwargs = {
            "env": env,
            "seed": config["seed"],
            "gamma": config["gamma"],
            "bufferSize": config["bufferSize"],
            "batchSize": config["batchSize"],
            "optimizerLR": config["optimizerLR"],
            "max_train_eps": config["max_train_eps"],
            "max_eval_eps": config["max_eval_eps"],
        }

        if algo_name == "NFQ":
            agent = NFQ(**base_kwargs, epochs=config.get("epochs", 5))
            final_eval = agent.runNFQ()
            torch.save(agent.q_network.state_dict(), file_path)

        elif algo_name == "DQN":
            agent = DQN(**base_kwargs, updateFrequency=config.get("updateFrequency", 10))
            final_eval = agent.runDQN()
            torch.save(agent.onlineNet.state_dict(), file_path)

        elif algo_name == "DDQN":
            agent = DDQN(**base_kwargs, updateFrequency=config.get("updateFrequency", 10))
            final_eval = agent.runDDQN()
            torch.save(agent.onlineNet.state_dict(), file_path)

        elif algo_name == "D3QN":
            agent = D3QN(
                **base_kwargs,
                updateFrequency=config.get("updateFrequency", 1),
                tau=config.get("tau", 0.05),
            )
            final_eval = agent.runD3QN()
            torch.save(agent.onlineNet.state_dict(), file_path)

        elif algo_name == "D3QN_PER":
            agent = D3QN_PER(
                **base_kwargs,
                updateFrequency=config.get("updateFrequency", 1),
                tau=config.get("tau", 0.05),
                alpha=config.get("alpha", 0.6),
                beta=config.get("beta", 0.4),
                beta_rate=config.get("beta_rate", 0.001),
            )
            final_eval = agent.runD3QN_PER()
            torch.save(agent.onlineNet.state_dict(), file_path)

        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        return {
            "Algorithm": algo_name,
            "Version": version,
            "Status": "Success",
            "Final Eval": final_eval,
        }

    except Exception as e:
        return {
            "Algorithm": algo_name,
            "Version": version,
            "Status": f"Failed: {str(e)}",
            "Final Eval": "N/A",
        }


def main():
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Batch train Deep RL policies concurrently.")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of parallel processes"
    )
    args = parser.parse_args()

    base_config = {
        "scaling_factor": 5,
        "arena_size": 500,
        "max_steps": 1000,
        "difficulty": 0,
        "box_speed": 2,
        "seed": 200,
        "epsilon_init": 1.0,
        "min_epsilon": 0.05,
        "max_eval_eps": 1,
        "wall_obstacles": True,
        "gamma": 0.99,
    }

    algorithms = ["NFQ", "DQN", "DDQN", "D3QN", "D3QN_PER"]

    global_keys = ["optimizerLR", "max_train_eps", "bufferSize", "epsilon_decay"]
    global_vals = [
        [1e-3, 5e-4],
        [500, 1000],
        [50000, 100000],
        [0.99],
    ]

    tasks = []
    version_counters = {algo: 1 for algo in algorithms}

    logging.info("Generating task combinations...")

    for algo in algorithms:
        algo_keys = []
        algo_vals = []

        if algo == "NFQ":
            algo_keys = ["epochs", "batchSize"]
            algo_vals = [[5, 10], [256, 512, 2048]]

        elif algo in ["DQN", "DDQN"]:
            algo_keys = ["updateFrequency", "batchSize"]
            algo_vals = [[10, 50], [256, 512]]

        elif algo == "D3QN":
            algo_keys = ["updateFrequency", "tau", "batchSize"]
            algo_vals = [[1, 5], [0.05, 0.01], [256, 512]]

        elif algo == "D3QN_PER":
            algo_keys = ["updateFrequency", "tau", "alpha", "beta", "beta_rate", "batchSize"]
            algo_vals = [[1], [0.05], [0.6], [0.4], [0.001], [256, 512]]

        all_keys = global_keys + algo_keys
        all_vals = global_vals + algo_vals

        for combo in itertools.product(*all_vals):
            config = copy.deepcopy(base_config)
            config.update(dict(zip(all_keys, combo)))

            tasks.append({"algorithm": algo, "config": config, "version": version_counters[algo]})
            version_counters[algo] += 1

    logging.info(f"Total combinations to train: {len(tasks)}")

    metadata_rows = []
    for t in tasks:
        c = t["config"]
        a = t["algorithm"]
        v = t["version"]

        expected_path = os.path.join("weightsPth", str(c["max_train_eps"]), f"{a}_v{v}.pth")

        metadata_rows.append(
            {
                "Algorithm": a,
                "Version": v,
                "Max Train Eps": c["max_train_eps"],
                "Wall Obstacles": c["wall_obstacles"],
                "Optimizer LR": c["optimizerLR"],
                "Buffer Size": c["bufferSize"],
                "Batch Size": c["batchSize"],
                "Epsilon Decay": c.get("epsilon_decay", "N/A"),
                "Update Frequency": c.get("updateFrequency", "N/A"),
                "Tau": c.get("tau", "N/A"),
                "Epochs": c.get("epochs", "N/A"),
                "Alpha": c.get("alpha", "N/A"),
                "Beta": c.get("beta", "N/A"),
                "Beta Rate": c.get("beta_rate", "N/A"),
                "File Path": expected_path,
                "Status": "Pending",
                "Final Eval": "N/A",
            }
        )

    df = pd.DataFrame(metadata_rows)
    df.to_csv("train_metadata_discrete.csv", index=False)
    logging.info("Created initial 'train_metadata_discrete.csv' with 'Pending' statuses.")

    logging.info(f"Starting execution with {args.workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_task, task): task for task in tasks}

        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            res = future.result()

            algo_name = res["Algorithm"]
            version = res["Version"]
            status = res["Status"]
            final_eval = res["Final Eval"]

            row_idx = df.index[
                (df["Algorithm"] == algo_name) & (df["Version"] == version)
            ].tolist()[0]
            df.at[row_idx, "Status"] = status
            df.at[row_idx, "Final Eval"] = final_eval
            df.to_csv("train_metadata_ddiscrete.csv", index=False)

            if "Failed" in status:
                logging.error(f"[{count}/{len(tasks)}] FAILED: {algo_name}_v{version} -> {status}")
            else:
                logging.info(
                    f"[{count}/{len(tasks)}] Finished: {algo_name}_v{version} -> {status} (Eval: {final_eval})"
                )

    logging.info("All tasks completed! Final metadata saved to 'train_metadata_discrete.csv'.")


if __name__ == "__main__":
    main()
