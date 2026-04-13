import os
import copy
import itertools
import pickle
import argparse
import logging
import pandas as pd
import concurrent.futures

from obelix import OBELIX

from train_qlearning import q_learning
from train_dqlearning import double_q_learning
from train_qlambda import q_learning_lambda
from train_sarsa_lambda import sarsa_lambda
from train_tsampling import trajectory_sampling


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("batch_training_tabular.log", mode="a"), logging.StreamHandler()],
)


def worker_task(task_spec):
    """Executes a single training run using the imported algorithm functions."""
    algo_name = task_spec["algorithm"]
    config = task_spec["config"]
    version = task_spec["version"]

    episodes = config["max_episodes"]
    save_dir = os.path.join("weights", str(episodes))
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{algo_name}_v{version}.pkl")

    env = OBELIX(
        scaling_factor=config["scaling_factor"],
        arena_size=config["arena_size"],
        max_steps=config["max_steps"],
        wall_obstacles=config["wall_obstacles"],
        difficulty=config["difficulty"],
        box_speed=config["box_speed"],
    )

    try:
        if algo_name == "qlearning":
            _, pi = q_learning(env, config)
        elif algo_name == "dqlearning":
            _, _, pi = double_q_learning(env, config)
        elif algo_name == "qlambda":
            _, pi = q_learning_lambda(env, config)
        elif algo_name == "sarsa_lambda":
            _, pi = sarsa_lambda(env, config)
        elif algo_name == "tsampling":
            _, pi = trajectory_sampling(env, config)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        with open(file_path, "wb") as f:
            pickle.dump(pi, f)

        return {
            "Algorithm": algo_name,
            "Version": version,
            "Status": "Success",
        }
    except Exception as e:
        return {
            "Algorithm": algo_name,
            "Version": version,
            "Status": f"Failed: {str(e)}",
        }


def main():
    parser = argparse.ArgumentParser(description="Batch train tabular RL policies concurrently.")
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
        "gamma": 0.99,
        "min_alpha": 0.01,
        "alpha_decay": 0.99,
        "epsilon_init": 1.0,
        "min_epsilon": 0.05,
        "epsilon_decay": 0.99,
        "lambda_param": 0.9,
    }

    algorithms = ["qlearning", "dqlearning", "qlambda", "sarsa_lambda", "tsampling"]
    wall_opts = [True, False]
    alpha_opts = [0.1, 0.5]
    episodes_opts = [500, 1000]
    trace_opts = [True, False]
    max_traj_opts = [10, 20]

    tasks = []
    version_counters = {algo: 1 for algo in algorithms}

    logging.info("Generating task combinations...")

    for algo in algorithms:
        if algo in ["qlearning", "dqlearning"]:
            combinations = itertools.product(wall_opts, alpha_opts, episodes_opts)
            for w, a, e in combinations:
                config = copy.deepcopy(base_config)
                config.update({"wall_obstacles": w, "alpha_init": a, "max_episodes": e})
                tasks.append(
                    {"algorithm": algo, "config": config, "version": version_counters[algo]}
                )
                version_counters[algo] += 1

        elif algo in ["qlambda", "sarsa_lambda"]:
            combinations = itertools.product(wall_opts, alpha_opts, episodes_opts, trace_opts)
            for w, a, e, t in combinations:
                config = copy.deepcopy(base_config)
                config.update(
                    {"wall_obstacles": w, "alpha_init": a, "max_episodes": e, "replace_trace": t}
                )
                tasks.append(
                    {"algorithm": algo, "config": config, "version": version_counters[algo]}
                )
                version_counters[algo] += 1

        elif algo == "tsampling":
            combinations = itertools.product(wall_opts, alpha_opts, episodes_opts, max_traj_opts)
            for w, a, e, m in combinations:
                config = copy.deepcopy(base_config)
                config.update(
                    {"wall_obstacles": w, "alpha_init": a, "max_episodes": e, "max_trajectory": m}
                )
                tasks.append(
                    {"algorithm": algo, "config": config, "version": version_counters[algo]}
                )
                version_counters[algo] += 1

    logging.info(f"Total combinations to train: {len(tasks)}")

    metadata_rows = []
    for t in tasks:
        c = t["config"]
        a = t["algorithm"]
        v = t["version"]

        expected_path = os.path.join("weights", str(c["max_episodes"]), f"{a}_v{v}.pkl")

        metadata_rows.append(
            {
                "Algorithm": a,
                "Version": v,
                "Max Episodes": c["max_episodes"],
                "Wall Obstacles": c["wall_obstacles"],
                "Alpha Init": c["alpha_init"],
                "Replace Trace": c.get("replace_trace", "N/A"),
                "Max Trajectory": c.get("max_trajectory", "N/A"),
                "File Path": expected_path,
                "Status": "Pending",
            }
        )

    df = pd.DataFrame(metadata_rows)
    df.to_csv("train_metadata_tabular.csv", index=False)
    logging.info("Created initial 'train_metadata_tabular.csv' with 'Pending' statuses.")

    logging.info(f"Starting execution with {args.workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_task, task): task for task in tasks}

        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            res = future.result()

            algo_name = res["Algorithm"]
            version = res["Version"]
            status = res["Status"]

            row_idx = df.index[
                (df["Algorithm"] == algo_name) & (df["Version"] == version)
            ].tolist()[0]
            df.at[row_idx, "Status"] = status
            df.to_csv("train_metadata_tabular.csv", index=False)

            if "Failed" in status:
                logging.error(f"[{count}/{len(tasks)}] FAILED: {algo_name}_v{version} -> {status}")
            else:
                logging.info(f"[{count}/{len(tasks)}] Finished: {algo_name}_v{version} -> {status}")

    logging.info("All tasks completed! Final metadata saved to 'train_metadata_tabular.csv'.")


if __name__ == "__main__":
    main()
