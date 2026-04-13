import os
import copy
import itertools
import argparse
import logging

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2

cv2.setNumThreads(0)

import pandas as pd
import torch
import concurrent.futures

from obelix import OBELIX
from train_td3 import TD3


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("batch_train_td3.log", mode="a"),
        logging.StreamHandler(),
    ],
)

METADATA_COLUMNS = [
    "Algorithm",
    "Version",
    "Max Train Eps",
    "Batch Size",
    "Policy LR",
    "Value LR",
    "Tau",
    "Train Policy Freq",
    "Update Freq Policy",
    "Update Freq Value",
    "File Path",
    "Status",
    "Final Eval",
]


def build_pending_row(config, version):
    expected_path = os.path.join("weightsPth", str(config["max_train_eps"]), f"TD3_v{version}.pth")
    return {
        "Algorithm": "TD3",
        "Version": version,
        "Max Train Eps": config["max_train_eps"],
        "Batch Size": config["batchSize"],
        "Policy LR": config["policyOptimizerLR"],
        "Value LR": config["valueOptimizerLR"],
        "Tau": config["tau"],
        "Train Policy Freq": config["trainPolicyFrequency"],
        "Update Freq Policy": config["updateFrequencyPolicy"],
        "Update Freq Value": config["updateFrequencyValue"],
        "File Path": expected_path,
        "Status": "Pending",
        "Final Eval": "N/A",
    }


def init_csv(tasks, csv_path):
    existing_df = None

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            existing_df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            logging.warning("Existing CSV unreadable — recreating.")
            existing_df = None

    if existing_df is None:
        rows = [build_pending_row(t["config"], t["version"]) for t in tasks]
        df = pd.DataFrame(rows, columns=METADATA_COLUMNS)
        df.to_csv(csv_path, index=False)
        logging.info(f"CSV created: {len(rows)} Pending rows -> {csv_path}")
        return tasks, df

    existing_versions = set(existing_df["Version"].tolist())
    new_rows = []
    for t in tasks:
        if t["version"] not in existing_versions:
            new_rows.append(build_pending_row(t["config"], t["version"]))

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=METADATA_COLUMNS)
        df = pd.concat([existing_df, new_df], ignore_index=True)
        df.to_csv(csv_path, index=False)
        logging.info(f"CSV updated: added {len(new_rows)} new rows.")
    else:
        df = existing_df

    success_versions = set(df.loc[df["Status"] == "Success", "Version"].tolist())
    pending = [t for t in tasks if t["version"] not in success_versions]
    logging.info(f"Resuming: {len(success_versions)} done, {len(pending)} remaining.")
    return pending, df


def worker_task(task_spec):
    """Executes a single TD3 training run."""
    config = task_spec["config"]
    version = task_spec["version"]

    episodes = config["max_train_eps"]
    save_dir = os.path.join("weightsPth", str(episodes))
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"TD3_v{version}.pth")

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
        agent = TD3(
            env=env,
            seed=config["seed"],
            gamma=config["gamma"],
            tau=config["tau"],
            bufferSize=config["bufferSize"],
            batchSize=config["batchSize"],
            minSamples=config["minSamples"],
            updateFrequencyPolicy=config["updateFrequencyPolicy"],
            updateFrequencyValue=config["updateFrequencyValue"],
            trainPolicyFrequency=config["trainPolicyFrequency"],
            policyOptimizerLR=config["policyOptimizerLR"],
            valueOptimizerLR=config["valueOptimizerLR"],
            maxGradNorm=config["maxGradNorm"],
            max_train_eps=config["max_train_eps"],
            max_eval_eps=config["max_eval_eps"],
            epsilon_init=config["epsilon_init"],
            min_epsilon=config["min_epsilon"],
            epsilon_decay=config["epsilon_decay"],
            gumbel_temp_init=config["gumbel_temp_init"],
            gumbel_temp_min=config["gumbel_temp_min"],
            gumbel_temp_decay=config["gumbel_temp_decay"],
            save_path=file_path,
        )

        best_eval = agent.runTD3()

        return {
            "Algorithm": "TD3",
            "Version": version,
            "Status": "Success",
            "Final Eval": round(float(best_eval), 4),
        }

    except Exception as e:
        return {
            "Algorithm": "TD3",
            "Version": version,
            "Status": f"Failed: {str(e)}",
            "Final Eval": "N/A",
        }


def main():
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Batch train TD3 policies concurrently.")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of parallel processes"
    )
    args = parser.parse_args()

    base_config = {
        "scaling_factor": 5,
        "arena_size": 500,
        "max_steps": 1000,
        "wall_obstacles": True,
        "difficulty": 2,
        "box_speed": 2,
        "seed": 200,
        "gamma": 0.99,
        "bufferSize": 100000,
        "minSamples": 1000,
        "maxGradNorm": 1.0,
        "max_eval_eps": 1,
        "epsilon_init": 1.0,
        "min_epsilon": 0.05,
        "epsilon_decay": 0.99,
        "gumbel_temp_init": 1.0,
        "gumbel_temp_min": 0.1,
        "gumbel_temp_decay": 0.995,
    }

    sweep_keys = [
        "max_train_eps",
        "batchSize",
        "optimizerLR",
        "tau",
        "trainPolicyFrequency",
    ]
    sweep_vals = [
        [500, 1000],
        [256, 512],
        [1e-3, 5e-4],
        [0.005, 0.01],
        [2, 5],
    ]

    all_tasks = []
    version = 1

    logging.info("Generating task combinations...")

    for combo in itertools.product(*sweep_vals):
        config = copy.deepcopy(base_config)
        params = dict(zip(sweep_keys, combo))

        config.update(
            {
                "max_train_eps": params["max_train_eps"],
                "batchSize": params["batchSize"],
                "policyOptimizerLR": params["optimizerLR"],
                "valueOptimizerLR": params["optimizerLR"],
                "tau": params["tau"],
                "trainPolicyFrequency": params["trainPolicyFrequency"],
                "updateFrequencyPolicy": params["trainPolicyFrequency"],
                "updateFrequencyValue": params["trainPolicyFrequency"],
            }
        )

        all_tasks.append({"algorithm": "TD3", "config": config, "version": version})
        version += 1

    logging.info(f"Total combinations: {len(all_tasks)}")

    csv_path = "train_metadata_td3.csv"
    pending_tasks, df = init_csv(all_tasks, csv_path)

    if not pending_tasks:
        logging.info("All tasks already completed. Exiting.")
        return

    logging.info(f"Running {len(pending_tasks)} tasks with {args.workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_task, task): task for task in pending_tasks}

        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            res = future.result()

            version = res["Version"]
            status = res["Status"]
            final_eval = res["Final Eval"]

            row_idx = df.index[df["Version"] == version].tolist()[0]
            df.at[row_idx, "Status"] = status
            df.at[row_idx, "Final Eval"] = final_eval
            df.to_csv(csv_path, index=False)

            if "Failed" in status:
                logging.error(f"[{count}/{len(pending_tasks)}] FAILED: TD3_v{version} -> {status}")
            else:
                logging.info(
                    f"[{count}/{len(pending_tasks)}] Finished: TD3_v{version} -> {status} (Eval: {final_eval})"
                )

    logging.info(f"All tasks completed! Final metadata saved to '{csv_path}'.")


if __name__ == "__main__":
    main()
