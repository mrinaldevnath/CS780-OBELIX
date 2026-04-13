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
from train_ppo import PPO


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("batch_training_ppo.log", mode="a"),
        logging.StreamHandler(),
    ],
)

METADATA_COLUMNS = [
    "Algorithm",
    "Version",
    "Max Train Eps",
    "Optimizer LR",
    "Rollout Steps",
    "PPO Epochs",
    "Clip Coef",
    "Entropy Coeff",
    "Gamma",
    "GAE Lambda",
    "File Path",
    "Status",
    "Final Eval",
]


def build_pending_row(config, version):
    expected_path = os.path.join("weightsPth", str(config["max_train_eps"]), f"PPO_v{version}.pth")
    return {
        "Algorithm": "PPO",
        "Version": version,
        "Max Train Eps": config["max_train_eps"],
        "Optimizer LR": config["optimizerLR"],
        "Rollout Steps": config["rollout_steps"],
        "PPO Epochs": config["ppo_epochs"],
        "Clip Coef": config["clip_coef"],
        "Entropy Coeff": config["entropy_coeff"],
        "Gamma": config["gamma"],
        "GAE Lambda": config["gae_lambda"],
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
    config = task_spec["config"]
    version = task_spec["version"]

    episodes = config["max_train_eps"]
    save_dir = os.path.join("weightsPth", str(episodes))
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"PPO_v{version}.pth")

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
        agent = PPO(
            env=env,
            seed=config["seed"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_coef=config["clip_coef"],
            clip_vloss=config["clip_vloss"],
            entropy_coeff=config["entropy_coeff"],
            vf_coeff=config["vf_coeff"],
            ppo_epochs=config["ppo_epochs"],
            num_minibatches=config["num_minibatches"],
            rollout_steps=config["rollout_steps"],
            optimizerLR=config["optimizerLR"],
            maxGradNorm=config["maxGradNorm"],
            max_train_eps=config["max_train_eps"],
            max_eval_eps=config["max_eval_eps"],
            anneal_lr=config["anneal_lr"],
            target_kl=config["target_kl"],
            norm_adv=config["norm_adv"],
            save_path=file_path,
        )

        best_eval = agent.runPPO()

        return {
            "Algorithm": "PPO",
            "Version": version,
            "Status": "Success",
            "Final Eval": round(float(best_eval), 4),
        }

    except Exception as e:
        return {
            "Algorithm": "PPO",
            "Version": version,
            "Status": f"Failed: {str(e)}",
            "Final Eval": "N/A",
        }


def main():
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Batch train PPO policies concurrently.")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(), help="Number of parallel processes"
    )
    args = parser.parse_args()

    base_config = {
        "scaling_factor": 5,
        "arena_size": 500,
        "max_steps": 1000,
        "wall_obstacles": True,
        "difficulty": 0,
        "box_speed": 2,
        "seed": 200,
        "max_train_eps": 2000,
        "max_eval_eps": 1,
        "clip_coef": 0.2,
        "clip_vloss": True,
        "vf_coeff": 0.5,
        "num_minibatches": 4,
        "gae_lambda": 0.95,
        "maxGradNorm": 0.5,
        "anneal_lr": True,
        "target_kl": None,
        "norm_adv": True,
    }

    sweep_keys = [
        "optimizerLR",
        "rollout_steps",
        "ppo_epochs",
        "entropy_coeff",
        "gamma",
    ]
    sweep_vals = [
        [3e-4, 1e-4],
        [512, 1024, 2048],
        [5, 10],
        [0.01, 0.05],
        [0.99, 0.995],
    ]

    all_tasks = []
    version = 1

    logging.info("Generating task combinations...")

    for combo in itertools.product(*sweep_vals):
        config = copy.deepcopy(base_config)
        params = dict(zip(sweep_keys, combo))
        config.update(params)

        all_tasks.append({"algorithm": "PPO", "config": config, "version": version})
        version += 1

    logging.info(f"Total combinations: {len(all_tasks)}")

    csv_path = "train_metadata_ppo.csv"
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
                logging.error(f"[{count}/{len(pending_tasks)}] FAILED: PPO_v{version} -> {status}")
            else:
                logging.info(
                    f"[{count}/{len(pending_tasks)}] Finished: PPO_v{version} -> {status} (Eval: {final_eval})"
                )

    logging.info(f"All tasks completed! Final metadata saved to '{csv_path}'.")


if __name__ == "__main__":
    main()
