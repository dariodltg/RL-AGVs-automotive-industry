"""
PPO training script for the AGV fleet environment.

Usage:
    python train_ppo.py                             # default settings
    python train_ppo.py --timesteps 500000
    python train_ppo.py --n_agvs 5 --load_level medium
    python train_ppo.py --run_name exp01 --timesteps 1000000

Load levels map to task_arrival_rate:
    low    = 0.08
    medium = 0.15  (default)
    high   = 0.25

Output:
    models/<run_name>.zip          final model
    models/<run_name>.json         hyperparameter sidecar (read by evaluate.py)
    models/<run_name>_best/        best model by eval reward
    models/checkpoints/            periodic checkpoints
    logs/tensorboard/<run_name>    TensorBoard logs
"""

import argparse
import json
import os
import time
from pathlib import Path

from src.env import AGVFleetEnv
from src.agents import PPOAgent

LOAD_LEVELS = {
    "low":    0.08,
    "medium": 0.15,
    "high":   0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent for AGV fleet")
    parser.add_argument("--timesteps",  type=int,   default=300_000)
    parser.add_argument("--n_agvs",     type=int,   default=4)
    parser.add_argument("--n_tasks",    type=int,   default=8)
    parser.add_argument("--max_steps",  type=int,   default=500)
    parser.add_argument("--load_level", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--run_name",   type=str,   default="ppo_fleet")
    parser.add_argument("--eval_freq",  type=int,   default=10_000)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="PPO learning rate")
    parser.add_argument("--n_steps",    type=int,   default=2048,
                        help="PPO steps per update")
    return parser.parse_args()


def make_env(args, seed: int) -> AGVFleetEnv:
    return AGVFleetEnv(
        n_agvs=args.n_agvs,
        n_tasks_max=args.n_tasks,
        max_steps=args.max_steps,
        task_arrival_rate=LOAD_LEVELS[args.load_level],
        seed=seed,
    )


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  PPO Training — AGV Fleet")
    print("=" * 60)
    print(f"  Run name      : {args.run_name}")
    print(f"  Timesteps     : {args.timesteps:,}")
    print(f"  AGVs          : {args.n_agvs}")
    print(f"  Load level    : {args.load_level} (rate={LOAD_LEVELS[args.load_level]})")
    print(f"  Max steps/ep  : {args.max_steps}")
    print(f"  Learning rate : {args.lr}")
    print(f"  Seed          : {args.seed}")
    print("=" * 60)

    # Training and evaluation environments (different seeds)
    train_env = make_env(args, seed=args.seed)
    eval_env  = make_env(args, seed=args.seed + 100)

    agent = PPOAgent(
        train_env,
        tensorboard_log="logs/tensorboard",
        learning_rate=args.lr,
        n_steps=args.n_steps,
    )

    start = time.time()
    agent.train(
        total_timesteps=args.timesteps,
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        save_path="models",
        run_name=args.run_name,
        progress_bar=True,
    )
    elapsed = time.time() - start

    # Save hyperparameter sidecar so evaluate.py can record them in the registry.
    sidecar = {
        "hyperparameters": {
            "learning_rate": args.lr,
            "n_steps":       args.n_steps,
            "batch_size":    64,
            "n_epochs":      10,
            "gamma":         0.99,
            "ent_coef":      0.01,
            "clip_range":    0.2,
            "vf_coef":       0.5,
            "max_grad_norm": 0.5,
        },
        "env_config": {
            "n_agvs":            args.n_agvs,
            "n_tasks_max":       args.n_tasks,
            "max_steps":         args.max_steps,
            "load_level":        args.load_level,
            "task_arrival_rate": LOAD_LEVELS[args.load_level],
        },
        "timesteps": args.timesteps,
        "seed":      args.seed,
    }
    sidecar_path = Path("models") / f"{args.run_name}.json"
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"Hyperparameter sidecar saved to {sidecar_path}")

    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"To visualize: python run_visual.py --agent ppo --model models/{args.run_name}")
    print("TensorBoard : tensorboard --logdir logs/tensorboard")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
