"""
Headless evaluation script — runs episodes without Pygame at full speed
and writes per-episode metrics to a CSV via MetricsLogger.

Usage:
    python scripts/evaluate.py --agent astar --episodes 50
    python scripts/evaluate.py --agent ppo --model models/ppo_fleet.zip
    python scripts/evaluate.py --agent random --n_agvs 6 --episodes 100
    python scripts/evaluate.py --agent astar --seeds 0 1 2 3 4

Output:
    runs/{agent}_{n_agvs}agvs_{timestamp}/metrics.csv
"""

import argparse
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import AGVFleetEnv
from src.agents import AStarAgent, PPOAgent
from src.logging import MetricsLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless AGV Fleet evaluation")
    parser.add_argument("--agent",    choices=["astar", "random", "ppo"], default="astar")
    parser.add_argument("--model",    type=str, default=None,
                        help="Path to trained PPO model (required if --agent ppo)")
    parser.add_argument("--n_agvs",   type=int, default=4)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps",    type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--seeds",    type=int, nargs="+", default=[42],
                        help="One or more seeds. Each seed produces an independent run.")
    return parser.parse_args()


def build_agent(args, env: AGVFleetEnv):
    if args.agent == "astar":
        return AStarAgent()
    if args.agent == "ppo":
        if args.model is None:
            print("ERROR: --model <path> is required when using --agent ppo")
            sys.exit(1)
        return PPOAgent.load(args.model, env)
    return None  # random


def _select_action(env, agent):
    return agent.select_action(env) if agent else env.action_space.sample()


def run_seed(args, seed: int) -> None:
    env = AGVFleetEnv(
        n_agvs=args.n_agvs,
        n_tasks_max=8,
        max_steps=args.steps,
        seed=seed,
    )
    agent = build_agent(args, env)

    rewards = []
    tasks   = []

    with MetricsLogger(agent=args.agent, n_agvs=args.n_agvs, seed=seed) as logger:
        print(f"  Logging → {logger.path}")

        for ep in range(1, args.episodes + 1):
            env.reset()
            if agent:
                agent.reset()

            total_reward = 0.0
            done         = False
            info         = {}

            while not done:
                action = _select_action(env, agent)
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            logger.log_episode(ep, total_reward, info)
            rewards.append(total_reward)
            tasks.append(info.get("tasks_completed", 0))

            if ep % 10 == 0 or ep == args.episodes:
                print(
                    f"  [{ep:>4}/{args.episodes}] "
                    f"reward={total_reward:+7.1f}  "
                    f"tasks={info.get('tasks_completed', 0):>3}  "
                    f"util={info.get('mean_utilization', 0):.1%}"
                )

    env.close()

    mean_r  = sum(rewards) / len(rewards)
    std_r   = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
    mean_t  = sum(tasks)   / len(tasks)
    print(f"\n  Reward : {mean_r:+.1f} ± {std_r:.1f}")
    print(f"  Tasks  : {mean_t:.1f} mean per episode")


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  AGV Fleet — Headless Evaluation")
    print("=" * 60)
    print(f"  Agent    : {args.agent}"
          + (f"  ({args.model})" if args.agent == "ppo" else ""))
    print(f"  AGVs     : {args.n_agvs}")
    print(f"  Episodes : {args.episodes}  x  {len(args.seeds)} seed(s)")
    print(f"  Seeds    : {args.seeds}")
    print("=" * 60)

    t0 = time.time()
    for seed in args.seeds:
        print(f"\nSeed {seed}")
        run_seed(args, seed)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
