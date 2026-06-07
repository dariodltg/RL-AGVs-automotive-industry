"""
Experiment 2 — Scaling: PPO vs A*+greedy vs random as fleet size grows.

Pipeline
--------
Phase A (train)  For each n_agvs in N_AGVS_SWEEP, train one PPO model with
                 WINNING_HP (= best combo from experiment 1) at TIMESTEPS
                 steps. Models go to models/grid_scaling/.
Phase B (eval)   For each (agent, n_agvs) pair, evaluate headless over
                 EVAL_SEEDS x EVAL_EPISODES and log every episode to
                 experiments.db. PPO uses the model trained in phase A;
                 A* and random need no training.

Before running
--------------
WINNING_HP below is initialised to PPO defaults. After experiment 1
finishes, query experiments.db (or open Grafana) to pick the best combo
and replace the three TODO values with that combo. Everything else is
already aligned with the experiment 1 setup so results are comparable.

Resumability
------------
Existing models/grid_scaling/ppo_scaling_n<N>.zip files skip the phase A
training for that n_agvs. Delete the .zip to force a retrain.

Usage
-----
    python run_grid_scaling.py
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents import AStarAgent, PPOAgent
from src.env import AGVFleetEnv
from src.experiments import ExperimentRegistry


# ---------------------------------------------------------------------------
# Experiment 2 — Scaling sweep
# ---------------------------------------------------------------------------

N_AGVS_SWEEP: List[int] = [2, 4, 6, 8]
AGENTS:       List[str] = ["ppo", "astar", "random"]

# Winning PPO hyperparameter combo from experiment 1.
WINNING_HP: Dict[str, Any] = {
    "learning_rate": 1e-3,    # experiment 1 winner
    "ent_coef":      0.01,    # experiment 1 winner
    "n_steps":       1024,    # experiment 1 winner
    # Fixed PPO HP (matched to experiment 1)
    "batch_size":    64,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
    "verbose":       0,
}

# Env conditions held fixed across the sweep (load_level = medium).
ENV_FIXED: Dict[str, Any] = {
    "n_tasks_max":       8,
    "max_steps":         500,
    "task_arrival_rate": 0.15,
}
LOAD_LEVEL = "medium"

TIMESTEPS     = 1_000_000
TRAIN_SEED    = 42
EVAL_SEEDS    = [0, 1, 2, 3, 4]
EVAL_EPISODES = 20

OUTPUT_DIR = Path("models/grid_scaling")
TB_BASE    = Path("logs/tensorboard")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tb_available() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


def _write_sidecar(zip_path: Path, n_agvs: int) -> None:
    sidecar = {
        "hyperparameters":  WINNING_HP,
        "env_config":       {**ENV_FIXED, "n_agvs": n_agvs, "load_level": LOAD_LEVEL},
        "timesteps":        TIMESTEPS,
        "train_seed":       TRAIN_SEED,
        "experiment":       "scaling",
    }
    zip_path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8")


def _build_agent(agent_type: str, model_path: Optional[Path], env):
    if agent_type == "ppo":
        return PPOAgent.load(str(model_path.with_suffix("")), env)
    if agent_type == "astar":
        return AStarAgent()
    return None  # random


# ---------------------------------------------------------------------------
# Phase A — Train one PPO per fleet size
# ---------------------------------------------------------------------------

def _train_ppo(n_agvs: int) -> Path:
    name     = f"ppo_scaling_n{n_agvs}"
    zip_path = OUTPUT_DIR / f"{name}.zip"
    if zip_path.exists():
        print(f"  [skip train] {zip_path.name} already exists")
        return zip_path

    train_env = AGVFleetEnv(seed=TRAIN_SEED, n_agvs=n_agvs, **ENV_FIXED)
    tb_log    = str(TB_BASE / name) if _tb_available() else None
    agent     = PPOAgent(train_env, tensorboard_log=tb_log, **WINNING_HP)
    agent.train(
        total_timesteps=TIMESTEPS,
        eval_env=None,
        eval_freq=TIMESTEPS + 1,
        save_path=str(OUTPUT_DIR),
        run_name=name,
        progress_bar=False,
    )
    _write_sidecar(zip_path, n_agvs)
    train_env.close()
    return zip_path


# ---------------------------------------------------------------------------
# Phase B — Eval each agent at each fleet size
# ---------------------------------------------------------------------------

def _eval(
    agent_type: str,
    model_path: Optional[Path],
    n_agvs: int,
    registry: ExperimentRegistry,
) -> None:
    for seed in EVAL_SEEDS:
        env   = AGVFleetEnv(seed=seed, n_agvs=n_agvs, **ENV_FIXED)
        agent = _build_agent(agent_type, model_path, env)

        hp: Dict[str, Any] = {"experiment": "scaling"}
        if agent_type == "ppo":
            hp.update(WINNING_HP)
            hp["model"] = model_path.name if model_path else None

        run_id = registry.start_run(
            agent=agent_type,
            seed=seed,
            n_agvs=n_agvs,
            n_tasks_max=ENV_FIXED["n_tasks_max"],
            max_steps=ENV_FIXED["max_steps"],
            task_arrival_rate=ENV_FIXED["task_arrival_rate"],
            hyperparameters=hp,
        )

        try:
            for ep in range(1, EVAL_EPISODES + 1):
                env.reset()
                done, total_reward = False, 0.0
                info: Dict[str, Any] = {}
                while not done:
                    if agent is None:
                        action = env.action_space.sample()
                    else:
                        action = agent.select_action(env)
                    _, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                registry.log_episode(run_id, ep, total_reward, info)
            registry.finish_run(run_id)
        except Exception:
            traceback.print_exc()
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TB_BASE.mkdir(parents=True, exist_ok=True)

    print(f"Sweep: n_agvs in {N_AGVS_SWEEP}")
    print(f"Agents: {AGENTS}")
    print(f"Train: {TIMESTEPS:,} timesteps, seed={TRAIN_SEED}")
    print(f"Eval:  {len(EVAL_SEEDS)} seeds x {EVAL_EPISODES} episodes per (agent, n_agvs)")
    print(f"WINNING_HP: lr={WINNING_HP['learning_rate']}  "
          f"ent_coef={WINNING_HP['ent_coef']}  n_steps={WINNING_HP['n_steps']}\n")

    # Phase A — Train PPO per fleet size
    models: Dict[int, Path] = {}
    for n in N_AGVS_SWEEP:
        print(f"[train] n_agvs={n}")
        t0 = time.time()
        try:
            models[n] = _train_ppo(n)
            print(f"  done in {time.time() - t0:.0f}s -> {models[n].name}\n")
        except Exception:
            traceback.print_exc()
            print(f"  [error] training n_agvs={n} failed; skipping eval for this size\n")

    # Phase B — Evaluate each agent at each fleet size
    registry = ExperimentRegistry()
    try:
        for n in N_AGVS_SWEEP:
            for agent_type in AGENTS:
                if agent_type == "ppo" and n not in models:
                    print(f"[eval] skip ppo n_agvs={n} (no trained model)")
                    continue
                t0 = time.time()
                model_path = models.get(n) if agent_type == "ppo" else None
                print(f"[eval] agent={agent_type:<6} n_agvs={n}")
                try:
                    _eval(agent_type, model_path, n, registry)
                    print(f"  done in {time.time() - t0:.0f}s")
                except Exception:
                    traceback.print_exc()
                    print(f"  [error] eval failed; continuing")

        print("\nScaling experiment complete.\n")
        registry.summary()
    finally:
        registry.close()


if __name__ == "__main__":
    main()
