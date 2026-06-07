"""
Experiment 3 — Robustness (zero-shot): PPO vs A*+greedy vs random under
varying task load.

Pipeline
--------
Phase A (train)  Train ONE PPO model with WINNING_HP at TRAIN_LOAD only.
                 Goes to models/grid_robustness/ppo_robustness.zip.
Phase B (eval)   Evaluate three agents (the same PPO model unchanged,
                 A* and random) under each load level in EVAL_LOADS.
                 PPO sees high/low at eval time WITHOUT having been
                 trained on those distributions — measures out-of-
                 distribution generalisation, which is the realistic
                 case for a factory whose load fluctuates over the day.

Before running
--------------
WINNING_HP below is initialised to PPO defaults. After experiment 1
finishes, replace the three TODO values with the best combo so the
robustness comparison uses the tuned policy.

Resumability
------------
If models/grid_robustness/ppo_robustness.zip already exists, phase A is
skipped. Delete the .zip to force a retrain.

Usage
-----
    python run_grid_robustness.py
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
# Experiment 3 — Robustness sweep
# ---------------------------------------------------------------------------

# Maps load_level label -> task_arrival_rate (Poisson rate per step).
LOAD_LEVELS: Dict[str, float] = {"low": 0.08, "medium": 0.15, "high": 0.25}
TRAIN_LOAD:  str           = "medium"
EVAL_LOADS:  List[str]     = ["low", "medium", "high"]
AGENTS:      List[str]     = ["ppo", "astar", "random"]

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

# Env conditions held fixed across the sweep (only load varies at eval).
ENV_FIXED: Dict[str, Any] = {
    "n_agvs":      4,
    "n_tasks_max": 8,
    "max_steps":   500,
}

TIMESTEPS     = 1_000_000
TRAIN_SEED    = 42
EVAL_SEEDS    = [0, 1, 2, 3, 4]
EVAL_EPISODES = 20

OUTPUT_DIR = Path("models/grid_robustness")
TB_BASE    = Path("logs/tensorboard")
MODEL_NAME = "ppo_robustness"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tb_available() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


def _write_sidecar(zip_path: Path) -> None:
    sidecar = {
        "hyperparameters":  WINNING_HP,
        "env_config":       {
            **ENV_FIXED,
            "task_arrival_rate": LOAD_LEVELS[TRAIN_LOAD],
            "load_level":        TRAIN_LOAD,
        },
        "timesteps":        TIMESTEPS,
        "train_seed":       TRAIN_SEED,
        "experiment":       "robustness",
        "eval_loads":       EVAL_LOADS,
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
# Phase A — Train ONE PPO at TRAIN_LOAD
# ---------------------------------------------------------------------------

def _train_ppo() -> Path:
    zip_path = OUTPUT_DIR / f"{MODEL_NAME}.zip"
    if zip_path.exists():
        print(f"  [skip train] {zip_path.name} already exists")
        return zip_path

    train_env = AGVFleetEnv(
        seed=TRAIN_SEED,
        task_arrival_rate=LOAD_LEVELS[TRAIN_LOAD],
        **ENV_FIXED,
    )
    tb_log = str(TB_BASE / MODEL_NAME) if _tb_available() else None
    agent  = PPOAgent(train_env, tensorboard_log=tb_log, **WINNING_HP)
    agent.train(
        total_timesteps=TIMESTEPS,
        eval_env=None,
        eval_freq=TIMESTEPS + 1,
        save_path=str(OUTPUT_DIR),
        run_name=MODEL_NAME,
        progress_bar=False,
    )
    _write_sidecar(zip_path)
    train_env.close()
    return zip_path


# ---------------------------------------------------------------------------
# Phase B — Eval each agent under each load (zero-shot for PPO)
# ---------------------------------------------------------------------------

def _eval(
    agent_type: str,
    model_path: Optional[Path],
    load_level: str,
    registry: ExperimentRegistry,
) -> None:
    rate = LOAD_LEVELS[load_level]

    for seed in EVAL_SEEDS:
        env   = AGVFleetEnv(seed=seed, task_arrival_rate=rate, **ENV_FIXED)
        agent = _build_agent(agent_type, model_path, env)

        hp: Dict[str, Any] = {
            "experiment":       "robustness",
            "eval_load_level":  load_level,
            "train_load_level": TRAIN_LOAD if agent_type == "ppo" else None,
        }
        if agent_type == "ppo":
            hp.update(WINNING_HP)
            hp["model"] = model_path.name if model_path else None

        run_id = registry.start_run(
            agent=agent_type,
            seed=seed,
            n_agvs=ENV_FIXED["n_agvs"],
            n_tasks_max=ENV_FIXED["n_tasks_max"],
            max_steps=ENV_FIXED["max_steps"],
            task_arrival_rate=rate,
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

    print(f"Train load:  {TRAIN_LOAD}  (rate={LOAD_LEVELS[TRAIN_LOAD]})")
    print(f"Eval loads:  {EVAL_LOADS}")
    print(f"Agents:      {AGENTS}")
    print(f"Train:       {TIMESTEPS:,} timesteps, seed={TRAIN_SEED}")
    print(f"Eval:        {len(EVAL_SEEDS)} seeds x {EVAL_EPISODES} episodes per (agent, load)")
    print(f"WINNING_HP:  lr={WINNING_HP['learning_rate']}  "
          f"ent_coef={WINNING_HP['ent_coef']}  n_steps={WINNING_HP['n_steps']}\n")

    # Phase A — train one PPO at TRAIN_LOAD
    print(f"[train] PPO @ load={TRAIN_LOAD}")
    t0 = time.time()
    try:
        model_path = _train_ppo()
        print(f"  done in {time.time() - t0:.0f}s -> {model_path.name}\n")
    except Exception:
        traceback.print_exc()
        print("  [error] training failed; aborting experiment")
        return

    # Phase B — eval each agent at each load
    registry = ExperimentRegistry()
    try:
        for load in EVAL_LOADS:
            for agent_type in AGENTS:
                t0 = time.time()
                mp = model_path if agent_type == "ppo" else None
                print(f"[eval] agent={agent_type:<6} load={load}")
                try:
                    _eval(agent_type, mp, load, registry)
                    print(f"  done in {time.time() - t0:.0f}s")
                except Exception:
                    traceback.print_exc()
                    print("  [error] eval failed; continuing")

        print("\nRobustness experiment complete.\n")
        registry.summary()
    finally:
        registry.close()


if __name__ == "__main__":
    main()
