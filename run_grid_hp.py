"""
Experiment 1 — PPO hyperparameter grid search.

Trains a PPO model per combination in GRID with FIXED env/training settings,
then evaluates each model headless across EVAL_SEEDS x EVAL_EPISODES and
logs every episode to experiments.db so combos can be compared in Grafana
(filter by the hyperparameters JSON column).

Layout on disk
--------------
    models/grid_hp/<combo_name>.zip     trained PPO model (final policy)
    models/grid_hp/<combo_name>.json    sidecar with full hyperparameters
    logs/tensorboard/<combo_name>/      TensorBoard scalars for that combo

Resumability
------------
If <combo_name>.zip already exists, the training phase is skipped for that
combo (eval still runs). Delete the .zip to force a retrain.

Usage
-----
    python run_grid_hp.py
"""

from __future__ import annotations

import itertools
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from src.agents import PPOAgent
from src.env import AGVFleetEnv
from src.experiments import ExperimentRegistry


# ---------------------------------------------------------------------------
# Grid (experiment 1)
# ---------------------------------------------------------------------------

GRID: Dict[str, List[Any]] = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "ent_coef":      [0.0, 0.01, 0.05],
    "n_steps":       [1024, 2048],
}

# PPO hyperparameters held constant across the grid.
FIXED_HP: Dict[str, Any] = {
    "batch_size":    64,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
    "verbose":       0,
}

# Environment / training conditions (not hyperparameters of the algorithm).
ENV_CONFIG: Dict[str, Any] = {
    "n_agvs":            4,
    "n_tasks_max":       8,
    "max_steps":         500,
    "task_arrival_rate": 0.15,      # load_level = medium
}
LOAD_LEVEL = "medium"
TIMESTEPS  = 200_000
TRAIN_SEED = 42

EVAL_SEEDS    = [0, 1, 2, 3, 4]
EVAL_EPISODES = 20

OUTPUT_DIR = Path("models/grid_hp")
TB_BASE    = Path("logs/tensorboard")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SHORT_NAME = {"learning_rate": "lr", "ent_coef": "ent", "n_steps": "ns"}


def _combo_name(combo: Dict[str, Any]) -> str:
    """Deterministic, filesystem-safe name for one hyperparameter combo."""
    parts = []
    for key in sorted(combo):
        short = _SHORT_NAME.get(key, key)
        val   = combo[key]
        parts.append(f"{short}{val:g}" if isinstance(val, float) else f"{short}{val}")
    return "ppo_" + "_".join(parts)


def _write_sidecar(zip_path: Path, hp: Dict[str, Any], combo: Dict[str, Any]) -> None:
    sidecar = {
        "hyperparameters":      hp,
        "grid_hyperparameters": combo,
        "env_config":           {**ENV_CONFIG, "load_level": LOAD_LEVEL},
        "timesteps":            TIMESTEPS,
        "train_seed":           TRAIN_SEED,
    }
    zip_path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8")


def _tb_available() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def _train_one(combo: Dict[str, Any], combo_name: str) -> Path:
    """Train one PPO model with the given combo. Returns path to the .zip."""
    zip_path = OUTPUT_DIR / f"{combo_name}.zip"
    if zip_path.exists():
        print(f"  [skip train] {zip_path.name} already exists")
        return zip_path

    hp = {**FIXED_HP, **combo}
    train_env = AGVFleetEnv(seed=TRAIN_SEED,       **ENV_CONFIG)

    tb_log = str(TB_BASE / combo_name) if _tb_available() else None
    agent  = PPOAgent(train_env, tensorboard_log=tb_log, **hp)

    # eval_env=None disables SB3's EvalCallback during training — we run our
    # own deterministic eval phase afterwards. eval_freq is set higher than
    # TIMESTEPS so the unconditional CheckpointCallback inside PPOAgent.train
    # never fires (no per-combo checkpoint clutter on disk).
    agent.train(
        total_timesteps=TIMESTEPS,
        eval_env=None,
        eval_freq=TIMESTEPS + 1,
        save_path=str(OUTPUT_DIR),
        run_name=combo_name,
        progress_bar=False,
    )

    _write_sidecar(zip_path, hp, combo)
    train_env.close()
    return zip_path


def _eval_one(
    model_path: Path,
    combo: Dict[str, Any],
    hp: Dict[str, Any],
    registry: ExperimentRegistry,
) -> None:
    """Evaluate a trained model across EVAL_SEEDS x EVAL_EPISODES."""
    for seed in EVAL_SEEDS:
        env   = AGVFleetEnv(seed=seed, **ENV_CONFIG)
        agent = PPOAgent.load(str(model_path.with_suffix("")), env)

        run_id = registry.start_run(
            agent="ppo",
            seed=seed,
            n_agvs=ENV_CONFIG["n_agvs"],
            n_tasks_max=ENV_CONFIG["n_tasks_max"],
            max_steps=ENV_CONFIG["max_steps"],
            task_arrival_rate=ENV_CONFIG["task_arrival_rate"],
            hyperparameters={
                **hp,
                "grid_combo": combo,
                "model":      model_path.name,
            },
        )

        try:
            for ep in range(1, EVAL_EPISODES + 1):
                env.reset()
                done, total_reward = False, 0.0
                info: Dict[str, Any] = {}
                while not done:
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

    keys, values = zip(*GRID.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Grid:  {len(combos)} combos")
    print(f"Train: {TIMESTEPS:,} timesteps each, seed={TRAIN_SEED}")
    print(f"Eval:  {len(EVAL_SEEDS)} seeds x {EVAL_EPISODES} episodes per combo")
    print(f"Env:   {ENV_CONFIG}  load_level={LOAD_LEVEL}\n")

    registry = ExperimentRegistry()
    try:
        for i, combo in enumerate(combos, 1):
            name = _combo_name(combo)
            print(f"[{i}/{len(combos)}] {name}")
            print(f"  combo: {combo}")
            t0 = time.time()
            try:
                zip_path = _train_one(combo, name)
                print(f"  trained in {time.time() - t0:.0f}s -> {zip_path.name}")
                t1 = time.time()
                _eval_one(zip_path, combo, {**FIXED_HP, **combo}, registry)
                print(f"  evaluated in {time.time() - t1:.0f}s")
            except Exception:
                traceback.print_exc()
                print(f"  [error] combo {name} failed; continuing\n")
                continue
            print(f"  total {time.time() - t0:.0f}s\n")

        print("Grid complete.\n")
        registry.summary()
    finally:
        registry.close()


if __name__ == "__main__":
    main()
