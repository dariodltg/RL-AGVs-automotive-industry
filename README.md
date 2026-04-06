# RL-AGVs Automotive Industry

Systematic evaluation of **PPO (Proximal Policy Optimization)** vs **A\* + Greedy** for AGV fleet management in an automotive components manufacturing plant.

Master's thesis — Industria 4.0, UNIR.

---

## Overview

A fleet of Automated Guided Vehicles (AGVs) transports materials between manufacturing cells and entry/exit points inside a simulated plant. Two fleet management strategies are compared:

| Strategy | Type | Description |
|---|---|---|
| **A\* + Greedy** | Classical baseline | Assigns each idle AGV the nearest pending task (Manhattan distance). Navigation via A\*. |
| **PPO** | Reinforcement learning | Learned policy trained with Stable-Baselines3 on the Gymnasium environment. |

---

## Project Structure

```
src/
├── env/
│   ├── plant_map.py        # 20x20 grid map (manufacturing cells, corridors, charging stations)
│   └── agv_fleet_env.py    # Gymnasium environment (action/observation spaces, reward, metrics)
├── navigation/
│   ├── pathfinder.py       # BasePathfinder abstract class (Strategy pattern)
│   └── astar.py            # A* pathfinder with Manhattan heuristic
└── agents/
    ├── base_agent.py       # BaseAgent abstract class (Template Method pattern)
    └── astar_agent.py      # Greedy nearest-task classical baseline
```

---

## Environment

**Observation space:** `Box(0.0, 1.0, shape=(60,), float32)`
Flat normalized vector: 5 features per AGV × 4 AGVs + 5 features per task slot × 8 task slots.

**Action space:** `MultiDiscrete([9, 9, 9, 9])`
Per AGV: `0` = wait, `1–8` = assign task at that index.

**Reward:**
- `+10` per completed task
- `−0.01` per timestep
- `−5` per collision
- `−0.5` per invalid action

**Key metrics** (available in `info` dict after each `step()`):

| Metric | Key |
|---|---|
| Tasks completed | `tasks_completed` |
| Average cycle time | `avg_cycle_time` |
| AGV utilization per vehicle | `agv_utilization` |
| Mean fleet utilization | `mean_utilization` |
| Collisions | `collisions` |

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: `gymnasium>=0.29`, `stable-baselines3>=2.3`, `numpy>=1.26`

---

## Running the smoke test

Verifies that the environment, pathfinder and baseline agent initialize and run correctly:

```bash
python smoke_test.py
```

Expected output summary:
```
[1/5] PlantMap          — grid layout printed, cell counts verified
[2/5] Environment       — action/observation spaces printed
[3/5] reset()           — observation shape and initial info verified
[4/5] Random agent      — 50-step episode with random actions
[5/5] AStarAgent        — 50-step episode with greedy baseline
Smoke test PASSED
```

---

## Using the environment

```python
from src.env import AGVFleetEnv
from src.agents import AStarAgent

env = AGVFleetEnv(n_agvs=4, n_tasks_max=8, max_steps=500, seed=42)
agent = AStarAgent()

obs, info = env.reset()
agent.reset()

for _ in range(500):
    action = agent.select_action(env)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(info)
env.close()
```

---

## Tech stack

- **RL training:** Python · Gymnasium · Stable-Baselines3 (PPO)
- **3D visualization:** CoppeliaSim + ZeroMQ bridge *(planned)*
- **Monitoring:** Grafana *(planned)*
- **Platform:** Windows 10
