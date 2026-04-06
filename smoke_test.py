"""
Smoke test — verifies the environment initializes and runs without errors.
"""

import sys
import numpy as np

print("Importing environment...")
from src.env import AGVFleetEnv, PlantMap

# ------------------------------------------------------------------
# 1. PlantMap
# ------------------------------------------------------------------
print("\n[1/4] PlantMap")
plant = PlantMap()
assert plant.grid.shape == (20, 20), "Grid shape mismatch"
assert len(plant.manufacturing_cells) > 0, "No manufacturing cells found"
assert len(plant.entry_exit_points) > 0, "No entry/exit points found"
assert len(plant.charging_stations) > 0, "No charging stations found"
print(plant)
print(f"  manufacturing cells : {plant.manufacturing_cells}")
print(f"  entry/exit points   : {plant.entry_exit_points}")
print(f"  charging stations   : {plant.charging_stations}")

# ------------------------------------------------------------------
# 2. Environment instantiation
# ------------------------------------------------------------------
print("\n[2/4] Environment instantiation")
env = AGVFleetEnv(n_agvs=4, n_tasks_max=8, max_steps=50, seed=42)
print(f"  action_space      : {env.action_space}")
print(f"  observation_space : {env.observation_space}")

# ------------------------------------------------------------------
# 3. Reset
# ------------------------------------------------------------------
print("\n[3/4] reset()")
obs, info = env.reset(seed=42)
assert obs.shape == env.observation_space.shape, f"Obs shape mismatch: {obs.shape}"
assert env.observation_space.contains(obs), "Observation out of bounds"
print(f"  obs shape  : {obs.shape}")
print(f"  info       : {info}")

# ------------------------------------------------------------------
# 4. Step loop with random actions
# ------------------------------------------------------------------
print("\n[4/4] step() loop (50 steps)")
total_reward = 0.0
for step in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    assert env.observation_space.contains(obs), f"Observation out of bounds at step {step}"
    if truncated:
        break

print(f"  steps run       : {info['step']}")
print(f"  total reward    : {total_reward:.2f}")
print(f"  tasks completed : {info['tasks_completed']}")
print(f"  collisions      : {info['collisions']}")
print(f"  mean util.      : {info['mean_utilization']:.2%}")

print("\n--- render ---")
env.render(mode="human")

env.close()
print("\nSmoke test PASSED")
