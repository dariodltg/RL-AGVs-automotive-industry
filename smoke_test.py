"""
Smoke test — verifies the environment and agents initialize and run without errors.
"""

import numpy as np

print("Importing modules...")
from src.env import AGVFleetEnv, PlantMap
from src.agents import AStarAgent

# ------------------------------------------------------------------
# 1. PlantMap
# ------------------------------------------------------------------
print("\n[1/5] PlantMap")
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
print("\n[2/5] Environment instantiation")
env = AGVFleetEnv(n_agvs=4, n_tasks_max=8, max_steps=50, seed=42)
print(f"  action_space      : {env.action_space}")
print(f"  observation_space : {env.observation_space}")

# ------------------------------------------------------------------
# 3. Reset
# ------------------------------------------------------------------
print("\n[3/5] reset()")
obs, info = env.reset(seed=42)
assert obs.shape == env.observation_space.shape, f"Obs shape mismatch: {obs.shape}"
assert env.observation_space.contains(obs), "Observation out of bounds"
print(f"  obs shape : {obs.shape}")
print(f"  info      : {info}")

# ------------------------------------------------------------------
# 4. Random agent — lower bound reference
# ------------------------------------------------------------------
print("\n[4/5] Random agent (50 steps)")
env.reset(seed=42)
total_reward = 0.0
for step in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    assert env.observation_space.contains(obs), f"Obs out of bounds at step {step}"
    if truncated:
        break

print(f"  steps run       : {info['step']}")
print(f"  total reward    : {total_reward:.2f}")
print(f"  tasks completed : {info['tasks_completed']}")
print(f"  collisions      : {info['collisions']}")
print(f"  mean util.      : {info['mean_utilization']:.2%}")

# ------------------------------------------------------------------
# 5. AStarAgent (greedy baseline) — deterministic reference
# ------------------------------------------------------------------
print("\n[5/5] AStarAgent greedy baseline (50 steps)")
env.reset(seed=42)
agent = AStarAgent()
agent.reset()
total_reward = 0.0
for step in range(50):
    action = agent.select_action(env)
    assert env.action_space.contains(action), f"Invalid action at step {step}: {action}"
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    assert env.observation_space.contains(obs), f"Obs out of bounds at step {step}"
    if truncated:
        break

print(f"  steps run       : {info['step']}")
print(f"  total reward    : {total_reward:.2f}")
print(f"  tasks completed : {info['tasks_completed']}")
print(f"  collisions      : {info['collisions']}")
print(f"  mean util.      : {info['mean_utilization']:.2%}")

print("\n--- render (AStarAgent final state) ---")
env.render(mode="human")

env.close()
print("\nSmoke test PASSED")
