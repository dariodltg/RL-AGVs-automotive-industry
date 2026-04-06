"""
Classical baseline agent: greedy nearest-task assignment + A* navigation.

Assignment policy (greedy):
  For each idle AGV, assign the pending task whose pickup cell is closest
  to the AGV's current position (Manhattan distance).
  Ties are broken by task creation order (lower id wins).

Navigation:
  Low-level movement is handled by AGVFleetEnv via its injected
  AStarPathfinder — the agent only decides *which* task each AGV takes,
  not *how* to get there.

This agent serves as the deterministic baseline against which PPO is
evaluated. It requires no training and has full observability of the
environment state, which is the standard assumption for classical
fleet management algorithms.
"""

from typing import TYPE_CHECKING, List

import numpy as np

from src.env.agv_fleet_env import AGVStatus, TaskStatus
from src.env.plant_map import PlantMap
from .base_agent import BaseAgent

if TYPE_CHECKING:
    from src.env.agv_fleet_env import AGVFleetEnv, Task


class AStarAgent(BaseAgent):
    """
    Greedy nearest-task agent using Manhattan distance for assignment decisions.

    At each timestep:
      1. Collect all PENDING tasks.
      2. For each IDLE AGV (in order of AGV id), assign the closest pending task.
      3. Remove assigned tasks from the candidate pool to avoid double-assignment.
      4. Return the resulting action array.

    AGVs that are not IDLE or have no available task receive action 0 (wait).
    """

    def select_action(self, env: "AGVFleetEnv") -> np.ndarray:
        action = np.zeros(env.n_agvs, dtype=np.int64)

        # Build pool of assignable tasks (PENDING only)
        pending: List["Task"] = [t for t in env.tasks if t.status == TaskStatus.PENDING]

        for agv_idx, agv in enumerate(env.agvs):
            if agv.status != AGVStatus.IDLE or not pending:
                continue

            # Greedy: pick the task with the nearest pickup cell
            nearest_task = min(
                pending,
                key=lambda t: (
                    PlantMap.manhattan_distance(agv.position, t.pickup),
                    t.id,  # tie-break by creation order
                ),
            )

            # Map task to its index in env.tasks (action is 1-indexed)
            task_env_idx = env.tasks.index(nearest_task)
            action[agv_idx] = task_env_idx + 1

            # Remove from pool so no other AGV is assigned the same task
            pending.remove(nearest_task)

        return action

    def reset(self) -> None:
        pass  # stateless agent, nothing to reset
