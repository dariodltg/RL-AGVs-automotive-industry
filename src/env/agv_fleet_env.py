"""
Gymnasium environment for AGV fleet management in an automotive stamping/welding plant.

Comparison study: PPO (reinforcement learning) vs A* + greedy (classical baseline).

Production flow:
  ENTRY → STAMPING → BUFFER → WELDING → EXIT

Each arrow is a transport task executed by an AGV. Tasks are generated
stochastically at each stage of the flow.

Action space: MultiDiscrete([n_tasks_max + 1] * n_agvs)
  - Per AGV: 0 = wait, 1..n_tasks_max = assign task at index i-1
  - Action is only applied when the AGV is IDLE

Observation space: flat Box(float32) containing:
  - Per AGV    : (row, col, status, battery, has_task)
  - Per task slot: (pickup_row, pickup_col, delivery_row, delivery_col, stage)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .plant_map import CellType, PlantMap, PRODUCTION_FLOW
from src.navigation import AStarPathfinder, BasePathfinder


# ---------------------------------------------------------------------------
# Status enumerations
# ---------------------------------------------------------------------------

class AGVStatus(IntEnum):
    IDLE               = 0
    MOVING_TO_PICKUP   = 1
    LOADING            = 2
    MOVING_TO_DELIVERY = 3
    UNLOADING          = 4
    CHARGING           = 5


# Statuses in which an AGV actively advances along its path each step
_MOVING_STATUSES = frozenset({AGVStatus.MOVING_TO_PICKUP, AGVStatus.MOVING_TO_DELIVERY})


class TaskStatus(IntEnum):
    PENDING     = 0
    ASSIGNED    = 1
    IN_PROGRESS = 2
    COMPLETED   = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id:           int
    pickup:       Tuple[int, int]
    delivery:     Tuple[int, int]
    stage:        int               # index into PRODUCTION_FLOW (0=entry→stamping, ..., 3=welding→exit)
    status:       TaskStatus = TaskStatus.PENDING
    assigned_agv: Optional[int] = None
    created_at:   int = 0
    completed_at: Optional[int] = None

    @property
    def cycle_time(self) -> Optional[int]:
        if self.completed_at is None:
            return None
        return self.completed_at - self.created_at

    @property
    def stage_name(self) -> str:
        names = ["Entry->Stamping", "Stamping->Buffer", "Buffer->Welding", "Welding->Exit"]
        return names[self.stage] if 0 <= self.stage < len(names) else "Unknown"


@dataclass
class AGV:
    id:       int
    position: Tuple[int, int]
    status:   AGVStatus = AGVStatus.IDLE
    battery:  float = 1.0          # [0.0, 1.0]
    task_id:  Optional[int] = None
    path:     List[Tuple[int, int]] = field(default_factory=list)

    # Physical configuration
    BATTERY_DRAIN_MOVING:  float = 0.002
    BATTERY_DRAIN_IDLE:    float = 0.0005
    BATTERY_CHARGE_RATE:   float = 0.02
    BATTERY_LOW_THRESHOLD: float = 0.2

    def drain(self) -> None:
        if self.status in (AGVStatus.MOVING_TO_PICKUP, AGVStatus.MOVING_TO_DELIVERY):
            self.battery = max(0.0, self.battery - self.BATTERY_DRAIN_MOVING)
        else:
            self.battery = max(0.0, self.battery - self.BATTERY_DRAIN_IDLE)

    def charge(self) -> bool:
        """Charges the battery. Returns True when fully charged."""
        self.battery = min(1.0, self.battery + self.BATTERY_CHARGE_RATE)
        return self.battery >= 1.0


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class AGVFleetEnv(gym.Env):
    """
    AGV fleet environment for an automotive stamping and welding plant.

    Production flow: ENTRY → STAMPING → BUFFER → WELDING → EXIT
    Tasks are generated stochastically at each stage of the flow.

    Parameters
    ----------
    n_agvs : int
        Number of AGVs in the fleet (default 4).
    n_tasks_max : int
        Maximum number of simultaneously active tasks (default 8).
    max_steps : int
        Maximum timesteps per episode (default 500).
    task_arrival_rate : float
        Probability of generating a new task at each timestep (default 0.15).
    pathfinder : Optional[BasePathfinder]
        Pathfinding strategy. Defaults to AStarPathfinder.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    # Features per AGV and per task in the observation vector
    AGV_FEATURES  = 5   # row, col, status, battery, has_task
    TASK_FEATURES = 6   # pickup_row, pickup_col, delivery_row, delivery_col, status, stage

    def __init__(
        self,
        n_agvs: int = 4,
        n_tasks_max: int = 8,
        max_steps: int = 500,
        task_arrival_rate: float = 0.15,
        pathfinder: Optional[BasePathfinder] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.n_agvs            = n_agvs
        self.n_tasks_max       = n_tasks_max
        self.max_steps         = max_steps
        self.task_arrival_rate = task_arrival_rate

        self.plant      = PlantMap()
        self.pathfinder = pathfinder if pathfinder is not None else AStarPathfinder()
        self._rng       = random.Random(seed)
        self._np_rng    = np.random.default_rng(seed)

        # Action space: per AGV, 0=wait or 1..n_tasks_max=assign task
        self.action_space = spaces.MultiDiscrete(
            [n_tasks_max + 1] * n_agvs,
            dtype=np.int64,
        )

        # Observation space: normalized flat vector [0, 1]
        obs_size = n_agvs * self.AGV_FEATURES + n_tasks_max * self.TASK_FEATURES
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # Internal state (initialized in reset)
        self.agvs:             List[AGV]  = []
        self.tasks:            List[Task] = []   # active task window
        self._all_tasks:       List[Task] = []   # full episode history
        self._step_count:      int = 0
        self._task_id_counter: int = 0

        self.metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng    = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)

        self._step_count      = 0
        self._task_id_counter = 0
        self.tasks            = []
        self._all_tasks       = []

        spawn_positions = self.plant.get_agv_spawn_positions(self.n_agvs)
        self.agvs = [
            AGV(id=i, position=spawn_positions[i])
            for i in range(self.n_agvs)
        ]

        # Seed with tasks spread across all production stages
        for stage in range(len(PRODUCTION_FLOW)):
            self._spawn_task(stage=stage)

        self.metrics = self._init_metrics()
        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        self._step_count += 1

        reward += self._apply_action(action)
        reward += self._move_agvs()
        self._update_batteries()
        reward += self._check_collisions()

        if self._rng.random() < self.task_arrival_rate:
            # Random stage weighted toward early stages (more raw material tasks)
            stage = self._rng.choices(
                range(len(PRODUCTION_FLOW)),
                weights=[4, 3, 2, 1],
            )[0]
            self._spawn_task(stage=stage)

        reward -= 0.01  # per-step penalty
        self._update_metrics()

        obs       = self._get_observation()
        truncated = self._step_count >= self.max_steps
        info      = self._get_info()

        return obs, reward, False, truncated, info

    def render(self, mode: str = "ansi") -> Optional[str]:
        symbols = {
            CellType.FREE:     ".",
            CellType.OBSTACLE: "#",
            CellType.ENTRY:    "I",
            CellType.STAMPING: "S",
            CellType.BUFFER:   "B",
            CellType.WELDING:  "W",
            CellType.EXIT:     "O",
            CellType.CHARGING: "C",
        }
        grid_display = np.full(
            (self.plant.GRID_SIZE, self.plant.GRID_SIZE), ".", dtype="<U2"
        )
        for r in range(self.plant.GRID_SIZE):
            for c in range(self.plant.GRID_SIZE):
                ct = CellType(self.plant.grid[r, c])
                grid_display[r, c] = symbols[ct]

        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                grid_display[task.pickup[0], task.pickup[1]] = "T"

        for agv in self.agvs:
            grid_display[agv.position[0], agv.position[1]] = str(agv.id)

        lines  = [" ".join(row) for row in grid_display]
        output = "\n".join(lines)
        output += f"\nStep: {self._step_count} | Tasks completed: {self.metrics['tasks_completed']}"

        if mode == "human":
            print(output)
            return None
        return output

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal logic
    # ------------------------------------------------------------------

    def _apply_action(self, action: np.ndarray) -> float:
        penalty = 0.0
        for agv_idx, task_choice in enumerate(action):
            agv = self.agvs[agv_idx]
            if task_choice == 0 or agv.status != AGVStatus.IDLE:
                continue

            task_idx = int(task_choice) - 1
            if task_idx >= len(self.tasks):
                penalty -= 0.5
                continue

            task = self.tasks[task_idx]
            if task.status != TaskStatus.PENDING:
                penalty -= 0.5
                continue

            task.status       = TaskStatus.ASSIGNED
            task.assigned_agv = agv.id
            agv.task_id       = task.id
            agv.status        = AGVStatus.MOVING_TO_PICKUP
            agv.path          = self._compute_path(agv.position, task.pickup)

        return penalty

    def _claim_stationary_cells(self) -> set:
        """Return the set of cells occupied by non-moving AGVs this tick."""
        return {
            agv.position
            for agv in self.agvs
            if agv.status not in _MOVING_STATUSES or not agv.path
        }

    def _resolve_movement_permissions(self) -> set:
        """
        Pre-pass: decide which AGVs may advance one cell this tick.

        Priority = ascending AGV id (AGV 0 moves first). An AGV is permitted
        only if its next cell is unclaimed. Permitted AGVs claim their
        destination; blocked AGVs re-claim their current cell so others avoid
        it. Swaps (A→B while B→A) are both permitted because each AGV frees
        its own cell before the other checks it.
        """
        claimed = self._claim_stationary_cells()
        permitted: set = set()

        for agv in sorted(self.agvs, key=lambda a: a.id):
            if agv.status not in _MOVING_STATUSES or not agv.path:
                continue
            next_cell = agv.path[0]
            if next_cell not in claimed:
                permitted.add(agv.id)
                claimed.add(next_cell)
            else:
                claimed.add(agv.position)

        return permitted

    def _move_agvs(self) -> float:
        reward = 0.0
        permitted = self._resolve_movement_permissions()

        for agv in self.agvs:
            if agv.status == AGVStatus.CHARGING:
                if agv.charge():
                    agv.status = AGVStatus.IDLE
                continue

            if agv.battery <= 0:
                agv.status = AGVStatus.CHARGING
                agv.path   = self._compute_path(agv.position, self._nearest_charging(agv.position))
                continue

            if agv.status == AGVStatus.IDLE:
                if agv.battery < agv.BATTERY_LOW_THRESHOLD:
                    agv.status = AGVStatus.CHARGING
                    agv.path   = self._compute_path(agv.position, self._nearest_charging(agv.position))
                continue

            if agv.path and agv.id in permitted:
                agv.position = agv.path.pop(0)
            # else: blocked this tick — path stays intact, AGV holds its cell

            if not agv.path:
                reward += self._handle_arrival(agv)

        return reward

    def _handle_arrival(self, agv: AGV) -> float:
        task = self._get_task_by_id(agv.task_id)
        if task is None:
            agv.status  = AGVStatus.IDLE
            agv.task_id = None
            return 0.0

        if agv.status == AGVStatus.MOVING_TO_PICKUP:
            task.status = TaskStatus.IN_PROGRESS
            agv.status  = AGVStatus.MOVING_TO_DELIVERY
            agv.path    = self._compute_path(agv.position, task.delivery)
            return 0.0

        if agv.status == AGVStatus.MOVING_TO_DELIVERY:
            task.status       = TaskStatus.COMPLETED
            task.completed_at = self._step_count
            agv.status        = AGVStatus.IDLE
            agv.task_id       = None
            self.metrics["tasks_completed"]           += 1
            self.metrics["tasks_by_stage"][task.stage] += 1
            self.metrics["total_cycle_time"]           += task.cycle_time or 0
            self.tasks = [t for t in self.tasks if t.id != task.id]

            # Reward scaled by stage: later stages are more valuable
            return 10.0 + task.stage * 2.0

        return 0.0

    def _update_batteries(self) -> None:
        for agv in self.agvs:
            agv.drain()

    def _check_collisions(self) -> float:
        positions: Dict[Tuple[int, int], int] = {}
        penalty = 0.0
        for agv in self.agvs:
            if agv.position in positions:
                penalty -= 5.0
                self.metrics["collisions"] += 1
            positions[agv.position] = positions.get(agv.position, 0) + 1
        return penalty

    def _spawn_task(self, stage: Optional[int] = None) -> Optional[Task]:
        """
        Generates a new task at the given production stage.
        If stage is None, picks a random stage weighted toward earlier stages.
        """
        if len(self.tasks) >= self.n_tasks_max:
            return None

        if stage is None:
            stage = self._rng.choices(
                range(len(PRODUCTION_FLOW)),
                weights=[4, 3, 2, 1],
            )[0]

        pickup_type, delivery_type = PRODUCTION_FLOW[stage]
        pickup_cells   = self.plant.cells_by_type[pickup_type]
        delivery_cells = self.plant.cells_by_type[delivery_type]

        if not pickup_cells or not delivery_cells:
            return None

        pickup   = self._rng.choice(pickup_cells)
        delivery = self._rng.choice(delivery_cells)

        task = Task(
            id=self._task_id_counter,
            pickup=pickup,
            delivery=delivery,
            stage=stage,
            created_at=self._step_count,
        )
        self._task_id_counter += 1
        self.tasks.append(task)
        self._all_tasks.append(task)
        return task

    # ------------------------------------------------------------------
    # Pathfinding
    # ------------------------------------------------------------------

    def _compute_path(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Delegates to the injected BasePathfinder (Strategy pattern)."""
        return self.pathfinder.compute_path(self.plant, start, goal)

    def _nearest_charging(self, position: Tuple[int, int]) -> Tuple[int, int]:
        return min(
            self.plant.charging_stations,
            key=lambda cs: PlantMap.manhattan_distance(position, cs),
        )

    def _get_task_by_id(self, task_id: Optional[int]) -> Optional[Task]:
        if task_id is None:
            return None
        for t in self._all_tasks:
            if t.id == task_id:
                return t
        return None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        G = self.plant.GRID_SIZE
        n_stages = len(PRODUCTION_FLOW)

        agv_obs = []
        for agv in self.agvs:
            agv_obs.extend([
                agv.position[0] / (G - 1),
                agv.position[1] / (G - 1),
                float(agv.status)  / len(AGVStatus),
                agv.battery,
                1.0 if agv.task_id is not None else 0.0,
            ])

        task_obs = []
        for i in range(self.n_tasks_max):
            if i < len(self.tasks):
                t = self.tasks[i]
                task_obs.extend([
                    t.pickup[0]   / (G - 1),
                    t.pickup[1]   / (G - 1),
                    t.delivery[0] / (G - 1),
                    t.delivery[1] / (G - 1),
                    float(t.status) / len(TaskStatus),
                    float(t.stage)  / n_stages,
                ])
            else:
                task_obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(agv_obs + task_obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Metrics and info
    # ------------------------------------------------------------------

    def _init_metrics(self) -> Dict[str, Any]:
        return {
            "tasks_completed":  0,
            "tasks_by_stage":   [0] * len(PRODUCTION_FLOW),
            "collisions":       0,
            "total_cycle_time": 0,
            "agv_idle_steps":   [0] * self.n_agvs,
            "total_steps":      0,
        }

    def _update_metrics(self) -> None:
        self.metrics["total_steps"] = self._step_count
        for i, agv in enumerate(self.agvs):
            if agv.status == AGVStatus.IDLE:
                self.metrics["agv_idle_steps"][i] += 1

    def _get_info(self) -> Dict[str, Any]:
        completed = self.metrics["tasks_completed"]
        avg_cycle = (
            self.metrics["total_cycle_time"] / completed
            if completed > 0 else 0.0
        )
        utilization = [
            1.0 - (self.metrics["agv_idle_steps"][i] / max(self._step_count, 1))
            for i in range(self.n_agvs)
        ]
        return {
            "step":             self._step_count,
            "tasks_completed":  completed,
            "tasks_by_stage":   self.metrics["tasks_by_stage"],
            "tasks_pending":    sum(1 for t in self.tasks if t.status == TaskStatus.PENDING),
            "collisions":       self.metrics["collisions"],
            "avg_cycle_time":   avg_cycle,
            "agv_utilization":  utilization,
            "mean_utilization": float(np.mean(utilization)),
        }
