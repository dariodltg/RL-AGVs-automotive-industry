"""
ZeroMQ bridge: Python simulation state → CoppeliaSim visualization.

Each env step, sync() updates AGV positions, body orientations, and status
light colors. With interp_steps > 1 the movement is smoothly interpolated.

Usage:
    bridge = CoppeliaSimBridge(n_agvs=4)
    bridge.sync(env.agvs, interp_steps=8, step_delay=0.01)
    bridge.close()
"""

import math
import sys
import time
from typing import List, Optional

from src.env.agv_fleet_env import AGV, AGVStatus

# ── Must match scene_builder.py constants ──────────────────────────────────
_CELL_SIZE:    float = 0.5
_COLOR_AMBIENT: int  = 0

# Rotation offset (radians) added to the computed heading.
# Adjust if the OmniPlatform's visual "front" doesn't match world +X.
_HEADING_OFFSET: float = 0.0

# AGVStatus → RGB color shown on the status-light sphere
_STATUS_COLORS: dict = {
    AGVStatus.IDLE:               [0.95, 0.95, 0.95],  # white
    AGVStatus.MOVING_TO_PICKUP:   [0.95, 0.85, 0.10],  # yellow
    AGVStatus.LOADING:            [0.20, 0.80, 0.20],  # green
    AGVStatus.MOVING_TO_DELIVERY: [0.95, 0.55, 0.05],  # orange
    AGVStatus.UNLOADING:          [0.20, 0.40, 0.85],  # blue
    AGVStatus.CHARGING:           [0.10, 0.85, 0.85],  # cyan
}


def _grid_to_world(row: int, col: int) -> List[float]:
    return [
        (col + 0.5) * _CELL_SIZE,
        (row + 0.5) * _CELL_SIZE,
        0.0,   # dummy root sits at Z=0; model hangs at _AGV_Z_OFFSET below
    ]


def _lerp(a: List[float], b: List[float], t: float) -> List[float]:
    return [a[i] + t * (b[i] - a[i]) for i in range(3)]


class CoppeliaSimBridge:
    """
    One-way bridge: pushes AGV state from Python to CoppeliaSim each step.

    Scene hierarchy expected (created by scene_builder.py):
        /AGVs/AGV_i            ← dummy root  (bridge moves + rotates this)
        /AGVs/AGV_i/AGV_i_Light ← status sphere (bridge colors this)
    """

    def __init__(self, n_agvs: int, host: str = 'localhost', port: int = 23000):
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        except ImportError:
            print("ERROR: pip install coppeliasim-zmqremoteapi-client")
            sys.exit(1)

        print(f"[bridge] Connecting to CoppeliaSim at {host}:{port} ...")
        self._client = RemoteAPIClient(host=host, port=port)
        self._sim    = self._client.require('sim')
        self._n_agvs = n_agvs
        self._bases:  List[int]
        self._lights: List[int]
        self._bases, self._lights = self._resolve_handles(n_agvs)

        # Last world position sent to CoppeliaSim (for interpolation)
        self._visual_pos: List[Optional[List[float]]] = [None] * n_agvs
        # Current heading angle per AGV (radians, world Z axis)
        self._heading: List[float] = [0.0] * n_agvs
        print(f"[bridge] Ready — controlling {n_agvs} AGVs.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the CoppeliaSim simulation.

        NOTE: robot models with internal Lua scripts (OmniPlatform, KUKA…)
        may fight kinematic setObjectPosition while physics runs.
        Leave the simulation stopped for pure visualization.
        """
        sim = self._sim
        if sim.getSimulationState() == sim.simulation_stopped:
            sim.startSimulation()
            print("[bridge] Simulation started.")

    def stop(self) -> None:
        sim = self._sim
        if sim.getSimulationState() != sim.simulation_stopped:
            sim.stopSimulation()
            print("[bridge] Simulation stopped.")

    def close(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # State sync
    # ------------------------------------------------------------------

    def sync(
        self,
        agvs: List[AGV],
        interp_steps: int = 1,
        step_delay: float = 0.0,
    ) -> None:
        """
        Push AGV positions, headings, and status colors to CoppeliaSim.

        Parameters
        ----------
        interp_steps : int
            Sub-frames per env step. 1 = instant jump, 8–15 = smooth slide.
        step_delay : float
            Seconds between sub-frames. Use total_delay / interp_steps.
        """
        targets: dict = {}
        colors:  dict = {}
        for agv in agvs:
            if agv.id >= self._n_agvs:
                continue
            targets[agv.id] = _grid_to_world(*agv.position)
            colors[agv.id]  = _STATUS_COLORS.get(agv.status, [0.95, 0.95, 0.95])

        if interp_steps <= 1:
            self._sync_discrete(targets, colors)
        else:
            self._sync_smooth(targets, colors, interp_steps, step_delay)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_heading(self, agv_id: int, from_pos: List[float], to_pos: List[float]) -> None:
        """Rotate the AGV dummy to face the direction of movement."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return  # not moving — keep current heading
        self._heading[agv_id] = math.atan2(dy, dx) + _HEADING_OFFSET
        # Dummy was created with orientation [0,0,0] → pure Z rotation works directly
        self._sim.setObjectOrientation(
            self._bases[agv_id], -1,
            [0.0, 0.0, self._heading[agv_id]],
        )

    def _sync_discrete(self, targets: dict, colors: dict) -> None:
        sim = self._sim
        for agv_id, target in targets.items():
            prev = self._visual_pos[agv_id]
            if prev is not None:
                self._update_heading(agv_id, prev, target)
            sim.setObjectPosition(self._bases[agv_id], -1, target)
            sim.setShapeColor(self._lights[agv_id], '', _COLOR_AMBIENT, colors[agv_id])
            self._visual_pos[agv_id] = target

    def _sync_smooth(
        self, targets: dict, colors: dict, interp_steps: int, step_delay: float
    ) -> None:
        sim = self._sim
        starts = {
            agv_id: (self._visual_pos[agv_id] or target)
            for agv_id, target in targets.items()
        }
        # Rotate to face destination before starting to slide
        for agv_id, target in targets.items():
            self._update_heading(agv_id, starts[agv_id], target)

        for step in range(1, interp_steps + 1):
            t = step / interp_steps
            for agv_id, target in targets.items():
                sim.setObjectPosition(
                    self._bases[agv_id], -1,
                    _lerp(starts[agv_id], target, t),
                )
            if step_delay > 0 and step < interp_steps:
                time.sleep(step_delay)

        for agv_id, target in targets.items():
            sim.setShapeColor(self._lights[agv_id], '', _COLOR_AMBIENT, colors[agv_id])
            self._visual_pos[agv_id] = target

    def _resolve_handles(self, n: int) -> tuple:
        sim = self._sim
        bases, lights = [], []
        for i in range(n):
            base_path  = f'/AGVs/AGV_{i}'
            light_path = f'/AGVs/AGV_{i}/AGV_{i}_Light'
            try:
                bases.append(sim.getObject(base_path))
            except Exception as exc:
                raise RuntimeError(
                    f"Object '{base_path}' not found. Run scene_builder.py first."
                ) from exc
            try:
                lights.append(sim.getObject(light_path))
            except Exception as exc:
                raise RuntimeError(
                    f"Object '{light_path}' not found. Re-run scene_builder.py."
                ) from exc
        return bases, lights
