"""
ZeroMQ bridge: Python simulation state → CoppeliaSim visualization.

Connects to a running CoppeliaSim instance that has the plant scene loaded
(built with scene_builder.py) and updates AGV positions + status colors
every simulation step.

Usage:
    bridge = CoppeliaSimBridge(n_agvs=4)
    bridge.start()
    # ... inside your step loop:
    bridge.sync(env.agvs)
    bridge.close()
"""

import sys
from typing import List

from src.env.agv_fleet_env import AGV, AGVStatus

# ── Must match scene_builder.py constants ──────────────────────────────────
_CELL_SIZE:    float = 0.5
_AGV_Z_OFFSET: float = 0.157  # same as scene_builder._AGV_Z_OFFSET
_COLOR_AMBIENT: int  = 0

# AGVStatus → RGB color shown in CoppeliaSim
_STATUS_COLORS: dict = {
    AGVStatus.IDLE:               [0.95, 0.95, 0.95],  # white
    AGVStatus.MOVING_TO_PICKUP:   [0.95, 0.85, 0.10],  # yellow
    AGVStatus.LOADING:            [0.20, 0.80, 0.20],  # green
    AGVStatus.MOVING_TO_DELIVERY: [0.95, 0.55, 0.05],  # orange
    AGVStatus.UNLOADING:          [0.20, 0.40, 0.85],  # blue
    AGVStatus.CHARGING:           [0.10, 0.85, 0.85],  # cyan
}


def _grid_to_world(row: int, col: int) -> list:
    """Convert grid (row, col) to CoppeliaSim world [x, y, z]."""
    return [
        (col + 0.5) * _CELL_SIZE,
        (row + 0.5) * _CELL_SIZE,
        _AGV_Z_OFFSET,              # keep AGV at the same Z it was placed at
    ]


class CoppeliaSimBridge:
    """
    One-way bridge: pushes AGV state from Python to CoppeliaSim each step.

    The scene must already contain AGV objects named AGV_0 … AGV_{n-1}
    under a parent dummy named 'AGVs' (created by scene_builder.py).

    Parameters
    ----------
    n_agvs : int
        Number of AGVs to control (must match the loaded scene).
    host : str
        CoppeliaSim hostname (default 'localhost').
    port : int
        ZMQ Remote API port (default 23000).
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
        print(f"[bridge] Got handles for {n_agvs} AGVs.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the CoppeliaSim simulation.

        NOTE: when using loaded robot models (e.g. KUKA Omnirob) their internal
        Lua scripts may fight setObjectPosition while the simulation runs.
        For pure kinematic visualization leave the simulation stopped — the
        viewport renders correctly regardless.
        """
        sim = self._sim
        if sim.getSimulationState() == sim.simulation_stopped:
            sim.startSimulation()
            print("[bridge] Simulation started.")

    def stop(self) -> None:
        """Stop the CoppeliaSim simulation."""
        sim = self._sim
        if sim.getSimulationState() != sim.simulation_stopped:
            sim.stopSimulation()
            print("[bridge] Simulation stopped.")

    def close(self) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # State sync
    # ------------------------------------------------------------------

    def sync(self, agvs: List[AGV]) -> None:
        """
        Push current AGV positions and statuses to CoppeliaSim.

        Moves each AGV's root dummy (the whole model follows) and updates
        the status-light color to reflect the AGV's current state.
        Call once per environment step, after env.step() returns.
        """
        sim = self._sim
        for agv in agvs:
            if agv.id >= self._n_agvs:
                continue
            sim.setObjectPosition(self._bases[agv.id], -1, _grid_to_world(*agv.position))
            color = _STATUS_COLORS.get(agv.status, [0.95, 0.95, 0.95])
            sim.setShapeColor(self._lights[agv.id], '', _COLOR_AMBIENT, color)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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
                    f"Object '{base_path}' not found in CoppeliaSim scene.\n"
                    "Run scene_builder.py first and save the scene."
                ) from exc
            try:
                lights.append(sim.getObject(light_path))
            except Exception as exc:
                raise RuntimeError(
                    f"Object '{light_path}' not found. Re-run scene_builder.py."
                ) from exc
        return bases, lights
