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
import threading
import time
from typing import Dict, List, Optional, Tuple

from src.env.agv_fleet_env import AGV, AGVStatus, Task, TaskStatus

# ── Must match scene_builder.py constants ──────────────────────────────────
_CELL_SIZE:    float = 0.5
_GRID_SIZE:    int   = 20   # must match PlantMap.GRID_SIZE
_COLOR_AMBIENT: int  = 0

# Task marker spheres (floating above cells)
_PRIM_SPHERE  = 1      # CoppeliaSim sphere type (sim.primitiveshape_sphere not always exposed)
_MARKER_Z     = 0.65   # height above ground (clears even 0.4 m station blocks)
_MARKER_R     = 0.055  # radius of regular pickup/delivery markers
_FLASH_R      = 0.10   # radius of completion flash sphere

# RGB colors for task markers (float [0,1])
_COLOR_PICKUP   = [0.95, 0.25, 0.25]   # red   — pending pickup
_COLOR_DELIVERY = [0.25, 0.90, 0.40]   # green — in-progress delivery
_COLOR_FLASH    = [1.00, 0.95, 0.30]   # yellow — brief flash on pickup/delivery event

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
    # X is mirrored so the CoppeliaSim viewport matches Pygame's left/right
    return [
        (_GRID_SIZE - 1 - col + 0.5) * _CELL_SIZE,
        (row + 0.5)                  * _CELL_SIZE,
        0.0,   # dummy root sits at Z=0; model hangs at _AGV_Z_OFFSET below
    ]


def _lerp(a: List[float], b: List[float], t: float) -> List[float]:
    return [a[i] + t * (b[i] - a[i]) for i in range(3)]


_CONNECT_TIMEOUT: float = 5.0  # seconds before giving up on CoppeliaSim


def _connect_zmq(host: str, port: int) -> tuple:
    """Open a ZeroMQ connection to CoppeliaSim with a hard timeout.

    Runs the blocking RemoteAPIClient handshake in a daemon thread so the
    caller is not stuck forever when CoppeliaSim is not running.

    Returns (client, sim).
    Raises ConnectionError after _CONNECT_TIMEOUT seconds with no response.
    """
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError as exc:
        raise ImportError("Run: pip install coppeliasim-zmqremoteapi-client") from exc

    result: list = [None]
    error:  list = [None]
    done = threading.Event()

    def _try() -> None:
        try:
            c = RemoteAPIClient(host, port)
            result[0] = (c, c.require('sim'))
        except Exception as exc:  # noqa: BLE001
            error[0] = exc
        done.set()

    threading.Thread(target=_try, daemon=True).start()
    if not done.wait(timeout=_CONNECT_TIMEOUT):
        raise ConnectionError(
            f"CoppeliaSim not reachable at {host}:{port} "
            f"after {_CONNECT_TIMEOUT:.0f} s.\n"
            "  → Start CoppeliaSim and load the plant scene, then try again."
        )
    if error[0] is not None:
        raise error[0]
    return result[0]


class CoppeliaSimBridge:
    """
    One-way bridge: pushes AGV state from Python to CoppeliaSim each step.

    Scene hierarchy expected (created by scene_builder.py):
        /AGVs/AGV_i            ← dummy root  (bridge moves + rotates this)
        /AGVs/AGV_i/AGV_i_Light ← status sphere (bridge colors this)
    """

    def __init__(self, n_agvs: int, host: str = 'localhost', port: int = 23000):
        print(f"[bridge] Connecting to CoppeliaSim at {host}:{port} ...")
        self._client, self._sim = _connect_zmq(host, port)
        self._n_agvs = n_agvs
        self._bases:  List[int]
        self._lights: List[int]
        self._bases, self._lights = self._resolve_handles(n_agvs)

        # Last world position sent to CoppeliaSim (for interpolation)
        self._visual_pos: List[Optional[List[float]]] = [None] * n_agvs
        # Current heading angle per AGV (radians, world Z axis)
        self._heading: List[float] = [0.0] * n_agvs

        # Task marker state
        self._task_markers:      Dict[int, int]             = {}  # task_id → sphere handle
        self._task_status_cache: Dict[int, TaskStatus]      = {}  # last seen status
        self._task_delivery:     Dict[int, Tuple[int, int]] = {}  # task_id → delivery cell
        self._flash_queue:       List[int]                  = []  # handles removed next tick

        # For 'both' render mode: world position at the start of the current step
        # (used by sync_frame to interpolate between step_start and target)
        self._step_start_pos: Dict[int, List[float]] = {}
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
    # Synchronised 'both' mode  (Pygame drives timing)
    # ------------------------------------------------------------------

    def sync_step(self, agvs: List[AGV], tasks: List[Task]) -> None:
        """
        Call once per env step when running in 'both' mode.

        Records the step-start world positions for sync_frame, rotates each
        AGV to face its new direction, and updates status-light colors and
        task markers.  Does NOT update AGV positions — sync_frame does that
        at Pygame's render rate so both visualisers move in lockstep.
        """
        sim = self._sim
        for agv in agvs:
            if agv.id >= self._n_agvs:
                continue
            target = _grid_to_world(*agv.position)
            start  = self._visual_pos[agv.id] or target
            self._step_start_pos[agv.id] = start
            self._update_heading(agv.id, start, target)
            color = _STATUS_COLORS.get(agv.status, [0.95, 0.95, 0.95])
            sim.setShapeColor(self._lights[agv.id], '', _COLOR_AMBIENT, color)
            self._visual_pos[agv.id] = target
        self.sync_tasks(tasks)

    def sync_frame(self, agvs: List[AGV], anim_t: dict) -> None:
        """
        Call every Pygame render frame (~60 fps) in 'both' mode.

        Interpolates each AGV's CoppeliaSim position using the same fractional
        progress `t` that Pygame uses, producing perfectly synchronised motion
        in both visualisers.

        Parameters
        ----------
        anim_t : dict {agv_id: float}
            renderer._anim_t — value in [0, 1], 0 = step start, 1 = step end.
        """
        sim = self._sim
        for agv in agvs:
            if agv.id >= self._n_agvs:
                continue
            t      = min(1.0, anim_t.get(agv.id, 1.0))
            target = _grid_to_world(*agv.position)
            start  = self._step_start_pos.get(agv.id) or target
            sim.setObjectPosition(self._bases[agv.id], -1, _lerp(start, target, t))

    # ------------------------------------------------------------------
    # Task markers
    # ------------------------------------------------------------------

    def sync_tasks(self, tasks: List[Task]) -> None:
        """
        Update floating task-marker spheres in CoppeliaSim.

        Rules:
          PENDING / ASSIGNED  → red sphere above pickup cell
          IN_PROGRESS         → green sphere above delivery cell
                                + yellow flash at pickup cell (one tick only)
          Completed (removed) → yellow flash at delivery cell (one tick only)

        Call once per env step, before or after sync().
        """
        for h in self._flash_queue:
            self._try_remove(h)
        self._flash_queue.clear()

        current_ids = {t.id for t in tasks}
        self._expire_completed_tasks(current_ids)
        for task in tasks:
            self._update_task_marker(task)

    def _expire_completed_tasks(self, current_ids: set) -> None:
        """Remove markers for tasks that have left the active list."""
        for tid in set(self._task_markers) - current_ids:
            self._try_remove(self._task_markers.pop(tid))
            self._task_status_cache.pop(tid, None)
            if tid in self._task_delivery:
                r, c = self._task_delivery.pop(tid)
                self._flash_queue.append(self._create_marker(r, c, _COLOR_FLASH, _FLASH_R))

    def _update_task_marker(self, task: Task) -> None:
        """Create or update the marker sphere for a single task."""
        prev_status = self._task_status_cache.get(task.id)
        self._task_status_cache[task.id] = task.status
        self._task_delivery[task.id]     = task.delivery

        if task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
            if task.id not in self._task_markers:
                self._task_markers[task.id] = self._create_marker(
                    *task.pickup, _COLOR_PICKUP, _MARKER_R
                )
        elif task.status == TaskStatus.IN_PROGRESS:
            self._handle_in_progress_marker(task, prev_status)

    def _handle_in_progress_marker(self, task: Task, prev_status: Optional[TaskStatus]) -> None:
        """Swap pickup→delivery marker and emit pickup flash on transition."""
        if prev_status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
            self._try_remove(self._task_markers.pop(task.id, -1))
            self._flash_queue.append(
                self._create_marker(*task.pickup, _COLOR_FLASH, _FLASH_R)
            )
        if task.id not in self._task_markers:
            self._task_markers[task.id] = self._create_marker(
                *task.delivery, _COLOR_DELIVERY, _MARKER_R
            )

    def _create_marker(self, row: int, col: int, color: list, radius: float) -> int:
        """Create a small sphere above a grid cell and return its handle."""
        sim  = self._sim
        d    = radius * 2
        wx   = (_GRID_SIZE - 1 - col + 0.5) * _CELL_SIZE   # mirrored, see _grid_to_world
        wy   = (row + 0.5)                  * _CELL_SIZE
        handle = sim.createPrimitiveShape(_PRIM_SPHERE, [d, d, d], 0)
        sim.setObjectPosition(handle, -1, [wx, wy, _MARKER_Z])
        sim.setShapeColor(handle, '', _COLOR_AMBIENT, color)
        return handle

    def _try_remove(self, handle: int) -> None:
        """Remove a CoppeliaSim object silently (ignore if already gone)."""
        if handle < 0:
            return
        try:
            self._sim.removeObjects([handle], False)
        except Exception:
            pass

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
