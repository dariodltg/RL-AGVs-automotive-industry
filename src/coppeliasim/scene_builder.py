"""
CoppeliaSim scene builder for the automotive AGV plant.

Reads the 20x20 PlantMap grid and programmatically creates a 3D scene in
CoppeliaSim using the ZeroMQ Remote API, mirroring the existing Pygame layout.

Cell types → 3D objects:
  FREE      → thin grey floor tile
  OBSTACLE  → tall dark cuboid (walls / fixed machinery)
  STAMPING  → blue station block
  BUFFER    → orange station block
  WELDING   → red station block
  ENTRY     → green low block
  EXIT      → yellow low block
  CHARGING  → cyan low block
  AGV       → white cylinder (one per spawn position)

Requirements:
    pip install coppeliasim-zmqremoteapi-client

Usage:
    1. Start CoppeliaSim and open a new empty scene.
    2. Run from the repo root:
           python -m src.coppeliasim.scene_builder
    3. Save the generated scene in CoppeliaSim (File > Save scene as ...).
"""

import sys
import time
from pathlib import Path


class SceneAlreadyExistsError(RuntimeError):
    """Raised when /Plant or /AGVs already exist in the CoppeliaSim scene."""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.env.plant_map import CellType, PlantMap

# Meters per grid cell: 20x20 grid → 10m × 10m plant footprint
CELL_SIZE: float = 0.5

# Visual properties per cell type: (rgb, height_m)
# Small gap (0.01 m) between cells is achieved via CELL_SIZE - _GAP below
_VISUALS: dict = {
    CellType.FREE:     ([0.78, 0.78, 0.78], 0.03),
    CellType.OBSTACLE: ([0.22, 0.22, 0.22], 1.50),
    CellType.ENTRY:    ([0.20, 0.80, 0.20], 0.20),
    CellType.STAMPING: ([0.20, 0.40, 0.85], 0.40),
    CellType.BUFFER:   ([0.95, 0.60, 0.10], 0.30),
    CellType.WELDING:  ([0.85, 0.20, 0.20], 0.40),
    CellType.EXIT:     ([0.95, 0.85, 0.10], 0.20),
    CellType.CHARGING: ([0.10, 0.85, 0.85], 0.15),
}

_GAP: float = 0.01  # gap between adjacent cells (visual separation)

# sim.primitiveshape_sphere is not exposed in all ZMQ client versions — use literal
_PRIM_SPHERE = 1

# CoppeliaSim color component integer (ambient/diffuse)
_COLOR_AMBIENT = 0

# AGV model file name (inside CoppeliaSim's models/robots/mobile/)
_AGV_MODEL_NAME: str = 'Omnidirectional platform.ttm'

# Scale: OmniPlatform already fits a 0.5m cell — no scaling needed
_AGV_SCALE: float = 1.0

# Z offset to lift the model so its base sits on Z=0 (ground).
# The OmniPlatform reference frame is roughly at the model centre (~0.16m up).
_AGV_Z_OFFSET: float = 0.157   # adjust here if the model is still sunk or floating

# Status light sphere added on top of each AGV
_LT_D: float = 0.07
_LT_Z: float = 0.35   # height above ground (~top of OmniPlatform)

# Ground plane extends this many cells beyond the plant border
_GROUND_MARGIN: int = 2

# Common names CoppeliaSim uses for its default scene floor object
_DEFAULT_FLOOR_NAMES = [
    '/ResizableFloor_5_25',
    '/Floor',
    'ResizableFloor_5_25',
    'Floor',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell_center(row: int, col: int, height: float) -> list:
    """World position [x, y, z] for the centre of a grid cell.

    CoppeliaSim coordinate system: Z is UP, XY is the ground plane.
      X = column direction  (right)
      Y = row direction     (depth / forward)
      Z = vertical height   (up)
    """
    return [
        (col + 0.5) * CELL_SIZE,   # X
        (row + 0.5) * CELL_SIZE,   # Y  (row → depth)
        height / 2.0,               # Z  (height above ground)
    ]


def _create_cuboid(sim, sizes: list, pos: list, color: list, name: str, parent: int) -> int:
    handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, sizes, 0)
    if not isinstance(handle, int) or handle < 0:
        raise RuntimeError(
            f"createPrimitiveShape failed for '{name}': returned {handle!r}\n"
            f"  sizes={sizes}  pos={pos}\n"
            f"  sim.primitiveshape_cuboid={sim.primitiveshape_cuboid!r}"
        )
    sim.setObjectPosition(handle, -1, pos)
    sim.setShapeColor(handle, '', _COLOR_AMBIENT, color)
    sim.setObjectAlias(handle, name)
    sim.setObjectParent(handle, parent, True)
    return handle


def _find_agv_model_path(sim) -> str:
    """Resolve the AGV .ttm path using CoppeliaSim's install directory."""
    import os
    app_path = sim.getStringParam(sim.stringparam_application_path)
    path = os.path.join(app_path, 'models', 'robots', 'mobile', _AGV_MODEL_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"AGV model not found at:\n  {path}\n"
            "Verify your CoppeliaSim installation contains the model library."
        )
    return path


def _load_agv_model(sim, idx: int, row: int, col: int, parent: int, model_path: str) -> tuple:
    """
    Load an AGV model at grid position (row, col).

    Hierarchy:
        AGV_{idx}          ← dummy at Z=0, orientation [0,0,0]  (bridge moves + rotates this)
        ├── <model root>   ← OmniPlatform at _AGV_Z_OFFSET, natural pose preserved
        └── AGV_{idx}_Light ← status sphere, bridge changes color

    Using a neutral dummy as root means bridge body-rotation is a simple
    setObjectOrientation([0, 0, heading]) with no Euler-angle composition needed.

    Returns (dummy_handle, light_handle).
    """
    world_x = (col + 0.5) * CELL_SIZE
    world_y = (row + 0.5) * CELL_SIZE
    name = f'AGV_{idx}'

    # Dummy root — bridge moves and rotates this; model + light follow
    dummy = sim.createDummy(0.01)
    sim.setObjectPosition(dummy, -1, [world_x, world_y, 0.0])
    sim.setObjectOrientation(dummy, -1, [0.0, 0.0, 0.0])
    sim.setObjectAlias(dummy, name)
    sim.setObjectParent(dummy, parent, True)

    # Load model as child of dummy at its natural Z offset
    model_root = sim.loadModel(model_path)
    if abs(_AGV_SCALE - 1.0) > 1e-6:
        sim.scaleObject(model_root, _AGV_SCALE, _AGV_SCALE, _AGV_SCALE, 0)
    sim.setObjectPosition(model_root, -1, [world_x, world_y, _AGV_Z_OFFSET])
    sim.setObjectParent(model_root, dummy, True)   # preserves world pos/orient

    # Status light as child of dummy (so it rotates with the body)
    light = sim.createPrimitiveShape(_PRIM_SPHERE, [_LT_D, _LT_D, _LT_D], 0)
    sim.setObjectPosition(light, -1, [world_x, world_y, _LT_Z])
    sim.setShapeColor(light, '', _COLOR_AMBIENT, [0.95, 0.95, 0.95])
    sim.setObjectAlias(light, f'{name}_Light')
    sim.setObjectParent(light, dummy, True)

    return dummy, light


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def _prepare_scene(sim) -> None:
    """
    Guard against running on a scene that already has plant objects,
    then remove the CoppeliaSim default floor if present.

    If /Plant or /AGVs already exist the function aborts with instructions
    because programmatic subtree removal via ZMQ is unreliable across
    CoppeliaSim versions. The clean solution is to start from an empty scene.
    """
    for alias in ['/Plant', '/AGVs']:
        try:
            sim.getObject(alias)
            raise SceneAlreadyExistsError(
                f"'{alias}' already exists in the scene. "
                "Open a fresh scene in CoppeliaSim (File → New scene) and try again."
            )
        except SceneAlreadyExistsError:
            raise
        except Exception:
            pass  # object not found → scene is clean for this alias

    for name in _DEFAULT_FLOOR_NAMES:
        try:
            sim.removeObjects([sim.getObject(name)], False)
            print(f"  Removed default floor: {name}")
            return
        except Exception:
            pass


def build_scene(host: str = 'localhost', port: int = 23000, n_agvs: int = 4) -> None:
    """Connect to CoppeliaSim and generate the plant scene.

    Parameters
    ----------
    n_agvs : int
        Number of AGV models to place (must match what run_coppeliasim.py uses).
        Maximum is len(PlantMap.AGV_SPAWN_POSITIONS) = 8.
    """
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError:
        print("ERROR: ZMQ client not installed.")
        print("       pip install coppeliasim-zmqremoteapi-client")
        sys.exit(1)

    print(f"Connecting to CoppeliaSim at {host}:{port} ...")
    client = RemoteAPIClient(host=host, port=port)
    sim = client.require('sim')
    print("Connected.")

    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        time.sleep(0.5)

    _prepare_scene(sim)

    plant = PlantMap()
    grid  = plant.grid
    n     = plant.GRID_SIZE

    # Root dummy: groups all plant objects in the scene hierarchy
    plant_root = sim.createDummy(0.01)
    sim.setObjectAlias(plant_root, 'Plant')

    # --- Ground plane: flat slab on XY plane, Z=-0.01 (just below ground level) ---
    ground_size = (n + _GROUND_MARGIN * 2) * CELL_SIZE
    ground_cx   = (n / 2.0) * CELL_SIZE
    ground_cy   = (n / 2.0) * CELL_SIZE
    _create_cuboid(
        sim,
        sizes=[ground_size, ground_size, 0.02],  # [X, Y, Z] — thin in Z (up)
        pos=[ground_cx, ground_cy, -0.01],
        color=[0.45, 0.45, 0.45],
        name='Ground',
        parent=plant_root,
    )

    # --- Build grid cells ---
    print(f"Building {n}×{n} grid ({n * n} cells)...")
    tile_w = CELL_SIZE - _GAP

    for r in range(n):
        for c in range(n):
            cell_type = CellType(int(grid[r, c]))
            color, height = _VISUALS[cell_type]

            _create_cuboid(
                sim,
                sizes=[tile_w, tile_w, height],  # [X, Y, Z] — height along Z (up)
                pos=_cell_center(r, c, height),
                color=color,
                name=f'{cell_type.name}_{r}_{c}',
                parent=plant_root,
            )

    # --- Place AGV models at spawn positions ---
    agv_root = sim.createDummy(0.01)
    sim.setObjectAlias(agv_root, 'AGVs')

    spawn_positions = plant.get_agv_spawn_positions(n_agvs)
    agv_path = _find_agv_model_path(sim)
    print(f"AGV model: {agv_path}")
    print(f"Placing {n_agvs} AGV models...")
    for i, (r, c) in enumerate(spawn_positions):
        _load_agv_model(sim, idx=i, row=r, col=c, parent=agv_root, model_path=agv_path)

    # --- Orient default camera to top-down view ---
    # CoppeliaSim Z-up: camera at high Z looks along -Z (down) with default orientation.
    plant_cx = (n / 2.0) * CELL_SIZE
    plant_cy = (n / 2.0) * CELL_SIZE
    try:
        cam = sim.getObject('/DefaultCamera')
        sim.setObjectPosition(cam, -1, [plant_cx, plant_cy, 14.0])
        sim.setObjectOrientation(cam, -1, [0.0, 0.0, 0.0])   # default → looks along -Z = down
    except Exception:
        pass

    print("\nScene generated successfully.")
    print(f"  Plant footprint : {n * CELL_SIZE:.1f} m × {n * CELL_SIZE:.1f} m")
    print(f"  Cells created   : {n * n}")
    print(f"  AGVs placed     : {n_agvs}")
    print("\nNext step: File > Save scene as ... in CoppeliaSim")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n_agvs', type=int, default=4, help='Number of AGVs to place (default 4)')
    p.add_argument('--host',   type=str, default='localhost')
    p.add_argument('--port',   type=int, default=23000)
    args = p.parse_args()
    try:
        build_scene(host=args.host, port=args.port, n_agvs=args.n_agvs)
    except SceneAlreadyExistsError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
