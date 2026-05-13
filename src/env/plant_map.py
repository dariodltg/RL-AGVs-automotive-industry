"""
Plant map for an automotive stamping and welding components facility.

20x20 grid representing a realistic production flow:

  ENTRY → STAMPING → BUFFER → WELDING → EXIT

Cell types:
  ENTRY      — raw material input points (sheet metal from suppliers)
  STAMPING   — stamping press cells (forming sheet metal parts)
  BUFFER     — intermediate WIP storage between stamping and welding
  WELDING    — welding station cells (assembling stamped parts)
  EXIT       — finished component output points
  CHARGING   — AGV charging stations
  OBSTACLE   — walls, fixed machinery, structural elements
  FREE       — circulation corridors

Layout (20x20):
  North wall + ENTRY points (top)
  STAMPING zone (left columns)   |  BUFFER zone (center)  |  WELDING zone (right columns)
  CHARGING stations (mid rows)
  South wall + EXIT points (bottom)
"""

from enum import IntEnum
from typing import List, Tuple

import numpy as np


class CellType(IntEnum):
    FREE     = 0
    OBSTACLE = 1
    ENTRY    = 2   # Raw material input
    STAMPING = 3   # Stamping press cells
    BUFFER   = 4   # WIP intermediate storage
    WELDING  = 5   # Welding station cells
    EXIT     = 6   # Finished component output
    CHARGING = 7   # AGV charging station


# ---------------------------------------------------------------------------
# 20x20 plant layout
#
# Production flow:  ENTRY → STAMPING → BUFFER → WELDING → EXIT
#
# Columns:
#   1-4   : STAMPING zone (left)
#   5-7   : west corridors / obstacles
#   8-11  : BUFFER zone (center)
#   12-14 : east corridors / obstacles
#   15-18 : WELDING zone (right)
#
# Rows:
#   0     : north wall
#   1     : ENTRY points + north corridor
#   2-5   : stamping + buffer + welding north block
#   6-7   : charging + corridor
#   8-9   : central corridor (AGV highway)
#   10-11 : charging + corridor
#   12-15 : stamping + buffer + welding south block
#   17-18 : EXIT points + south corridor
#   19    : south wall
#
# Key:
#   1=OBSTACLE  2=ENTRY  3=STAMPING  4=BUFFER  5=WELDING  6=EXIT  7=CHARGING
# ---------------------------------------------------------------------------

_GRID_LAYOUT = [
    # 0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
    [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 0  north wall
    [1,  0,  2,  0,  0,  0,  0,  0,  0,  2,  2,  0,  0,  0,  0,  0,  0,  2,  0,  1],  # 1  ENTRY points
    [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],  # 2  north corridor
    [1,  3,  3,  0,  0,  1,  1,  0,  4,  4,  4,  4,  0,  1,  1,  0,  5,  5,  0,  1],  # 3  prod block N
    [1,  3,  3,  0,  0,  1,  1,  0,  4,  4,  4,  4,  0,  1,  1,  0,  5,  5,  0,  1],  # 4
    [1,  3,  3,  0,  0,  1,  1,  0,  4,  4,  4,  4,  0,  1,  1,  0,  5,  5,  0,  1],  # 5
    [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],  # 6  corridor
    [1,  0,  0,  0,  0,  0,  0,  0,  7,  0,  0,  7,  0,  0,  0,  0,  0,  0,  0,  1],  # 7  charging N
    [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],  # 8  AGV highway
    [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],  # 9  AGV highway
    [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],  # 10 corridor
    [1,  0,  0,  0,  0,  0,  0,  0,  7,  0,  0,  7,  0,  0,  0,  0,  0,  0,  0,  1],  # 11 charging S
    [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],  # 12 corridor
    [1,  3,  3,  0,  0,  1,  1,  0,  4,  4,  4,  4,  0,  1,  1,  0,  5,  5,  0,  1],  # 13 prod block S
    [1,  3,  3,  0,  0,  1,  1,  0,  4,  4,  4,  4,  0,  1,  1,  0,  5,  5,  0,  1],  # 14
    [1,  3,  3,  0,  0,  1,  1,  0,  4,  4,  4,  4,  0,  1,  1,  0,  5,  5,  0,  1],  # 15
    [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],  # 16 south corridor
    [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],  # 17 south corridor
    [1,  0,  6,  0,  0,  0,  0,  0,  0,  6,  6,  0,  0,  0,  0,  0,  0,  6,  0,  1],  # 18 EXIT points
    [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],  # 19 south wall
]

# Production flow: maps each task stage to (pickup_cell_type, delivery_cell_type)
PRODUCTION_FLOW = [
    (CellType.ENTRY,    CellType.STAMPING),  # stage 0: raw material → stamping
    (CellType.STAMPING, CellType.BUFFER),    # stage 1: stamped parts → buffer
    (CellType.BUFFER,   CellType.WELDING),   # stage 2: buffer → welding
    (CellType.WELDING,  CellType.EXIT),      # stage 3: welded parts → exit
]


class PlantMap:
    """
    2D grid representation of the automotive stamping/welding plant.

    Coordinates: (row, col) where (0, 0) is the top-left corner.
    Row axis grows downward, col axis grows rightward.

    Production flow: ENTRY → STAMPING → BUFFER → WELDING → EXIT
    """

    GRID_SIZE: int = 20

    def __init__(self):
        self.grid: np.ndarray = np.array(_GRID_LAYOUT, dtype=np.int8)
        assert self.grid.shape == (self.GRID_SIZE, self.GRID_SIZE)

        self.entry_cells:     List[Tuple[int, int]] = self._find_cells(CellType.ENTRY)
        self.stamping_cells:  List[Tuple[int, int]] = self._find_cells(CellType.STAMPING)
        self.buffer_cells:    List[Tuple[int, int]] = self._find_cells(CellType.BUFFER)
        self.welding_cells:   List[Tuple[int, int]] = self._find_cells(CellType.WELDING)
        self.exit_cells:      List[Tuple[int, int]] = self._find_cells(CellType.EXIT)
        self.charging_stations: List[Tuple[int, int]] = self._find_cells(CellType.CHARGING)
        self.free_cells:      List[Tuple[int, int]] = self._find_cells(CellType.FREE)

        # Convenience map: CellType → list of positions
        self.cells_by_type = {
            CellType.ENTRY:    self.entry_cells,
            CellType.STAMPING: self.stamping_cells,
            CellType.BUFFER:   self.buffer_cells,
            CellType.WELDING:  self.welding_cells,
            CellType.EXIT:     self.exit_cells,
            CellType.CHARGING: self.charging_stations,
        }

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    def cell_type(self, row: int, col: int) -> CellType:
        return CellType(self.grid[row, col])

    def is_passable(self, row: int, col: int) -> bool:
        """AGVs can traverse any non-obstacle cell."""
        return self.grid[row, col] != CellType.OBSTACLE

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.GRID_SIZE and 0 <= col < self.GRID_SIZE

    def neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """4-connected passable neighbors."""
        candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        return [(r, c) for r, c in candidates if self.in_bounds(r, c) and self.is_passable(r, c)]

    # ------------------------------------------------------------------
    # AGV spawn positions (free cells on the AGV highway, rows 8-9)
    # ------------------------------------------------------------------

    AGV_SPAWN_POSITIONS: List[Tuple[int, int]] = [
        (8, 3), (8, 6), (8, 13), (8, 16),
        (9, 3), (9, 6), (9, 13), (9, 16),
    ]

    def get_agv_spawn_positions(self, n_agvs: int) -> List[Tuple[int, int]]:
        assert n_agvs <= len(self.AGV_SPAWN_POSITIONS), (
            f"Maximum {len(self.AGV_SPAWN_POSITIONS)} AGVs supported for spawn"
        )
        return self.AGV_SPAWN_POSITIONS[:n_agvs]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _find_cells(self, cell_type: CellType) -> List[Tuple[int, int]]:
        rows, cols = np.where(self.grid == int(cell_type))
        return list(zip(rows.tolist(), cols.tolist()))

    @staticmethod
    def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def __repr__(self) -> str:
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
        lines = []
        for row in range(self.GRID_SIZE):
            line = " ".join(symbols[CellType(self.grid[row, col])] for col in range(self.GRID_SIZE))
            lines.append(line)
        return "\n".join(lines)
