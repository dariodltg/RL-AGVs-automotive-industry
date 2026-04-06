"""
Strategy interface for AGV pathfinding algorithms.

Concrete implementations (A*, Theta*) must subclass BasePathfinder
and implement compute_path(). The environment calls this interface,
so swapping algorithms requires no changes outside the navigation module.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from src.env.plant_map import PlantMap


class BasePathfinder(ABC):
    """
    Abstract base class for grid pathfinding algorithms.

    All pathfinders operate on a PlantMap grid and return an ordered
    list of (row, col) cells from start (exclusive) to goal (inclusive).
    An empty list means the goal is unreachable or start == goal.
    """

    @abstractmethod
    def compute_path(
        self,
        plant: PlantMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Compute a path from start to goal on the given plant map.

        Parameters
        ----------
        plant : PlantMap
            The grid map with obstacle and cell type information.
        start : Tuple[int, int]
            Starting cell (row, col).
        goal : Tuple[int, int]
            Destination cell (row, col).

        Returns
        -------
        List[Tuple[int, int]]
            Ordered list of cells to traverse, excluding start, including goal.
            Empty list if start == goal or no path exists.
        """
        ...
