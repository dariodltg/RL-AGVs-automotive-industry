"""
A* pathfinding algorithm for 4-connected grid navigation.

Uses Manhattan distance as the admissible heuristic, which guarantees
optimal paths on grids with uniform movement cost (no diagonals).

Time complexity  : O(n log n) where n = number of passable cells
Space complexity : O(n)
"""

import heapq
from typing import Dict, List, Optional, Tuple

from src.env.plant_map import PlantMap
from .pathfinder import BasePathfinder


class AStarPathfinder(BasePathfinder):
    """
    A* pathfinder for 4-connected grid maps.

    Finds the shortest obstacle-free path between two cells using
    Manhattan distance as the heuristic. Returns an empty list when
    start == goal or no path exists.
    """

    def compute_path(
        self,
        plant: PlantMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        if start == goal:
            return []

        if not plant.is_passable(*goal):
            return []

        # open_heap entries: (f_score, tie_breaker, node)
        # tie_breaker ensures stable ordering when f_scores are equal
        counter = 0
        open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, counter, start))

        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct_path(came_from, goal)

            for neighbor in plant.neighbors(*current):
                tentative_g = g_score[current] + 1.0  # uniform step cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + PlantMap.manhattan_distance(neighbor, goal)
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_heap, (f_score, counter, neighbor))

        return []  # no path found

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Walks came_from backwards to reconstruct the path (start-exclusive, goal-inclusive)."""
        path: List[Tuple[int, int]] = []
        node: Optional[Tuple[int, int]] = goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path[1:]  # exclude start cell
