from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import heapq
import json
import numpy as np

FREE_CLASSES = {1, 2}  # ground, road_markings
OCCUPIED_CLASSES = {3, 4, 6, 7, 8}  # natural, building, pole, car, fence


@dataclass
class GridSpec:
    resolution: float = 0.5
    inflate: int = 1


def build_grid(
    xy: np.ndarray, labels: np.ndarray, spec: GridSpec
) -> Tuple[np.ndarray, Tuple[float, float], float]:
    """Return occupancy grid (1=occupied, 0=free), origin, resolution."""
    mins = xy.min(axis=0)
    xy_shift = xy - mins
    ij = np.floor(xy_shift / spec.resolution).astype(int)
    max_i = ij[:, 0].max() + 1
    max_j = ij[:, 1].max() + 1
    grid = np.zeros((max_i, max_j), dtype=np.uint8)

    for (i, j), lbl in zip(ij, labels):
        if lbl in OCCUPIED_CLASSES:
            grid[i, j] = 1

    if spec.inflate > 0:
        grid = inflate_grid(grid, spec.inflate)

    return grid, mins, spec.resolution


def inflate_grid(grid: np.ndarray, r: int) -> np.ndarray:
    padded = np.pad(grid, r, mode="edge")
    inflated = grid.copy()
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di == 0 and dj == 0:
                continue
            inflated |= padded[r + di : r + di + grid.shape[0], r + dj : r + dj + grid.shape[1]]
    return inflated


def astar(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    open_set = [(h(start), 0, start, None)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_cost = {start: 0}
    visited = set()

    while open_set:
        _, g, current, parent = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        came_from[current] = parent
        if current == goal:
            break
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nxt = (current[0] + di, current[1] + dj)
            if (
                0 <= nxt[0] < grid.shape[0]
                and 0 <= nxt[1] < grid.shape[1]
                and grid[nxt] == 0
            ):
                tentative = g + 1
                if tentative < g_cost.get(nxt, 1e9):
                    g_cost[nxt] = tentative
                    f = tentative + h(nxt)
                    heapq.heappush(open_set, (f, tentative, nxt, current))

    if goal not in came_from:
        return []

    # reconstruct
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    return list(reversed(path))


def save_path(path: List[Tuple[int, int]], origin_xy: Tuple[float, float], res: float, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    metric = [{"x": origin_xy[0] + p[0] * res, "y": origin_xy[1] + p[1] * res} for p in path]
    out.write_text(json.dumps({"path": metric}, indent=2))


