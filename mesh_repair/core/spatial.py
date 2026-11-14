"""Uniform-grid spatial index used for proximity queries.

Deliberately dependency-free (no scipy).  Cells are keyed by integer division
of coordinates, so it works for models in any unit as long as the caller
passes a cell size in model units.
"""
from __future__ import annotations

import numpy as np


class GridIndex:
    def __init__(self, points: np.ndarray, cell: float):
        self.points = np.asarray(points, dtype=np.float64)
        self.cell = float(cell) if cell > 0 else 1e-9
        if len(self.points) == 0:
            self.cells = np.zeros((0, 3), dtype=np.int64)
            self.buckets: dict[tuple[int, int, int], list[int]] = {}
            return
        cells = np.floor(self.points / self.cell)
        # non-finite coordinates cannot be bucketed; park them far away so
        # they never match a real neighbour instead of raising on cast.
        finite = np.isfinite(cells)
        cells = np.where(finite, cells, 0.0)
        self.cells = cells.astype(np.int64)
        self.valid = np.all(finite, axis=1)
        self.buckets = {}
        for i, c in enumerate(self.cells):
            if not self.valid[i]:
                continue
            self.buckets.setdefault((int(c[0]), int(c[1]), int(c[2])), []).append(i)

    def close_pairs(self, tol: float) -> list[tuple[int, int]]:
        """All vertex pairs within tol; each pair returned once."""
        if len(self.points) == 0:
            return []
        # Query radius spans at most two cells on each side when tol <= cell.
        reach = int(np.ceil(tol / self.cell)) if self.cell > 0 else 1
        reach = max(1, min(reach, 2))
        seen: set[tuple[int, int]] = set()
        pts = self.points
        for (cx, cy, cz), members in self.buckets.items():
            for dx in range(-reach, reach + 1):
                for dy in range(-reach, reach + 1):
                    for dz in range(-reach, reach + 1):
                        other = self.buckets.get((cx + dx, cy + dy, cz + dz))
                        if not other:
                            continue
                        for i in members:
                            for j in other:
                                if i >= j:
                                    continue
                                key = (i, j)
                                if key in seen:
                                    continue
                                d = pts[i] - pts[j]
                                if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= tol * tol:
                                    seen.add(key)
        return sorted(seen)


def points_in_sphere(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=np.int64)
    d = points - center
    return np.nonzero(np.einsum("ij,ij->i", d, d) <= radius * radius)[0]
