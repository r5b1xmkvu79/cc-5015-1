"""Triangle mesh container and geometric primitives."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_triangles(faces: np.ndarray) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces 必须是 (N, 3) 三角面索引数组")
    return f


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    name: str = "mesh"

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = _as_triangles(self.faces)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices 必须是 (N, 3) 数组")

    # -- basic copies ---------------------------------------------------
    def copy(self, name: str | None = None) -> "Mesh":
        return Mesh(self.vertices.copy(), self.faces.copy(), name or self.name)

    # -- measurements ---------------------------------------------------
    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])

    def bounds(self) -> np.ndarray:
        if self.vertex_count == 0:
            return np.zeros((2, 3), dtype=np.float64)
        return np.stack([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    def diagonal(self) -> float:
        if self.vertex_count == 0:
            return 0.0
        span = self.bounds()[1] - self.bounds()[0]
        return float(np.linalg.norm(span))

    def extent(self) -> np.ndarray:
        if self.vertex_count == 0:
            return np.zeros(3)
        return self.bounds()[1] - self.bounds()[0]

    def triangle_areas(self) -> np.ndarray:
        return triangle_areas(self.vertices, self.faces)

    def area(self) -> float:
        return float(self.triangle_areas().sum())

    def signed_volume(self) -> float:
        if self.face_count == 0:
            return 0.0
        v = self.vertices[self.faces]
        return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)

    def face_normals(self, normalized: bool = True) -> np.ndarray:
        v = self.vertices[self.faces]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        if normalized:
            lengths = np.linalg.norm(n, axis=1, keepdims=True)
            lengths[lengths == 0] = 1.0
            n = n / lengths
        return n

    def face_centroids(self) -> np.ndarray:
        if self.face_count == 0:
            return np.zeros((0, 3))
        return self.vertices[self.faces].mean(axis=1)

    def vertex_normals(self) -> np.ndarray:
        if self.face_count == 0:
            return np.zeros_like(self.vertices)
        fn = self.face_normals(normalized=False)
        vn = np.zeros_like(self.vertices)
        for k in range(3):
            np.add.at(vn, self.faces[:, k], fn)
        lengths = np.linalg.norm(vn, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        return vn / lengths

    def stats(self) -> dict:
        b = self.bounds()
        return {
            "vertices": self.vertex_count,
            "faces": self.face_count,
            "area": self.area(),
            "signed_volume": self.signed_volume(),
            "bbox_min": [float(x) for x in b[0]],
            "bbox_max": [float(x) for x in b[1]],
            "extent": [float(x) for x in self.extent()],
            "diagonal": self.diagonal(),
        }

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not np.all(np.isfinite(self.vertices)):
            issues.append("顶点坐标含 NaN/Inf")
        if self.face_count:
            lo = int(self.faces.min())
            hi = int(self.faces.max())
            if lo < 0 or hi >= self.vertex_count:
                issues.append(f"面索引越界: [{lo}, {hi}] / 顶点数 {self.vertex_count}")
        return issues

    def used_vertices(self) -> np.ndarray:
        if self.face_count == 0:
            return np.zeros(0, dtype=np.int64)
        ids = np.unique(self.faces.reshape(-1))
        if np.any(self.faces < 0):
            ids = ids[ids >= 0]
        return ids

    def compact(self) -> tuple["Mesh", np.ndarray, np.ndarray]:
        """Remove unreferenced vertices.

        Returns (new_mesh, kept_old_vertex_ids, unused_old_vertex_ids).
        """
        used = self.used_vertices()
        mapping = np.full(self.vertex_count, -1, dtype=np.int64)
        mapping[used] = np.arange(used.size, dtype=np.int64)
        new_faces = mapping[self.faces] if self.face_count else self.faces.copy()
        new_vertices = self.vertices[used].copy()
        unused = np.setdiff1d(np.arange(self.vertex_count), used, assume_unique=True)
        return Mesh(new_vertices, new_faces, self.name), used, unused


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros(0, dtype=np.float64)
    v = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1
    )


def triangle_edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    v = vertices[faces]
    return np.stack(
        [
            np.linalg.norm(v[:, 1] - v[:, 0], axis=1),
            np.linalg.norm(v[:, 2] - v[:, 1], axis=1),
            np.linalg.norm(v[:, 0] - v[:, 2], axis=1),
        ],
        axis=1,
    )


def triangle_aspect_ratio(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Longest edge divided by shortest altitude; degenerate -> inf."""
    if len(faces) == 0:
        return np.zeros(0)
    edges = triangle_edge_lengths(vertices, faces)
    areas = triangle_areas(vertices, faces)
    longest = edges.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        height = 2.0 * areas / longest
        ratio = np.where(height > 0, longest / height, np.inf)
    return ratio


def project_to_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Best-fit plane through points.

    Returns (origin, unit_normal, axis_u, max_plane_distance).
    """
    origin = points.mean(axis=0)
    centered = points - origin
    if len(points) < 3:
        return origin, np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), 0.0
    cov = centered.T @ centered
    w, vec = np.linalg.eigh(cov)
    normal = vec[:, 0]
    nlen = np.linalg.norm(normal)
    normal = normal / nlen if nlen else np.array([0.0, 0.0, 1.0])
    distances = np.abs(centered @ normal)
    in_plane = vec[:, 2]
    return origin, normal, in_plane, float(distances.max())


def polygon_area_2d(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_orientation(poly: np.ndarray) -> int:
    return 1 if polygon_area_2d(poly) >= 0 else -1


def clip_polygon_to_convex(points: np.ndarray) -> np.ndarray:
    """Not used for repair; kept out of the pipeline deliberately.

    Repair only accepts simple, near-planar boundary loops, so no clipping
    heuristic is applied to designed openings.
    """
    return points
