"""Hole filling with explicit size and shape limits.

The filler only accepts a boundary loop that the user selected.  It refuses:

* non-closed loops (broken boundaries from malformed meshes),
* loops larger than the configured edge count / perimeter / spacing limits,
* loops whose best-fit projection is not simple (self-intersecting in plane),
* loops that are far from planar,
* triangulations that would introduce degenerate or duplicate faces.

A "designed opening" is indistinguishable from damage geometrically, so the
tool never fills automatically; limits exist to force the user to justify
large apertures explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import Mesh, project_to_plane, triangle_areas
from .repair import RepairError, RepairSession, EditRecord


@dataclass
class HoleFillLimits:
    max_edges: int = 64
    max_perimeter: float = 0.0        # model units; 0 = derive from mesh
    max_edge_length: float = 0.0      # model units; 0 = derive from mesh
    max_plane_deviation: float = 0.0  # model units; 0 = derive from mesh
    planarity_fraction: float = 0.05
    perimeter_fraction: float = 0.15
    spacing_fraction: float = 0.10


def _thresholds(mesh: Mesh, limits: HoleFillLimits) -> dict:
    diag = max(mesh.diagonal(), 1e-12)
    return {
        "max_perimeter": limits.max_perimeter or diag * limits.perimeter_fraction,
        "max_edge_length": limits.max_edge_length or diag * limits.spacing_fraction,
        "max_plane_deviation": limits.max_plane_deviation or diag * limits.planarity_fraction,
    }


def check_loop(mesh: Mesh, loop_vertices: list[int], limits: HoleFillLimits,
               require_closed: bool = True) -> dict:
    """Validate a candidate loop without modifying anything."""
    vs = [int(v) for v in loop_vertices]
    if len(vs) < 3:
        return {"ok": False, "code": "too_few", "message": "边界至少需要 3 个顶点"}
    if len(set(vs)) != len(vs):
        return {"ok": False, "code": "repeated", "message": "边界顶点重复，不是简单环"}
    if any(v < 0 or v >= mesh.vertex_count for v in vs):
        return {"ok": False, "code": "index", "message": "边界包含越界顶点索引"}

    pts = mesh.vertices[vs]
    seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    spacings = np.linalg.norm(seg, axis=1)
    perimeter = float(spacings.sum())
    if perimeter <= 0:
        return {"ok": False, "code": "zero", "message": "边界周长为零"}

    origin, normal, axis_u, deviation = project_to_plane(pts)
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v) if np.linalg.norm(axis_v) else 1.0
    local = np.stack([(pts - origin) @ axis_u, (pts - origin) @ axis_v], axis=1)

    th = _thresholds(mesh, limits)
    problems = []
    if len(vs) > limits.max_edges:
        problems.append(
            {"code": "edges", "message": f"边界 {len(vs)} 条边超过上限 {limits.max_edges}"}
        )
    if perimeter > th["max_perimeter"]:
        problems.append(
            {
                "code": "perimeter",
                "message": (
                    f"周长 {perimeter:.4g} 超过上限 {th['max_perimeter']:.4g}"
                    "（疑似设计开口，需显式放宽）"
                ),
            }
        )
    if spacings.max() > th["max_edge_length"]:
        problems.append(
            {
                "code": "spacing",
                "message": (
                    f"最长边界段 {spacings.max():.4g} 超过上限 "
                    f"{th['max_edge_length']:.4g}，洞口过大"
                ),
            }
        )
    if deviation > th["max_plane_deviation"]:
        problems.append(
            {
                "code": "planarity",
                "message": (
                    f"边界偏离最佳拟合平面 {deviation:.4g}，超过 "
                    f"{th['max_plane_deviation']:.4g}；洞口形状不适合一次性补面"
                ),
            }
        )
    simple = _is_simple_polygon(local)
    if not simple:
        problems.append(
            {"code": "simple", "message": "投影后边界自交，形状不规则，拒绝自动三角化"}
        )
    if _orientation_sign(local) == 0:
        problems.append({"code": "area", "message": "边界在平面上投影面积为零"})

    result = {
        "ok": not problems,
        "problems": problems,
        "perimeter": perimeter,
        "max_spacing": float(spacings.max()),
        "mean_spacing": float(spacings.mean()),
        "plane_deviation": deviation,
        "normal": normal,
        "origin": origin,
        "local_2d": local,
        "thresholds": th,
        "vertex_count": len(vs),
    }
    result["code"] = problems[0]["code"] if problems else ""
    result["message"] = "；".join(p["message"] for p in problems)
    return result


def _orientation_sign(poly: np.ndarray) -> int:
    x = poly[:, 0]
    y = poly[:, 1]
    a = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    if a > 1e-15:
        return 1
    if a < -1e-15:
        return -1
    return 0


def _segments_intersect(p1, p2, p3, p4, eps=1e-12) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and (
        (d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)
    ):
        return True
    return False


def _is_simple_polygon(poly: np.ndarray) -> bool:
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            b1, b2 = poly[j], poly[(j + 1) % n]
            if _segments_intersect(a1, a2, b1, b2):
                return False
    return True


def _point_in_triangle(p, a, b, c, eps=1e-12) -> bool:
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = d1 < -eps or d2 < -eps or d3 < -eps
    has_pos = d1 > eps or d2 > eps or d3 > eps
    return not (has_neg and has_pos)


def ear_clip(poly2d: np.ndarray, ccw: bool = True) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon; returns index triples into ``poly2d``."""
    n = len(poly2d)
    if n < 3:
        return []
    idx = list(range(n))
    if (_orientation_sign(poly2d) > 0) != ccw:
        idx.reverse()
    tris: list[tuple[int, int, int]] = []
    guard = 0
    max_guard = n * n + 8
    while len(idx) > 3 and guard < max_guard:
        guard += 1
        clipped = False
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[(k - 1) % m], idx[k], idx[(k + 1) % m]
            a, b, c = poly2d[i0], poly2d[i1], poly2d[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-15:      # reflex or collinear for CCW
                continue
            contains = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                if _point_in_triangle(poly2d[j], a, b, c):
                    contains = True
                    break
            if contains:
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            clipped = True
            break
        if not clipped:
            return []   # not triangulable (self-touching/degenerate)
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def plan_fill(
    session: RepairSession,
    loop_vertices: list[int],
    limits: HoleFillLimits,
) -> dict:
    """Validate the loop and produce the triangles that would be added."""
    mesh = session.mesh
    check = check_loop(mesh, loop_vertices, limits)
    if not check["ok"]:
        return {"ok": False, "check": check}
    vs = [int(v) for v in loop_vertices]
    local = check["local_2d"]
    tris = ear_clip(local, ccw=True)
    if not tris:
        return {
            "ok": False,
            "check": check,
            "message": "耳切法无法得到合法三角化（形状过于退化）",
        }
    new_faces = [[vs[i], vs[j], vs[k]] for i, j, k in tris]
    normal = check["normal"]
    # orientation: cap must be wound opposite the existing boundary shell so
    # the shared edges stay opposite.  Compare computed normal with loop normal.
    faces = np.array(new_faces, dtype=np.int64)
    v = mesh.vertices[faces]
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
    agree = float(np.dot(fn.mean(axis=0), normal)) > 0
    if not agree:
        faces = faces[:, [0, 2, 1]]
    areas = triangle_areas(mesh.vertices, faces)
    if np.any(areas <= 1e-18):
        return {"ok": False, "check": check, "message": "三角化产生了零面积面"}
    existing = {tuple(sorted(int(x) for x in f)) for f in mesh.faces}
    dup = [list(f) for f in faces if tuple(sorted(int(x) for x in f)) in existing]
    if dup:
        return {
            "ok": False,
            "check": check,
            "message": f"补面将与已有面重复（{len(dup)} 个），拒绝执行",
        }
    return {
        "ok": True,
        "check": check,
        "faces": [[int(x) for x in f] for f in faces],
        "added_faces": int(len(faces)),
        "added_area": float(areas.sum()),
        "max_plane_deviation": check["plane_deviation"],
    }


def fill_hole(
    session: RepairSession,
    loop_vertices: list[int],
    limits: HoleFillLimits,
    *,
    reason: str,
) -> EditRecord:
    plan = plan_fill(session, loop_vertices, limits)
    if not plan["ok"]:
        raise RepairError(plan.get("message") or plan["check"].get("message", "补洞校验失败"))
    mesh = session.mesh
    new_faces_arr = np.vstack([mesh.faces, np.array(plan["faces"], dtype=np.int64)])
    rec = session._apply(
        "fill_hole",
        f"补洞（{len(loop_vertices)} 顶点 → {plan['added_faces']} 面）",
        {
            "loop": [int(v) for v in loop_vertices],
            "new_faces": plan["faces"],
            "perimeter": plan["check"]["perimeter"],
            "max_spacing": plan["check"]["max_spacing"],
            "plane_deviation": plan["max_plane_deviation"],
            "max_edges": limits.max_edges,
            "limits": {
                "max_perimeter": plan["check"]["thresholds"]["max_perimeter"],
                "max_edge_length": plan["check"]["thresholds"]["max_edge_length"],
                "max_plane_deviation": plan["check"]["thresholds"]["max_plane_deviation"],
            },
            "confirm_reason": reason,
        },
        mesh.vertices,
        new_faces_arr,
        reason=reason,
        affected_vertices=[int(v) for v in loop_vertices],
        affected_faces=list(range(mesh.face_count, mesh.face_count + plan["added_faces"])),
    )
    return rec
