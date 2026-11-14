"""Benchmark mesh generators.

Each generator returns a Mesh plus a description of the defects it is
*intended* to contain.  Tests assert that detection finds exactly those and
that repairs do not silently alter the rest of the geometry.

Covered cases:

* ``thin_wall``         – two closed plates a small distance apart; the two
                         layers must survive any proximity merge.
* ``duplicate_faces``  – a closed box with copied and back-facing copy faces.
* ``holed_plate``       – a closed plate with a genuine through-hole plus a
                         designed open notch that must NOT be filled.
* ``mixed_orientation`` – outward box plus an inward-facing closed shell.
* ``open_shell``        – a box missing a wall, so orientation is unprovable.
* ``combined_stress``   – all of the above in one mesh.
"""
from __future__ import annotations

import numpy as np

from ..core.mesh import Mesh
from ..core.topology import build_edge_incidence


def _grid_plate(x0: float, y0: float, w: float, h: float,
                nx: int = 4, ny: int = 4) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(x0, x0 + w, nx + 1)
    ys = np.linspace(y0, y0 + h, ny + 1)
    verts = []
    index = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            index[(i, j)] = len(verts)
            verts.append((x, y))
    faces = []
    for j in range(ny):
        for i in range(nx):
            a = index[(i, j)]
            b = index[(i + 1, j)]
            c = index[(i + 1, j + 1)]
            d = index[(i, j + 1)]
            faces.append((a, b, c))
            faces.append((a, c, d))
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def _sheet_to_solid(plate2d: np.ndarray, faces: np.ndarray, z0: float, thickness: float,
                    name: str) -> Mesh:
    """Extrude a 2D sheet (CCW) into a closed thin plate."""
    n = len(plate2d)
    top = np.column_stack([plate2d, np.full(n, z0 + thickness)])
    bottom = np.column_stack([plate2d, np.full(n, z0)])
    verts = np.vstack([top, bottom])
    nf = faces
    top_faces = nf
    bottom_faces = nf[:, [0, 2, 1]] + n
    inc = build_edge_incidence(top_faces)
    walls = []
    for eid in inc.boundary_edges():
        a, b = inc.half_edges[eid][0]
        # Boundary half-edge (a,b) runs CCW on the top face; the outward wall
        # normal is (b-a) x +Z, which these two winding orders produce.
        walls.append((int(a), int(b) + n, int(b)))
        walls.append((int(a), int(a) + n, int(b) + n))
    all_faces = np.vstack([top_faces, bottom_faces, np.array(walls, dtype=np.int64)])
    return Mesh(verts, all_faces, name=name)


def thin_wall(distance: float = 0.4, size: float = 40.0,
              thickness: float = 1.0) -> tuple[Mesh, dict]:
    """Two closed plates separated by ``distance`` clear space.

    Layer 0 occupies z in [0, thickness]; layer 1 starts at thickness+distance.
    A proximity merge at any threshold >= distance would fuse them into one
    solid; the benchmark asserts 2 components survive.
    """
    p, f = _grid_plate(0, 0, size, size)
    plate_a = _sheet_to_solid(p, f, 0.0, thickness, "wall_a")
    plate_b = _sheet_to_solid(p, f, thickness + distance, thickness, "wall_b")
    v = np.vstack([plate_a.vertices, plate_b.vertices])
    faces = np.vstack([plate_a.faces, plate_b.faces + plate_a.vertex_count])
    mesh = Mesh(v, faces, name="thin_wall")
    return mesh, {
        "expected": {"components": 2, "clearance": float(distance)},
        "notes": "两层闭合薄壁，净间距小于常见焊接阈值；焊接必须保持 2 个组件",
    }


def duplicate_faces() -> tuple[Mesh, dict]:
    v, f = _box(0, 0, 0, 20, 30, 10)
    dup_same = f[[0, 1]]
    dup_rev = f[[2]][:, [0, 2, 1]]
    faces = np.vstack([f, dup_same, dup_rev])
    mesh = Mesh(v, faces, name="duplicate_faces")
    return mesh, {
        "expected": {
            # two same-winding copies + one reversed copy = three groups
            "duplicate_face_instances": 3,
            "same_winding_groups": 2,
            "reversed_groups": 1,
            "base_faces": int(len(f)),
        },
        "notes": "两个同向重复面 + 一个反向重复面",
    }


def holed_plate(size: float = 60.0, thickness: float = 4.0,
                cell: float = 2.0) -> tuple[Mesh, dict]:
    """Closed plate carrying two openings of very different size.

    Built as a fully closed plate first, then:

    * ``小破损孔``: the two top triangles of one cell are removed, leaving a
      small 4-vertex boundary loop that a bounded filler may repair.
    * ``大设计开口``: top and bottom triangles over a 6x6-cell region are
      removed, producing large boundary loops that the default limits must
      reject — a designer's window, not damage.

    Both loops are geometrically similar in shape; only size distinguishes
    them, which is exactly why the limit exists.
    """
    n = int(round(size / cell))
    xs = np.linspace(0, size, n + 1)
    ys = np.linspace(0, size, n + 1)
    verts2d = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    idx = lambda i, j: j * (n + 1) + i

    top_faces = []
    for j in range(n):
        for i in range(n):
            a, b = idx(i, j), idx(i + 1, j)
            c, d = idx(i + 1, j + 1), idx(i, j + 1)
            top_faces.append((a, b, c))
            top_faces.append((a, c, d))

    mesh = _sheet_to_solid(verts2d, np.array(top_faces, dtype=np.int64), 0.0,
                           thickness, "holed_plate")
    nv = len(verts2d)
    # locate triangle indices by their canonical vertex sets
    def tri_index(verts_tri):
        want = tuple(sorted(int(v) for v in verts_tri))
        hits = [k for k, tri in enumerate(mesh.faces) if tuple(sorted(int(v) for v in tri)) == want]
        return hits

    small_cell = (int(n * 0.3), int(n * 0.3))
    sm_i, sm_j = small_cell
    sm_tris = []
    for tri in [(idx(sm_i, sm_j), idx(sm_i + 1, sm_j), idx(sm_i + 1, sm_j + 1)),
                (idx(sm_i, sm_j), idx(sm_i + 1, sm_j + 1), idx(sm_i, sm_j + 1))]:
        sm_tris.extend(tri_index(tri))

    # large opening: remove top and bottom triangles in a cell block
    big_i0, big_j0 = int(n * 0.55), int(n * 0.35)
    big_cells = 6
    big_tris = []
    for di in range(big_cells):
        for dj in range(big_cells):
            ci, cj = big_i0 + di, big_j0 + dj
            for tri in [(idx(ci, cj), idx(ci + 1, cj), idx(ci + 1, cj + 1)),
                        (idx(ci, cj), idx(ci + 1, cj + 1), idx(ci, cj + 1))]:
                big_tris.extend(tri_index(tri))
                # bottom copy uses +nv and reversed winding
                bot = tuple(int(v) + nv for v in tri)
                hits = [k for k, f in enumerate(mesh.faces)
                        if tuple(sorted(int(v) for v in f)) == tuple(sorted(bot))]
                big_tris.extend(hits)

    remove = sorted(set(sm_tris + big_tris))
    if not remove:
        raise RuntimeError("holed_plate 基准构造失败：未定位到要删除的三角形")
    faces = np.delete(mesh.faces, remove, axis=0)
    out = Mesh(mesh.vertices, faces, name="holed_plate")
    # the removed cells leave no unreferenced vertices in the cells themselves,
    # but drop any that arise from construction so the benchmark isolates the
    # openings as its only intentional defect.
    out, _kept, _dropped = out.compact()
    inc = build_edge_incidence(out.faces)
    loops = []
    for lp in _loops(inc):
        pts = out.vertices[lp["vertices"]]
        seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)
        loops.append(float(np.linalg.norm(seg, axis=1).sum()))
    loops.sort()
    return out, {
        "expected": {
            "small_hole_perimeter": loops[0] if loops else None,
            "large_hole_perimeter": loops[-1] if loops else None,
            "small_hole_repairable": True,
            "large_hole_is_designed": True,
            "removed_faces": len(remove),
        },
        "notes": "小孔可补；大开口为设计窗，默认补洞限制应拒绝",
    }


def _loops(inc):
    from ..core.topology import boundary_loops
    return boundary_loops(inc)


def _box(x0, y0, z0, sx, sy, sz, inward: bool = False) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([
        [x0, y0, z0], [x0 + sx, y0, z0], [x0 + sx, y0 + sy, z0], [x0, y0 + sy, z0],
        [x0, y0, z0 + sz], [x0 + sx, y0, z0 + sz], [x0 + sx, y0 + sy, z0 + sz],
        [x0, y0 + sy, z0 + sz],
    ], dtype=np.float64)
    f = np.array([
        [0, 2, 1], [0, 3, 2],   # bottom
        [4, 5, 6], [4, 6, 7],   # top
        [0, 1, 5], [0, 5, 4],   # front
        [1, 2, 6], [1, 6, 5],   # right
        [2, 3, 7], [2, 7, 6],   # back
        [3, 0, 4], [3, 4, 7],   # left
    ], dtype=np.int64)
    if inward:
        f = f[:, [0, 2, 1]]
    return v, f


def mixed_orientation() -> tuple[Mesh, dict]:
    v0, f0 = _box(0, 0, 0, 20, 20, 20)
    v1, f1 = _box(40, 0, 0, 10, 10, 10, inward=True)
    f1 = f1 + len(v0)
    mesh = Mesh(np.vstack([v0, v1]), np.vstack([f0, f1]), name="mixed_orientation")
    return mesh, {
        "expected": {"components": 2, "inward_components": 1, "closed_components": 2},
        "notes": "正常朝外封闭盒 + 整体朝内封闭盒；后者需人工确认后才翻转",
    }


def open_shell() -> tuple[Mesh, dict]:
    v, f = _box(0, 0, 0, 20, 20, 20)
    faces = np.delete(f, [2, 3], axis=0)
    mesh = Mesh(v, faces, name="open_shell")
    return mesh, {
        "expected": {"open_boundary_loops": 1, "ambiguous_orientation": True},
        "notes": "缺顶面的盒子：方向不可由体积判断，必须标记待确认",
    }


def multi_hole(size: float = 48.0, thickness: float = 3.0,
               cell: float = 2.0) -> tuple[Mesh, dict]:
    """Closed plate with three damage holes of increasing size.

    Unlike ``holed_plate`` the holes are all genuine damage, but two of them
    exceed the default fill limits.  This is the multi-hole case used by the
    strategy experiment: Route A patches what fits and leaves the rest;
    Route B (transactional) must apply nothing at all.
    """
    n = int(round(size / cell))
    xs = np.linspace(0, size, n + 1)
    ys = np.linspace(0, size, n + 1)
    verts2d = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    idx = lambda i, j: j * (n + 1) + i

    top_faces = []
    for j in range(n):
        for i in range(n):
            a, b = idx(i, j), idx(i + 1, j)
            c, d = idx(i + 1, j + 1), idx(i, j + 1)
            top_faces.append((a, b, c))
            top_faces.append((a, c, d))

    mesh = _sheet_to_solid(verts2d, np.array(top_faces, dtype=np.int64), 0.0,
                           thickness, "multi_hole")
    nv = len(verts2d)

    def cell_tris(ci, cj, both_sides: bool = True):
        tris = []
        tri_pairs = [
            (idx(ci, cj), idx(ci + 1, cj), idx(ci + 1, cj + 1)),
            (idx(ci, cj), idx(ci + 1, cj + 1), idx(ci, cj + 1)),
        ]
        for tri in tri_pairs:
            want = tuple(sorted(int(v) for v in tri))
            tris.extend(k for k, f in enumerate(mesh.faces)
                        if tuple(sorted(int(v) for v in f)) == want)
            if both_sides:
                bot = tuple(sorted(int(v) + nv for v in tri))
                tris.extend(k for k, f in enumerate(mesh.faces)
                            if tuple(sorted(int(v) for v in f)) == bot)
        return tris

    # two small cells (both fillable by default) and two larger blocks
    # (over the default perimeter limit).
    remove = set(cell_tris(int(n * 0.2), int(n * 0.2)))
    remove.update(cell_tris(int(n * 0.75), int(n * 0.8)))
    for di in range(2):
        for dj in range(2):
            remove.update(cell_tris(int(n * 0.55) + di, int(n * 0.2) + dj))
    for di in range(3):
        for dj in range(2):
            remove.update(cell_tris(int(n * 0.25) + di, int(n * 0.6) + dj))
    out = Mesh(mesh.vertices, np.delete(mesh.faces, sorted(remove), axis=0),
               name="multi_hole")
    out, _kept, _dropped = out.compact()
    loop_perims = _loop_perimeters(out)
    return out, {
        "expected": {
            "holes": len(loop_perims),
            "perimeters": loop_perims,
            "small_hole_repairable": True,
            "larger_holes_over_limit": True,
        },
        "notes": "多洞板：四处小破损在默认限制内、两处较大破损超限；用于比较局部修补与事务式全有全无",
    }


def _loop_perimeters(mesh: Mesh) -> list[float]:
    inc = build_edge_incidence(mesh.faces)
    out = []
    for lp in _loops(inc):
        pts = mesh.vertices[lp["vertices"]]
        seg = np.diff(np.vstack([pts, pts[:1]]), axis=0)
        out.append(round(float(np.linalg.norm(seg, axis=1).sum()), 4))
    out.sort()
    return out


def nonmanifold_junction() -> tuple[Mesh, dict]:
    """Two closed outward boxes touching along exactly one vertical edge.

    Box A occupies [0,10]x[0,10], box B [10,20]x[10,20]; in the xy plane
    they meet only at (10,10), so the shared vertical edge carries four
    faces (two from each closed box).  This isolates a non-manifold
    *junction* without any duplicated or face-to-face glued surfaces.

    A correct repair must separate the sheets without deleting a single
    triangle; area, face count and each box's volume must be preserved.
    """
    va, fa = _box(0, 0, 0, 10, 10, 10)
    vb, fb = _box(10, 10, 0, 10, 10, 10)
    mesh = Mesh(np.vstack([va, vb]), np.vstack([fa, fb + 8]),
                name="nonmanifold_junction")
    snap = _weld_coincident(mesh)
    inc = build_edge_incidence(snap.faces)
    nm = len(inc.nonmanifold_edges())
    return snap, {
        "expected": {
            "nonmanifold_edges": nm,
            "faces": snap.face_count,
            "components_before": 1,
            "components_after_split": 2,
        },
        "notes": "两个封闭盒仅沿一条棱相交，形成非流形边；拆分必须保面、保面积与各盒体积",
    }


def _weld_coincident(mesh: Mesh, ndigits: int = 6) -> Mesh:
    """Merge vertices at identical rounded coordinates (benchmark fixture only)."""
    key_to_new: dict[tuple, int] = {}
    remap = np.full(mesh.vertex_count, -1, dtype=np.int64)
    verts: list[tuple] = []
    for i, p in enumerate(mesh.vertices):
        key = tuple(round(float(x), ndigits) for x in p)
        if key not in key_to_new:
            key_to_new[key] = len(verts)
            verts.append(key)
        remap[i] = key_to_new[key]
    faces = remap[mesh.faces]
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    return Mesh(np.array(verts, dtype=np.float64), faces[keep], name=mesh.name)


def combined_stress() -> tuple[Mesh, dict]:
    wall, _ = thin_wall(distance=0.3, size=25.0)
    v_b, f_b = _box(100, 0, 0, 16, 16, 8)
    f_b = np.vstack([f_b, f_b[:2]])       # duplicate faces
    f_b[0] = f_b[0][[0, 2, 1]]            # one reversed face
    v1, f1 = _box(200, 0, 0, 10, 10, 10, inward=True)
    f_b = f_b + len(wall.vertices)
    f1 = f1 + len(wall.vertices) + len(v_b)
    v = np.vstack([wall.vertices, v_b, v1])
    f = np.vstack([wall.faces, f_b, f1])
    return Mesh(v, f, name="combined_stress"), {
        "expected": {"components": 4, "duplicate_faces": 2, "inward_components": 1},
        "notes": "综合压力样本：薄壁 + 重复面 + 反向面 + 整体朝内组件",
    }


GENERATORS = {
    "thin_wall": thin_wall,
    "duplicate_faces": duplicate_faces,
    "holed_plate": holed_plate,
    "mixed_orientation": mixed_orientation,
    "open_shell": open_shell,
    "combined_stress": combined_stress,
    "multi_hole": multi_hole,
    "nonmanifold_junction": nonmanifold_junction,
}


def build(name: str, **kwargs) -> tuple[Mesh, dict]:
    return GENERATORS[name](**kwargs)


def write_all(directory) -> list:
    from pathlib import Path
    from ..core import io_mesh

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in GENERATORS:
        mesh, _meta = build(name)
        path = out / f"{name}.stl"
        io_mesh.save_mesh(mesh, path)
        written.append(path)
    return written
