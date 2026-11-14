"""Topology and defect analysis.

Every detector returns indices into the *current* mesh so the UI can select
the exact entities and the audit log can name them.  Detection never
modifies geometry.

Design rules encoded here:

* Coincident vertex positions are reported, never welded.  Thin walls have
  legitimately coincident points on opposite surfaces.
* A component is only auto-flippable when it is closed (manifold-closed) so
  the signed volume is a reliable oracle.  Open shells are marked
  "unconfirmed" and the UI must ask.
* Boundary loops are reported with their perimeter and per-vertex spacing so
  the hole filler can decide whether the opening is a designed feature.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh import (
    Mesh,
    triangle_areas,
    triangle_aspect_ratio,
    triangle_edge_lengths,
)

# Edge key packs two vertex ids into one int64.  Vertex ids above 2^31 are
# rejected at load time, which is far beyond any practical mesh.
_SHIFT = np.int64(1) << np.int64(32)
_MAX_INDEX = int(_SHIFT) - 1


def _pack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a.astype(np.int64) * _SHIFT + b.astype(np.int64)


# ---------------------------------------------------------------------------
# Edge incidence
# ---------------------------------------------------------------------------


@dataclass
class EdgeIncidence:
    keys: np.ndarray                # (E,) sorted unique undirected keys
    faces: list[list[int]]          # per edge, incident face ids
    half_edges: list[list[tuple[int, int]]]  # per edge, directed (a,b) per face
    endpoints: np.ndarray           # (E, 2) canonical (lo, hi)
    face_edges: list[list[int]] = field(default_factory=list)  # per face, edge ids

    def counts(self) -> np.ndarray:
        return np.array([len(f) for f in self.faces], dtype=np.int64)

    def boundary_edges(self) -> np.ndarray:
        return np.nonzero(self.counts() == 1)[0]

    def nonmanifold_edges(self) -> np.ndarray:
        return np.nonzero(self.counts() > 2)[0]


def build_edge_incidence(faces: np.ndarray) -> EdgeIncidence:
    if len(faces) == 0:
        return EdgeIncidence(
            np.zeros(0, dtype=np.int64), [], [], np.zeros((0, 2), dtype=np.int64), []
        )
    if faces.max() > _MAX_INDEX:
        raise ValueError("顶点索引超出可打包范围")

    nf = len(faces)
    a = faces.reshape(-1)                          # corner i   -> (i)
    b = np.roll(faces, -1, axis=1).reshape(-1)     # next corner
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    keys = _pack(lo, hi)
    face_ids = np.repeat(np.arange(nf, dtype=np.int64), 3)

    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    face_sorted = face_ids[order]
    a_sorted = a[order]
    b_sorted = b[order]

    unique, starts = np.unique(keys_sorted, return_index=True)
    ends = np.append(starts[1:], len(keys_sorted))
    grouped_faces: list[list[int]] = []
    grouped_half: list[list[tuple[int, int]]] = []
    for s, e in zip(starts, ends):
        grouped_faces.append(sorted(int(x) for x in face_sorted[s:e]))
        grouped_half.append([(int(a_sorted[k]), int(b_sorted[k])) for k in range(s, e)])
    endpoints = np.stack(
        [
            (unique // int(_SHIFT)).astype(np.int64),
            (unique % int(_SHIFT)).astype(np.int64),
        ],
        axis=1,
    )

    # per-face edge ids, used for linear closure tests and adjacency display
    edge_of_corner = np.full(len(keys_sorted), -1, dtype=np.int64)
    edge_of_corner[order] = np.repeat(np.arange(len(unique), dtype=np.int64), ends - starts)
    face_edges = edge_of_corner.reshape(nf, 3).tolist()
    return EdgeIncidence(unique, grouped_faces, grouped_half, endpoints,
                         [[int(e) for e in row] for row in face_edges])


def _find_edge(inc: EdgeIncidence, a: int, b: int) -> int | None:
    lo, hi = (a, b) if a < b else (b, a)
    key = np.int64(lo) * _SHIFT + np.int64(hi)
    idx = int(np.searchsorted(inc.keys, key))
    if idx < len(inc.keys) and int(inc.keys[idx]) == int(key):
        return idx
    return None


def boundary_loops(inc: EdgeIncidence) -> list[dict]:
    """Trace boundary loops using boundary half-edges.

    The half-edge direction comes from the single incident face.  For a
    manifold boundary each boundary vertex has exactly one outgoing and one
    incoming boundary half-edge.  Loops are reported closed only when the
    trace returns to its start; malformed chains are flagged so the filler
    refuses them.  Chains are traced from their free (incoming-only) ends so
    a broken boundary is reported as one chain, not fragments.
    """
    bnd = [int(e) for e in inc.boundary_edges()]
    if not bnd:
        return []
    nxt: dict[int, int] = {}
    incoming: dict[int, int] = {}
    for eid in bnd:
        a, b = inc.half_edges[eid][0]
        # duplicated outgoing means a non-manifold boundary vertex; keep first
        nxt.setdefault(a, b)
        incoming[b] = incoming.get(b, 0) + 1
    chain_starts = [v for v in nxt if incoming.get(v, 0) == 0]
    # Closed loops have no free start; seed them from every unvisited edge.
    ordered_starts = sorted(chain_starts) + sorted(nxt)
    visited_edges: set[tuple[int, int]] = set()
    loops: list[dict] = []
    for start in ordered_starts:
        if start not in nxt:
            continue
        if (start, nxt[start]) in visited_edges:
            continue
        loop = [start]
        cur = start
        closed = False
        guard = 0
        while guard <= len(bnd) + 1:
            guard += 1
            if cur not in nxt:
                break
            edge = (cur, nxt[cur])
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            nxt_v = nxt[cur]
            if nxt_v == start:
                closed = True
                break
            if nxt_v in loop:
                # self-touching chain; refuse to treat as a simple loop
                break
            loop.append(nxt_v)
            cur = nxt_v
        loops.append({"vertices": loop, "closed": closed})
    return loops


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------


class _DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def vertex_components(n_vertices: int, faces: np.ndarray) -> np.ndarray:
    ds = _DisjointSet(n_vertices)
    for tri in faces.tolist():
        ds.union(tri[0], tri[1])
        ds.union(tri[1], tri[2])
    roots = np.array([ds.find(i) for i in range(n_vertices)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def face_components(faces: np.ndarray, n_vertices: int) -> tuple[np.ndarray, int]:
    if len(faces) == 0:
        return np.zeros(0, dtype=np.int64), 0
    vlabels = vertex_components(n_vertices, faces)
    if np.any(faces < 0) or np.any(faces >= n_vertices):
        return np.zeros(len(faces), dtype=np.int64), 1
    _, labels = np.unique(vlabels[faces[:, 0]], return_inverse=True)
    return labels.astype(np.int64), int(labels.max()) + 1


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class DefectGroup:
    kind: str
    label: str
    instances: list[dict]
    severity: str = "warning"
    note: str = ""
    auto_repairable: bool = False

    def __len__(self) -> int:
        return len(self.instances)

    def summary(self) -> str:
        return f"{self.label} {len(self.instances)}"


@dataclass
class AnalysisResult:
    groups: dict[str, DefectGroup] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def add(self, group: DefectGroup) -> None:
        if len(group):
            self.groups[group.kind] = group

    def total_instances(self) -> int:
        return sum(len(g) for g in self.groups.values())

    def kinds(self) -> list[str]:
        return list(self.groups)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def find_degenerate_faces(
    mesh: Mesh,
    area_eps: float,
    aspect_max: float = 1e4,
    sliver_ratio: float = 0.01,
) -> list[dict]:
    if mesh.face_count == 0:
        return []
    areas = triangle_areas(mesh.vertices, mesh.faces)
    edges = triangle_edge_lengths(mesh.vertices, mesh.faces)
    aspect = triangle_aspect_ratio(mesh.vertices, mesh.faces)
    repeated = (
        (mesh.faces[:, 0] == mesh.faces[:, 1])
        | (mesh.faces[:, 1] == mesh.faces[:, 2])
        | (mesh.faces[:, 0] == mesh.faces[:, 2])
    )
    longest = edges.max(axis=1)
    shortest = edges.min(axis=1)
    needle = np.divide(shortest, longest, out=np.zeros_like(longest), where=longest > 0)
    area_cut = area_eps * area_eps * 0.5
    mask = repeated | (areas <= area_cut) | (needle < sliver_ratio) | (aspect > aspect_max)
    out = []
    for fid in np.nonzero(mask)[0]:
        fid = int(fid)
        reasons = []
        if repeated[fid]:
            reasons.append("顶点索引重复")
        if areas[fid] <= area_cut:
            reasons.append(f"面积≈{areas[fid]:.3g}")
        if longest[fid] > 0 and needle[fid] < sliver_ratio:
            reasons.append(f"最短/最长边 {needle[fid]:.2e}")
        if np.isfinite(aspect[fid]) and aspect[fid] > aspect_max:
            reasons.append(f"长宽比 {aspect[fid]:.0f}")
        out.append(
            {
                "faces": [fid],
                "vertices": sorted({int(v) for v in mesh.faces[fid]}),
                "reason": "；".join(reasons) or "退化",
                "area": float(areas[fid]),
            }
        )
    return out


def find_duplicate_faces(mesh: Mesh) -> list[dict]:
    if mesh.face_count == 0:
        return []
    canon = np.sort(mesh.faces, axis=1)
    view: dict[tuple[int, int, int], list[int]] = {}
    for fid in range(mesh.face_count):
        key = tuple(int(x) for x in canon[fid])
        view.setdefault(key, []).append(fid)
    groups = []
    for tri, ids in view.items():
        if len(ids) < 2:
            continue
        windings = {(int(mesh.faces[i, 0]), int(mesh.faces[i, 1]), int(mesh.faces[i, 2])) for i in ids}
        groups.append(
            {
                "faces": sorted(ids),
                "vertices": list(tri),
                "reason": "同向重复" if len(windings) == 1 else "反向重复",
                "same_winding": len(windings) == 1,
            }
        )
    return groups


def find_duplicate_vertices(mesh: Mesh, tol: float) -> list[dict]:
    if mesh.vertex_count == 0 or tol <= 0:
        return []
    from .spatial import GridIndex

    index = GridIndex(mesh.vertices, cell=tol)
    pairs = index.close_pairs(tol)
    if not pairs:
        return []
    ds = _DisjointSet(mesh.vertex_count)
    for a, b in pairs:
        ds.union(a, b)
    clusters: dict[int, set[int]] = {}
    for a, b in pairs:
        clusters.setdefault(ds.find(a), set()).update((a, b))
    out = []
    for _root, members in clusters.items():
        ids = sorted(members)
        p0 = mesh.vertices[ids[0]]
        spread = float(np.linalg.norm(mesh.vertices[ids] - p0, axis=1).max())
        out.append(
            {
                "vertices": ids,
                "faces": [],
                "reason": f"{len(ids)} 个顶点间距 ≤ {tol:.4g}：最大 {spread:.3g}",
                "max_spread": spread,
                "exact": bool(spread == 0.0),
            }
        )
    return out


def find_nonmanifold_edges(mesh: Mesh, inc: EdgeIncidence) -> list[dict]:
    out = []
    for eid in inc.nonmanifold_edges():
        eid = int(eid)
        fs = sorted(set(inc.faces[eid]))
        if len(fs) < 3:
            continue
        out.append(
            {
                "edges": [eid],
                "vertices": [int(x) for x in inc.endpoints[eid]],
                "faces": fs,
                "reason": f"{len(fs)} 个面共享该边",
            }
        )
    return out


def find_nonmanifold_vertices(mesh: Mesh, inc: EdgeIncidence) -> list[dict]:
    """Bow-tie vertices: incident faces split into disjoint fans.

    Only edges incident to the vertex that are *manifold* (exactly 2 faces)
    join the faces into one fan; boundary edges do not connect fans.
    """
    if mesh.face_count == 0:
        return []
    vert_faces: dict[int, list[int]] = {}
    for fid in range(mesh.face_count):
        unique_vs = set(int(v) for v in mesh.faces[fid])
        for v in unique_vs:
            vert_faces.setdefault(v, []).append(fid)

    # vertex -> incident manifold edges with exactly two faces
    vert_edges: dict[int, list[tuple[int, int, int]]] = {}
    for eid in range(len(inc.keys)):
        fs = inc.faces[eid]
        if len(fs) != 2:
            continue
        a, b = int(inc.endpoints[eid, 0]), int(inc.endpoints[eid, 1])
        vert_edges.setdefault(a, []).append((eid, fs[0], fs[1]))
        vert_edges.setdefault(b, []).append((eid, fs[0], fs[1]))

    out = []
    for v, flist in vert_faces.items():
        if len(flist) < 2:
            continue
        ds = _DisjointSet(len(flist))
        pos = {f: i for i, f in enumerate(flist)}
        for _eid, f1, f2 in vert_edges.get(v, []):
            if f1 in pos and f2 in pos:
                ds.union(pos[f1], pos[f2])
        if len({ds.find(i) for i in range(len(flist))}) > 1:
            out.append(
                {
                    "vertices": [int(v)],
                    "faces": sorted(flist),
                    "reason": f"{len(flist)} 个面分成多个不连通面扇",
                }
            )
    return out


def find_open_boundaries(mesh: Mesh, inc: EdgeIncidence) -> list[dict]:
    out = []
    for loop in boundary_loops(inc):
        vs = loop["vertices"]
        if len(vs) < 3:
            continue
        pts = mesh.vertices[vs]
        # loop may not repeat the start vertex; close it for measurement
        closed_pts = pts
        seg = np.diff(np.vstack([closed_pts, closed_pts[:1]]), axis=0)
        perimeter = float(np.linalg.norm(seg, axis=1).sum())
        spacings = np.linalg.norm(seg, axis=1)
        out.append(
            {
                "vertices": [int(v) for v in vs],
                "faces": [],
                "edges": [],
                "closed": bool(loop["closed"]),
                "reason": (
                    f"边界环 {len(vs)} 顶点，周长 {perimeter:.4g}"
                    if loop["closed"]
                    else f"开放边界链 {len(vs)} 顶点（未闭合，不可自动补洞）"
                ),
                "perimeter": perimeter,
                "max_spacing": float(spacings.max()) if len(spacings) else 0.0,
                "mean_spacing": float(spacings.mean()) if len(spacings) else 0.0,
                "points": pts,
            }
        )
    return out


def find_unreferenced_vertices(mesh: Mesh) -> list[dict]:
    used = np.zeros(mesh.vertex_count, dtype=bool)
    if mesh.face_count:
        used[mesh.used_vertices()] = True
    dangling = np.nonzero(~used)[0]
    if len(dangling) == 0:
        return []
    return [
        {
            "vertices": [int(v) for v in dangling],
            "faces": [],
            "reason": f"{len(dangling)} 个顶点未被任何面引用",
        }
    ]


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


def winding_conflicts(mesh: Mesh, inc: EdgeIncidence) -> tuple[list[int], list[int], dict]:
    """Find a small set of faces to flip so shared edges are opposite.

    Each shared edge constrains its two faces to opposite winding.  BFS
    assigns a flip parity per connected face shell; within each shell the
    complement (flip everything instead of nothing) is chosen when it yields
    fewer flips.  Contradictory parities mean the surface is non-orientable
    at that edge, which is reported and never auto-flipped.

    Returns (conflict_edges, faces_to_flip, diagnostics).
    """
    nf = mesh.face_count
    parity = np.full(nf, -1, dtype=np.int8)
    adjacency: list[list[tuple[int, bool, int]]] = [[] for _ in range(nf)]
    conflict_edges: list[int] = []
    for eid, halves in enumerate(inc.half_edges):
        fs = inc.faces[eid]
        if len(fs) != 2 or fs[0] == fs[1]:
            continue
        f1, f2 = fs
        same_direction = halves[0] == halves[1]
        if same_direction:
            conflict_edges.append(eid)
        adjacency[f1].append((f2, same_direction, eid))
        adjacency[f2].append((f1, same_direction, eid))

    non_orientable: set[int] = set()
    shells: list[list[int]] = []
    for seed in range(nf):
        if parity[seed] != -1:
            continue
        parity[seed] = 0
        stack = [seed]
        shell: list[int] = []
        while stack:
            f = stack.pop()
            shell.append(f)
            for g, same_direction, eid in adjacency[f]:
                wanted = parity[f] ^ (1 if same_direction else 0)
                if parity[g] == -1:
                    parity[g] = wanted
                    stack.append(g)
                elif parity[g] != wanted:
                    non_orientable.add(f)
                    non_orientable.add(g)
        shells.append(shell)

    flip_faces: list[int] = []
    for shell in shells:
        ones = [f for f in shell if parity[f] == 1]
        zeros = [f for f in shell if parity[f] == 0]
        flip_faces.extend(ones if len(ones) <= len(zeros) else zeros)
    flip_faces.sort()
    return sorted(set(conflict_edges)), flip_faces, {
        "consistent": not conflict_edges,
        "non_orientable_faces": sorted(non_orientable),
    }


def orientation_report(mesh: Mesh, inc: EdgeIncidence) -> dict:
    """Winding consistency plus per-component inside/outside classification.

    Winding parities come from :func:`winding_conflicts`; each fully closed
    component then gets an independent volume test.  Open components keep
    ``outward = None`` because volume sign is meaningless without a lid.
    """
    result = {
        "conflict_edges": [],
        "conflict_faces": [],
        "components": [],
        "inward_components": [],
        "ambiguous_components": [],
        "open_components": [],
        "consistent": True,
        "non_orientable_faces": [],
        "boundary_edge_count": int(len(inc.boundary_edges())),
        "nonmanifold_edge_count": int(len(inc.nonmanifold_edges())),
    }
    if mesh.face_count == 0:
        return result

    edges, flip_faces, diag = winding_conflicts(mesh, inc)
    result["conflict_edges"] = edges
    result["conflict_faces"] = flip_faces
    result["consistent"] = diag["consistent"]
    result["non_orientable_faces"] = diag["non_orientable_faces"]

    labels, ncomp = face_components(mesh.faces, mesh.vertex_count)
    for comp in range(ncomp):
        fid_arr = np.nonzero(labels == comp)[0]
        if len(fid_arr) == 0:
            continue
        sub_faces = mesh.faces[fid_arr]
        v = mesh.vertices[sub_faces]
        volume = float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)
        closed = _faces_closed(inc, fid_arr)
        entry = {
            "id": comp,
            "faces": [int(f) for f in fid_arr],
            "face_count": int(len(fid_arr)),
            "signed_volume": volume,
            "closed": bool(closed),
            "outward": bool(volume > 0) if closed else None,
            "area": float(triangle_areas(mesh.vertices, sub_faces).sum()),
        }
        result["components"].append(entry)
        if closed:
            if volume < 0:
                result["inward_components"].append(comp)
        else:
            result["open_components"].append(comp)
            result["ambiguous_components"].append(comp)
    return result


def _faces_closed(inc: EdgeIncidence, face_ids: np.ndarray) -> bool:
    """Linear-time closure test for one connected face component.

    Every edge of every member face must have exactly two incident faces and
    both must be inside the component.  A non-manifold edge inside the
    component fails this naturally; defects in *other* components do not
    affect whether this component encloses a volume, so no global veto.
    """
    members = set(int(x) for x in face_ids)
    for f in face_ids:
        for eid in inc.face_edges[int(f)]:
            fs = inc.faces[eid]
            if len(fs) != 2:
                return False
            if fs[0] not in members or fs[1] not in members:
                return False
    return True


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def analyze(
    mesh: Mesh,
    *,
    unit: str,
    dup_tolerance: float,
    degenerate_edge: float,
    aspect_max: float = 1e4,
) -> AnalysisResult:
    res = AnalysisResult()
    mesh_errors = mesh.validate()
    if mesh_errors:
        res.warnings.extend(mesh_errors)
        res.metrics = {
            "vertices": mesh.vertex_count,
            "faces": mesh.face_count,
            "unit": unit,
            "invalid": True,
            "errors": mesh_errors,
        }
        # An out-of-range face index corrupts every topological query, so
        # report the invalidity and stop rather than emitting nonsense.
        return res
    inc = build_edge_incidence(mesh.faces)
    orient = orientation_report(mesh, inc)
    labels, ncomp = face_components(mesh.faces, mesh.vertex_count)

    res.metrics = {
        "vertices": mesh.vertex_count,
        "faces": mesh.face_count,
        "triangles_degenerate": 0,
        "boundary_edges": orient["boundary_edge_count"],
        "nonmanifold_edges": orient["nonmanifold_edge_count"],
        "conflict_edges": len(orient["conflict_edges"]),
        "components": ncomp,
        "unit": unit,
        "dup_tolerance": dup_tolerance,
        "degenerate_edge": degenerate_edge,
        "area": mesh.area(),
        "signed_volume": mesh.signed_volume(),
        "diagonal": mesh.diagonal(),
        "bounding_box": mesh.extent().tolist(),
    }

    degen = find_degenerate_faces(mesh, degenerate_edge, aspect_max=aspect_max)
    res.metrics["triangles_degenerate"] = len(degen)
    res.add(
        DefectGroup(
            "degenerate_faces",
            "退化面",
            degen,
            severity="error",
            note="零面积、重复索引或极度狭长；可删除被明确选中的面",
            auto_repairable=True,
        )
    )
    dupes_faces = find_duplicate_faces(mesh)
    res.add(
        DefectGroup(
            "duplicate_faces",
            "重复面",
            dupes_faces,
            severity="error",
            note="顶点组合相同的面；同向重复可删除多余者，反向重复需确认",
            auto_repairable=True,
        )
    )
    res.add(
        DefectGroup(
            "nonmanifold_edges",
            "非流形边",
            find_nonmanifold_edges(mesh, inc),
            severity="error",
            note="被三个及以上面共享；本工具不自动拆分，需人工定位",
        )
    )
    res.add(
        DefectGroup(
            "nonmanifold_vertices",
            "非流形顶点",
            find_nonmanifold_vertices(mesh, inc),
            severity="warning",
            note="面扇不连通的蝴蝶结顶点；不自动修复",
        )
    )
    dup_verts = find_duplicate_vertices(mesh, dup_tolerance)
    res.add(
        DefectGroup(
            "duplicate_vertices",
            "重合/邻近顶点",
            dup_verts,
            severity="warning",
            note="仅为报告。默认不焊接；薄壁两侧重合点必须结合单位与连通关系人工确认",
            auto_repairable=False,
        )
    )
    boundaries = find_open_boundaries(mesh, inc)
    res.add(
        DefectGroup(
            "open_boundaries",
            "开放边界",
            boundaries,
            severity="warning",
            note="设计开口与破损开口外观相同；补洞需人工选择且受尺寸/形状限制",
        )
    )
    res.add(
        DefectGroup(
            "unreferenced_vertices",
            "孤立顶点",
            find_unreferenced_vertices(mesh),
            severity="info",
            note="未被任何面引用，可在确认后移除",
            auto_repairable=True,
        )
    )

    conflicts = []
    if orient["conflict_faces"]:
        conflicts.append(
            {
                "faces": orient["conflict_faces"],
                "vertices": [],
                "reason": f"{len(orient['conflict_faces'])} 个面与相邻面绕向冲突",
            }
        )
    res.add(
        DefectGroup(
            "winding_conflicts",
            "绕向冲突",
            conflicts,
            severity="error",
            note="共享边被同向使用；可对选中面翻转绕向",
            auto_repairable=True,
        )
    )
    inward = []
    for comp in orient["components"]:
        if comp["closed"] and comp["outward"] is False:
            inward.append(
                {
                    "faces": comp["faces"],
                    "vertices": [],
                    "component": comp["id"],
                    "reason": f"封闭组件整体朝内（体积 {comp['signed_volume']:.4g}）",
                }
            )
    res.add(
        DefectGroup(
            "inward_components",
            "整体朝内组件",
            inward,
            severity="warning",
            note="仅对封闭组件用体积符号判断；翻转前仍需确认语义（内腔/模具可能本就朝内）",
        )
    )
    for comp in orient["components"]:
        if not comp["closed"]:
            res.warnings.append(
                f"组件 {comp['id']} 不封闭（{comp['face_count']} 面），"
                "无法用体积符号可靠判断内外方向，已标记为待确认"
            )
    if any(not d["exact"] for d in dup_verts):
        res.warnings.append(
            "存在非零间距的邻近顶点，可能属于相邻薄壁；默认不焊接，"
            "合并阈值需按模型单位与连通关系人工设定"
        )
    res.metrics["orientation"] = orient
    return res
