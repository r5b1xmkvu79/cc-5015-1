"""Recomputable topology invariants and attribute measurements.

Every number written by the strategy experiment comes from this module, so
the markdown tables and the JSON artifact can be regenerated from code
instead of being transcribed by hand.

Two kinds of evidence are kept separate:

* topology invariants (V, E, F, chi, components, boundary loops, genus,
  non-manifold counts, orientability) describe *what the surface is*;
* attributes (area, volume, bounding box, per-component volume, face key
  multiset) describe *what the asset carries*.  A repair that "fixes" the
  topology by deleting a difficult region shows up immediately as a changed
  face-key multiset and a shrunken area/volume.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .mesh import Mesh
from .topology import (
    boundary_loops,
    build_edge_incidence,
    face_components,
    find_nonmanifold_vertices,
    winding_conflicts,
)

_QUANT = 6


def _vkey(p) -> str:
    return f"{float(p[0]):.{_QUANT}f},{float(p[1]):.{_QUANT}f},{float(p[2]):.{_QUANT}f}"


def face_key(mesh: Mesh, face_id: int) -> str:
    """Canonical coordinate key, independent of vertex indices."""
    return "|".join(sorted(_vkey(p) for p in mesh.vertices[mesh.faces[face_id]]))


def face_key_multiset(mesh: Mesh) -> dict[str, int]:
    out: dict[str, int] = {}
    for fid in range(mesh.face_count):
        key = face_key(mesh, fid)
        out[key] = out.get(key, 0) + 1
    return out


def mesh_hash(mesh: Mesh) -> str:
    """Exact byte hash used to prove a failed operation changed nothing."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return h.hexdigest()


def _component_metrics(mesh: Mesh, inc, labels: np.ndarray, comp: int,
                       orientable: bool) -> dict:
    fid_arr = np.nonzero(labels == comp)[0]
    faces = mesh.faces[fid_arr]
    used = np.unique(faces)
    edge_ids = sorted({int(e) for f in fid_arr for e in inc.face_edges[int(f)]})
    # Count only edges whose both endpoints are in this component; an edge
    # shared with another component would otherwise inflate E_c.
    member_vertices = set(int(v) for v in used)
    own_edges = [
        e for e in edge_ids
        if int(inc.endpoints[e, 0]) in member_vertices
        and int(inc.endpoints[e, 1]) in member_vertices
    ]
    boundary_c = 0
    for e in own_edges:
        fs = [int(f) for f in inc.faces[e] if int(f) in set(int(x) for x in fid_arr)]
        if len(fs) == 1:
            boundary_c += 1
    v_c = len(member_vertices)
    e_c = len(own_edges)
    f_c = int(len(fid_arr))
    chi = v_c - e_c + f_c
    closed = boundary_c == 0
    genus = None
    if closed and orientable and chi % 2 == 0:
        genus = (2 - chi) // 2
    vol = float(np.einsum(
        "ij,ij->i",
        mesh.vertices[faces][:, 0],
        np.cross(mesh.vertices[faces][:, 1], mesh.vertices[faces][:, 2]),
    ).sum() / 6.0) if f_c else 0.0
    area = float(mesh.triangle_areas()[fid_arr].sum()) if f_c else 0.0
    return {
        "id": int(comp),
        "faces": f_c,
        "vertices": v_c,
        "edges": e_c,
        "chi": int(chi),
        "boundary_edges": int(boundary_c),
        "closed": bool(closed),
        "genus": genus,
        "area": area,
        "signed_volume": vol,
    }


@dataclass
class Invariants:
    vertices: int = 0
    edges: int = 0
    faces: int = 0
    components: int = 0
    boundary_edges: int = 0
    boundary_loops: int = 0
    closed_boundary_loops: int = 0
    nonmanifold_edges: int = 0
    nonmanifold_vertices: int = 0
    euler_characteristic: int = 0
    orientable: bool = True
    closed_components: int = 0
    total_genus: int = 0
    area: float = 0.0
    signed_volume: float = 0.0
    bbox_min: list[float] = field(default_factory=list)
    bbox_max: list[float] = field(default_factory=list)
    diagonal: float = 0.0
    per_component: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "vertices": self.vertices,
            "edges": self.edges,
            "faces": self.faces,
            "components": self.components,
            "boundary_edges": self.boundary_edges,
            "boundary_loops": self.boundary_loops,
            "closed_boundary_loops": self.closed_boundary_loops,
            "nonmanifold_edges": self.nonmanifold_edges,
            "nonmanifold_vertices": self.nonmanifold_vertices,
            "euler_characteristic": self.euler_characteristic,
            "orientable": self.orientable,
            "closed_components": self.closed_components,
            "total_genus": self.total_genus,
            "area": round(self.area, 9),
            "signed_volume": round(self.signed_volume, 9),
            "bbox_min": [round(float(x), 9) for x in self.bbox_min],
            "bbox_max": [round(float(x), 9) for x in self.bbox_max],
            "diagonal": round(self.diagonal, 9),
            "per_component": [
                {k: (round(v, 9) if isinstance(v, float) else v)
                 for k, v in c.items()}
                for c in self.per_component
            ],
        }


def measure(mesh: Mesh) -> Invariants:
    inc = build_edge_incidence(mesh.faces)
    used = mesh.used_vertices()
    v = int(used.size)
    e = int(len(inc.keys))
    f = int(mesh.face_count)
    labels, ncomp = face_components(mesh.faces, mesh.vertex_count)
    loops = boundary_loops(inc)
    closed_loops = sum(1 for lp in loops if lp["closed"])
    nm_edges = int(len(inc.nonmanifold_edges()))
    nm_verts = len(find_nonmanifold_vertices(mesh, inc))
    _, _flip, diag = winding_conflicts(mesh, inc)
    orientable = not diag.get("non_orientable_faces")
    b = mesh.bounds()
    comps = [_component_metrics(mesh, inc, labels, c, orientable) for c in range(ncomp)]
    closed_comps = [c for c in comps if c["closed"]]
    total_genus = sum(c["genus"] for c in closed_comps if c["genus"] is not None)
    return Invariants(
        vertices=v,
        edges=e,
        faces=f,
        components=int(ncomp),
        boundary_edges=int(len(inc.boundary_edges())),
        boundary_loops=len(loops),
        closed_boundary_loops=closed_loops,
        nonmanifold_edges=nm_edges,
        nonmanifold_vertices=nm_verts,
        euler_characteristic=v - e + f,
        orientable=orientable,
        closed_components=len(closed_comps),
        total_genus=int(total_genus),
        area=float(mesh.area()),
        signed_volume=float(mesh.signed_volume()),
        bbox_min=[float(x) for x in b[0]],
        bbox_max=[float(x) for x in b[1]],
        diagonal=float(mesh.diagonal()),
        per_component=comps,
    )


def invariant_delta(before: Invariants, after: Invariants) -> dict:
    keys = [
        "vertices", "edges", "faces", "components", "boundary_edges",
        "boundary_loops", "closed_boundary_loops", "nonmanifold_edges",
        "nonmanifold_vertices", "euler_characteristic", "closed_components",
        "total_genus",
    ]
    out = {}
    for k in keys:
        bv, av = getattr(before, k), getattr(after, k)
        out[k] = {"before": bv, "after": av, "delta": av - bv}
    for k in ("area", "signed_volume", "diagonal"):
        bv, av = getattr(before, k), getattr(after, k)
        out[k] = {"before": bv, "after": av, "delta": av - bv}
    return out


def attribute_delta(before: Mesh, after: Mesh) -> dict:
    """Compare attributes and prove which faces survived by coordinate key.

    Vertex indices are not stable across compaction, so the face key uses
    rounded coordinates.  ``preserved`` counts keys present in both, with
    multiplicity; a repair that deletes a hard region shrinks this number.
    """
    bk = face_key_multiset(before)
    ak = face_key_multiset(after)
    preserved = sum(min(n, ak.get(key, 0)) for key, n in bk.items())
    removed = sum(bk.values()) - preserved
    added = sum(ak.values()) - preserved
    ib, ia = measure(before), measure(after)
    return {
        "faces_before": before.face_count,
        "faces_after": after.face_count,
        "preserved_face_keys": int(preserved),
        "removed_face_keys": int(removed),
        "added_face_keys": int(added),
        "vertices_before": ib.vertices,
        "vertices_after": ia.vertices,
        "area_before": ib.area,
        "area_after": ia.area,
        "area_delta": ia.area - ib.area,
        "volume_before": ib.signed_volume,
        "volume_after": ia.signed_volume,
        "volume_delta": ia.signed_volume - ib.signed_volume,
        "bbox_before": [ib.bbox_max[i] - ib.bbox_min[i] for i in range(3)],
        "bbox_after": [ia.bbox_max[i] - ia.bbox_min[i] for i in range(3)],
        "diagonal_before": ib.diagonal,
        "diagonal_after": ia.diagonal,
        "components_before": ib.components,
        "components_after": ia.components,
    }
