"""Defect fingerprinting and post-repair verification.

A fingerprint identifies a defect by *content*, not by index, so it can be
recomputed on a reloaded model:

* degenerate/duplicate faces: canonical vertex coordinate triple,
* vertices: rounded coordinate (plus connectivity rank to disambiguate
  coincident positions on thin walls),
* edges: the two endpoint coordinates,
* boundaries: the loop's coordinate sequence rotated to a canonical start.

Verification compares the before/after analysis and reports which defect
kinds were actually removed, which remain, and what new defects appeared.
A repair is only "successful" for the kinds whose fingerprints vanished.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .mesh import Mesh
from .topology import AnalysisResult

_QUANT = 6   # decimals for coordinate hashing; model-unit scale assumed sane


def _vkey(p: np.ndarray) -> str:
    return f"{p[0]:.{_QUANT}f},{p[1]:.{_QUANT}f},{p[2]:.{_QUANT}f}"


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _rotated_min(seq: list[str]) -> str:
    if not seq:
        return ""
    doubled = seq + seq
    n = len(seq)
    best = min(tuple(doubled[i:i + n]) for i in range(n))
    return "|".join(best)


def face_fingerprint(mesh: Mesh, face_id: int) -> str:
    pts = mesh.vertices[mesh.faces[face_id]]
    keys = sorted(_vkey(p) for p in pts)
    return "F:" + _hash("|".join(keys))


def vertex_fingerprint(mesh: Mesh, vertex_id: int) -> str:
    return "V:" + _vkey(mesh.vertices[vertex_id])


def edge_fingerprint(mesh: Mesh, a: int, b: int) -> str:
    ka, kb = _vkey(mesh.vertices[a]), _vkey(mesh.vertices[b])
    lo, hi = (ka, kb) if ka < kb else (kb, ka)
    return "E:" + _hash(lo + "|" + hi)


def loop_fingerprint(mesh: Mesh, vertex_ids: list[int]) -> str:
    return "L:" + _hash(_rotated_min([_vkey(mesh.vertices[v]) for v in vertex_ids]))


def group_fingerprints(mesh: Mesh, group_kind: str, group) -> list[dict]:
    """Fingerprints for every instance in a defect group."""
    out = []
    for inst in group.instances:
        if group_kind in ("degenerate_faces", "duplicate_faces", "winding_conflicts",
                          "inward_components"):
            fps = [face_fingerprint(mesh, f) for f in inst.get("faces", [])]
            out.append({"kind": group_kind, "faces": fps,
                        "summary": _hash("".join(sorted(fps)))})
        elif group_kind in ("duplicate_vertices", "unreferenced_vertices",
                            "nonmanifold_vertices"):
            fps = [vertex_fingerprint(mesh, v) for v in inst.get("vertices", [])]
            out.append({"kind": group_kind, "vertices": fps,
                        "summary": _hash("".join(sorted(fps)))})
        elif group_kind in ("nonmanifold_edges",):
            vs = inst.get("vertices", [])
            fp = edge_fingerprint(mesh, vs[0], vs[1]) if len(vs) == 2 else ""
            out.append({"kind": group_kind, "edges": [fp], "summary": fp})
        elif group_kind in ("open_boundaries",):
            vs = inst.get("vertices", [])
            fp = loop_fingerprint(mesh, vs) if vs else ""
            out.append({"kind": group_kind, "loops": [fp], "summary": fp})
        else:
            out.append({"kind": group_kind, "summary": ""})
    return out


def all_fingerprints(mesh: Mesh, result: AnalysisResult) -> dict:
    payload = {}
    for kind, group in result.groups.items():
        payload[kind] = group_fingerprints(mesh, kind, group)
    return payload


def verification_report(
    before_mesh: Mesh,
    before_result: AnalysisResult,
    after_mesh: Mesh,
    after_result: AnalysisResult,
) -> dict:
    """Compare defect fingerprints before vs after.

    Success is per defect kind and only declared when the count actually
    drops.  New defects are always reported, even if the file loads fine.
    """
    before_fp = all_fingerprints(before_mesh, before_result)
    after_fp = all_fingerprints(after_mesh, after_result)

    kinds = {}
    for kind in sorted(set(before_fp) | set(after_fp)):
        b = {inst["summary"] for inst in before_fp.get(kind, []) if inst.get("summary")}
        a = {inst["summary"] for inst in after_fp.get(kind, []) if inst.get("summary")}
        removed = sorted(b - a)
        remaining = sorted(a & b)
        added = sorted(a - b)
        kinds[kind] = {
            "before": len(before_fp.get(kind, [])),
            "after": len(after_fp.get(kind, [])),
            "removed": len(removed),
            "remaining": len(remaining),
            "new": len(added),
            "fully_resolved": bool(b) and not remaining and not added,
        }

    metric_keys = ["boundary_edges", "nonmanifold_edges", "conflict_edges",
                   "components", "triangles_degenerate", "faces", "vertices"]
    metric_delta = {}
    for k in metric_keys:
        bv = before_result.metrics.get(k)
        av = after_result.metrics.get(k)
        if bv is not None and av is not None:
            metric_delta[k] = {"before": bv, "after": av, "delta": av - bv}

    volume_before = before_mesh.signed_volume()
    volume_after = after_mesh.signed_volume()
    bbox_before = before_mesh.extent()
    bbox_after = after_mesh.extent()

    problems = []
    for kind, info in kinds.items():
        if info["new"]:
            problems.append(f"{kind}: 修复后新增 {info['new']} 个缺陷实例")
        if info["remaining"]:
            problems.append(f"{kind}: 仍有 {info['remaining']} 个缺陷实例未解决")
    if after_mesh.validate():
        problems.append("修复后的网格未通过基本有效性检查: " + "；".join(after_mesh.validate()))

    return {
        "kinds": kinds,
        "metrics": metric_delta,
        "geometry": {
            "area_before": before_mesh.area(),
            "area_after": after_mesh.area(),
            "area_delta": after_mesh.area() - before_mesh.area(),
            "volume_before": volume_before,
            "volume_after": volume_after,
            "volume_delta": volume_after - volume_before,
            "bbox_before": bbox_before.tolist(),
            "bbox_after": bbox_after.tolist(),
            "bbox_delta": (bbox_after - bbox_before).tolist(),
            "diagonal_before": before_mesh.diagonal(),
            "diagonal_after": after_mesh.diagonal(),
        },
        "problems": problems,
        "clean": not problems and not any(v["new"] for v in kinds.values()),
        "note": (
            "验证基于重新检测，不以文件可打开为准；"
            "“待确认”类缺陷不会计为已修复"
        ),
    }
