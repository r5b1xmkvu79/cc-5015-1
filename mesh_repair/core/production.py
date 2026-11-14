"""Production repair junction: defect instance -> kernel -> before/after.

The GUI, the CLI helper, and the strategy experiment all go through this
module.  It exists so that "select a bad face in the viewport, run repair,
compare the mesh" has an executable core rather than a text description.

Every call:

1. validates the chosen defect instance against the *current* mesh,
2. runs exactly one kernel operation (never a whole-mesh sweep),
3. measures topology invariants and attributes before/after,
4. returns a structured result the UI can render and the audit can store.

Failures raise :class:`RepairError` *before* mutating anything; the caller
gets the same mesh hash it started with.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import holes, repair
from .invariants import attribute_delta, measure, mesh_hash
from .mesh import Mesh
from .repair import RepairError, RepairSession
from .topology import analyze


@dataclass
class RepairOutcome:
    kind: str
    index: int
    op: str
    applied: bool
    message: str
    record: dict | None = None
    before: dict | None = None
    after: dict | None = None
    delta: dict | None = None
    attributes: dict | None = None
    mesh_changed: bool = False
    hash_before: str = ""
    hash_after: str = ""
    # The face ids the user should look at afterwards (highlighted on canvas)
    focus_faces: list[int] = field(default_factory=list)
    focus_vertices: list[int] = field(default_factory=list)
    focus_loops: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "index": self.index,
            "op": self.op,
            "applied": self.applied,
            "message": self.message,
            "record": self.record,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "attributes": self.attributes,
            "mesh_changed": self.mesh_changed,
            "hash_before": self.hash_before,
            "hash_after": self.hash_after,
            "focus_faces": list(self.focus_faces),
            "focus_vertices": list(self.focus_vertices),
            "focus_loops": [list(x) for x in self.focus_loops],
        }


PRODUCTION_KINDS = (
    "duplicate_faces",
    "degenerate_faces",
    "nonmanifold_edges",
    "winding_conflicts",
    "inward_components",
    "duplicate_vertices",
    "open_boundaries",
    "unreferenced_vertices",
)


def locate_instance(result, kind: str, inst: dict) -> int | None:
    """Return the live index of ``inst`` inside ``result`` for ``kind``."""
    group = result.groups.get(kind)
    if group is None:
        return None
    for i, candidate in enumerate(group.instances):
        if candidate is inst:
            return i
    # Instances are rebuilt between analyses, so fall back to content match.
    key = (kind, tuple(sorted(int(v) for v in inst.get("vertices", []))),
           tuple(sorted(int(f) for f in inst.get("faces", []))))
    for i, candidate in enumerate(group.instances):
        ckey = (kind, tuple(sorted(int(v) for v in candidate.get("vertices", []))),
                tuple(sorted(int(f) for f in candidate.get("faces", []))))
        if ckey == key:
            return i
    return None


def find_defect_for_faces(result, faces: list[int]) -> tuple[str, int] | None:
    """Map a canvas face selection to the defect instance that contains it.

    Used by the "pick a bad face, run repair" path.  Preference goes to the
    instance covering the most selected faces; ties break by defect kind
    order so the result is deterministic.
    """
    wanted = {int(f) for f in faces}
    if not wanted:
        return None
    order = {k: i for i, k in enumerate(PRODUCTION_KINDS)}
    best: tuple[int, int, str, int] | None = None
    for kind, group in result.groups.items():
        if kind not in order:
            continue
        for idx, inst in enumerate(group.instances):
            hit = wanted & {int(f) for f in inst.get("faces", [])}
            if not hit:
                continue
            rank = (len(hit), -order[kind], kind, idx)
            if best is None or rank > best:
                best = rank
    return (best[2], best[3]) if best else None


def apply_from_selection(
    session: RepairSession,
    result,
    *,
    faces: list[int] | None = None,
    kind: str | None = None,
    index: int | None = None,
    reason: str = "",
    limits: holes.HoleFillLimits | None = None,
) -> RepairOutcome:
    """Entry point for the GUI: explicit defect ref, or a canvas face pick."""
    if kind is not None and index is not None:
        return apply_to_instance(session, kind, index, reason=reason, limits=limits)
    found = find_defect_for_faces(result, faces or [])
    if found is None:
        raise RepairError(
            "选中的面不属于任何可自动修复的缺陷实例；请先在缺陷列表中选择，"
            "或确认该面确实是重复/退化/绕向/非流形缺陷的一部分"
        )
    return apply_to_instance(session, found[0], found[1], reason=reason, limits=limits)


def _delta(before, after) -> dict:
    a = measure(before).to_dict()
    b = measure(after).to_dict()
    keys = [
        "vertices", "edges", "faces", "components", "boundary_edges",
        "boundary_loops", "nonmanifold_edges", "nonmanifold_vertices",
        "euler_characteristic", "total_genus", "area", "signed_volume",
    ]
    out = {}
    for k in keys:
        out[k] = {"before": a[k], "after": b[k],
                  "delta": (b[k] - a[k]) if isinstance(a[k], (int, float)) else None}
    return out


def _current_instance(session: RepairSession, kind: str, index: int) -> dict:
    """Re-detect and return the instance at ``index`` of ``kind`` now.

    Detection indices move after earlier repairs, so the UI must pass the
    current analysis index.  We re-analyze on the live mesh and refuse a
    stale index rather than repairing the wrong entity.
    """
    result = analyze(session.mesh, unit=session.unit, dup_tolerance=0.01,
                     degenerate_edge=0.001)
    group = result.groups.get(kind)
    if group is None or index < 0 or index >= len(group.instances):
        raise RepairError(
            f"缺陷 {kind} 的第 {index} 个实例在当前网格中已不存在；请重新检测后再选"
        )
    return group.instances[index]


def _preview_union(mesh: Mesh, face_ids: list[int]) -> list[int]:
    """Faces adjacent across an edge to the selection, for viewport context."""
    from .topology import build_edge_incidence

    inc = build_edge_incidence(mesh.faces)
    out = set(int(f) for f in face_ids)
    for f in face_ids:
        for eid in inc.face_edges[int(f)]:
            out.update(int(x) for x in inc.faces[eid])
    return sorted(out)


def apply_to_instance(
    session: RepairSession,
    kind: str,
    index: int,
    *,
    reason: str = "",
    limits: holes.HoleFillLimits | None = None,
) -> RepairOutcome:
    """Run the kernel operation appropriate for one selected defect instance."""
    if kind not in PRODUCTION_KINDS:
        raise RepairError(f"缺陷类型 {kind} 没有生产修复入口")
    inst = _current_instance(session, kind, index)
    mesh_before = session.mesh.copy()
    hash_before = mesh_hash(session.mesh)
    limits = limits or holes.HoleFillLimits()
    rec = None
    op = ""
    message = ""
    default_reason = reason or f"用户选择 {kind}#{index}"

    try:
        if kind == "duplicate_faces":
            triple = tuple(sorted(int(v) for v in inst.get("vertices", [])))
            if len(triple) != 3:
                raise RepairError("重复面实例缺少顶点三元组")
            recs = repair.remove_duplicate_faces(session, [inst])
            if not recs:
                raise RepairError("该重复面组在当前网格中已无冗余副本")
            rec = recs[0]
            op = "remove_duplicate_faces"
            message = rec.label

        elif kind == "degenerate_faces":
            faces = [int(f) for f in inst.get("faces", [])]
            rec = repair.delete_faces(session, faces, reason=default_reason)
            op = "delete_faces"
            message = f"删除退化面 {len(faces)} 个；此操作可撤销"

        elif kind == "nonmanifold_edges":
            vs = [int(v) for v in inst.get("vertices", [])]
            if len(vs) != 2:
                raise RepairError("非流形边实例缺少两个端点")
            # Separating the endpoint fans first often makes the edge itself
            # manifold; only then can it be addressed by its two endpoints.
            recs = repair.split_nonmanifold_vertex_fans(
                session, vs, reason=default_reason
            )
            if not recs:
                rec = repair.split_nonmanifold_edge(
                    session, vs[0], vs[1], reason=default_reason,
                    confirm_reason=default_reason,
                )
                recs = [rec]
            rec = recs[-1]
            op = "split_nonmanifold_edge"
            message = f"拆分非流形交汇 {len(recs)} 步；无面被删除"

        elif kind == "winding_conflicts":
            faces = [int(f) for f in inst.get("faces", [])]
            rec = repair.flip_faces(session, faces, reason=default_reason)
            op = "flip_faces"
            message = rec.label

        elif kind == "inward_components":
            faces = [int(f) for f in inst.get("faces", [])]
            rec = repair.flip_component(session, faces, confirm_reason=default_reason)
            op = "flip_component"
            message = rec.label

        elif kind == "duplicate_vertices":
            group = [int(v) for v in inst.get("vertices", [])]
            spread = float(inst.get("max_spread", 0.0))
            threshold = spread if spread > 0 else 0.0
            recs = repair.weld_vertices_selected(
                session, [group], unit=session.unit, threshold=threshold,
                confirmed_reason=default_reason,
            )
            if not recs:
                raise RepairError("顶点组不足两个，未执行焊接")
            rec = recs[0]
            op = "weld_vertices"
            message = rec.label

        elif kind == "open_boundaries":
            vs = [int(v) for v in inst.get("vertices", [])]
            plan = holes.plan_fill(session, vs, limits)
            if not plan.get("ok"):
                why = plan.get("message") or plan.get("check", {}).get("message", "超限")
                raise RepairError(why)
            rec = holes.fill_hole(session, vs, limits, reason=default_reason)
            op = "fill_hole"
            message = rec.label

        elif kind == "unreferenced_vertices":
            vs = [int(v) for v in inst.get("vertices", [])]
            rec = repair.remove_isolated_vertices(session, vs)
            op = "remove_isolated_vertices"
            message = rec.label

    except RepairError:
        # Nothing was applied: _apply validates before assigning.  Prove it.
        if mesh_hash(session.mesh) != hash_before:
            raise RepairError("内部错误：失败的修复改变了网格，请联系维护者")
        raise

    after = analyze(session.mesh, unit=session.unit, dup_tolerance=0.01,
                    degenerate_edge=0.001)
    outcome = RepairOutcome(
        kind=kind, index=index, op=op, applied=True, message=message,
        record=rec.to_dict() if rec else None,
        before=measure(mesh_before).to_dict(),
        after=measure(session.mesh).to_dict(),
        delta=_delta(mesh_before, session.mesh),
        attributes=attribute_delta(mesh_before, session.mesh),
        mesh_changed=mesh_hash(session.mesh) != hash_before,
        hash_before=hash_before,
        hash_after=mesh_hash(session.mesh),
    )

    # Map post-repair focus back to live indices.  Added faces survive at the
    # tail of the face array, and the new defect position is reported below.
    if kind == "open_boundaries":
        outcome.focus_faces = sorted(set(
            int(f) for i in after.groups.get("open_boundaries", type(
                "g", (), {"instances": []})()).instances
            for f in i.get("faces", [])
        ))
        # the freshly filled region: tail faces recorded by the operation
        if rec is not None:
            outcome.focus_faces = [int(f) for f in rec.added_faces] + outcome.focus_faces
    else:
        faces = [int(f) for f in inst.get("faces", [])]
        if rec is not None:
            faces = [f for f in faces if f < session.mesh.face_count]
        outcome.focus_faces = _preview_union(session.mesh, faces) if faces else []
    outcome.focus_vertices = [int(v) for v in inst.get("vertices", [])
                              if int(v) < session.mesh.vertex_count]
    if kind == "open_boundaries" and rec is not None:
        outcome.focus_loops = [[int(v) for v in inst.get("vertices", [])]]
    return outcome
