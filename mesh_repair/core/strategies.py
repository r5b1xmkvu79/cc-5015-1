"""Two repair routes, side by side, on real meshes.

Route A --- local connection edits (局部连接):
    each defect instance is edited in place by a minimal connectivity
    change (delete duplicate face, split a non-manifold junction, fill one
    selected loop).  No global retriangulation, no "delete the hard
    region": every operation is recorded individually and can be undone.

Route B --- build a valid topology first, then patch (先构建有效拓扑再修补):
    the mesh is first reduced to a validated manifold scaffold.  The
    scaffold is a connectivity plan (a half-edge graph) whose junctions
    are split and whose loops are inventoried *before* any geometry is
    patched.  Patching then runs on the scaffold and is applied to the
    real mesh only when every loop in the plan has a bounded solution.

Both routes share one rule: a failed plan changes nothing.  Route B is
stricter --- it refuses to apply anything if the scaffold is not fully
manifold or if any loop exceeds its limits, whereas Route A applies the
local edits that passed and reports the rest.

The point of the experiment is *not* that one route "wins" globally.  It
is to show, with recomputable invariants and attribute deltas, where each
route trades duplication, gaps, and non-manifold junctions differently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import holes, repair
from .invariants import (
    Invariants,
    attribute_delta,
    face_key_multiset,
    invariant_delta,
    measure,
    mesh_hash,
)
from .mesh import Mesh
from .repair import RepairError, RepairSession
from .topology import analyze, build_edge_incidence


@dataclass
class StepResult:
    name: str
    applied: bool
    detail: str
    records: list[dict] = field(default_factory=list)


@dataclass
class RouteResult:
    route: str
    steps: list[StepResult] = field(default_factory=list)
    invariants_before: Invariants | None = None
    invariants_after: Invariants | None = None
    attributes: dict = field(default_factory=dict)
    mesh_changed: bool = False
    hash_before: str = ""
    hash_after: str = ""
    refused: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "route": self.route,
            "steps": [
                {"name": s.name, "applied": s.applied, "detail": s.detail,
                 "records": s.records}
                for s in self.steps
            ],
            "invariants_before": self.invariants_before.to_dict() if self.invariants_before else None,
            "invariants_after": self.invariants_after.to_dict() if self.invariants_after else None,
            "invariant_delta": invariant_delta(self.invariants_before, self.invariants_after)
            if self.invariants_before and self.invariants_after else None,
            "attributes": self.attributes,
            "mesh_changed": self.mesh_changed,
            "hash_before": self.hash_before,
            "hash_after": self.hash_after,
            "refused": self.refused,
        }


def _default_limits(mesh: Mesh) -> holes.HoleFillLimits:
    return holes.HoleFillLimits()


def _loop_plan_or_none(session: RepairSession, vertices: list[int],
                       limits: holes.HoleFillLimits) -> dict:
    try:
        return holes.plan_fill(session, vertices, limits)
    except Exception as exc:  # planning must never mutate; turn into a refusal
        return {"ok": False, "message": f"计划失败：{exc}"}


# ---------------------------------------------------------------------------
# Route A: local connection edits
# ---------------------------------------------------------------------------


def route_local_connections(
    mesh: Mesh,
    *,
    unit: str = "mm",
    limits: holes.HoleFillLimits | None = None,
    fill_loops: bool = True,
    split_junctions: bool = True,
) -> tuple[RouteResult, RepairSession]:
    """Apply minimal local edits to each detected defect, one at a time."""
    session = RepairSession(mesh, unit=unit)
    limits = limits or _default_limits(mesh)
    before = analyze(mesh, unit=unit, dup_tolerance=0.01, degenerate_edge=0.001)
    result = RouteResult(
        route="local_connections",
        invariants_before=measure(mesh),
        hash_before=mesh_hash(mesh),
    )

    # 1. duplicate faces: delete the redundant copies, keep the first.
    dup = before.groups.get("duplicate_faces")
    if dup:
        try:
            recs = repair.remove_duplicate_faces(session, dup.instances)
            result.steps.append(StepResult(
                "重复面：删除冗余副本",
                bool(recs),
                f"{len(recs)} 个组；每个组保留最低编号面",
                [r.to_dict() for r in recs],
            ))
        except RepairError as exc:
            result.refused.append(f"重复面：{exc}")
            result.steps.append(StepResult("重复面：删除冗余副本", False, str(exc)))

    # 2. non-manifold junctions: split by copying vertices/faces, never
    #    deleting.  Vertex fans must be separated too, otherwise the shared
    #    endpoints keep the sheets joined and no component ever separates.
    if split_junctions:
        try:
            recs = repair.repair_nonmanifold_junctions(
                session, reason="路线A：局部拆分非流形交汇"
            )
            result.steps.append(StepResult(
                "非流形交汇：分离面扇与多余面片",
                bool(recs),
                f"{len(recs)} 步拆分；无面被删除",
                [r.to_dict() for r in recs],
            ))
        except RepairError as exc:
            result.refused.append(f"非流形交汇：{exc}")
            result.steps.append(StepResult("非流形交汇：分离面扇与多余面片", False, str(exc)))

    # 3. every closed boundary loop within limits gets its own patch; loops
    #    over the limit are refused and reported individually (no global
    #    "delete the region" shortcut).
    if fill_loops:
        applied = 0
        refused = 0
        recs_all: list[dict] = []
        # Re-plan after each fill: filling changes boundary tracing, so an
        # inventory taken once can contain loops that merge/split later.
        # Each iteration fills at most one loop and re-detects before the
        # next, which keeps every recorded loop addressable.
        seen_refusals: set[str] = set()
        for _ in range(512):
            res = analyze(session.mesh, unit=unit, dup_tolerance=0.01,
                          degenerate_edge=0.001)
            group = res.groups.get("open_boundaries")
            if not group:
                break
            filled_this_round = False
            for inst in group.instances:
                vs = [int(v) for v in inst.get("vertices", [])]
                if not inst.get("closed") or len(vs) < 3:
                    key = f"open:{len(vs)}"
                    if key not in seen_refusals:
                        seen_refusals.add(key)
                        refused += 1
                        result.refused.append(
                            f"开口 {len(vs)} 顶点：未闭合，不能自动补洞"
                        )
                    continue
                plan = _loop_plan_or_none(session, vs, limits)
                if plan.get("ok"):
                    rec = holes.fill_hole(
                        session, vs, limits,
                        reason="路线A：局部补洞（限制内）",
                    )
                    applied += 1
                    recs_all.append(rec.to_dict())
                    filled_this_round = True
                    break
                key = f"loop:{len(vs)}:{round(float(inst.get('perimeter', 0)), 4)}"
                if key not in seen_refusals:
                    seen_refusals.add(key)
                    refused += 1
                    result.refused.append(
                        f"开口 {len(vs)} 顶点 / 周长 "
                        f"{inst.get('perimeter', 0):.4g}："
                        f"{plan.get('message') or plan.get('check', {}).get('message', '超限')}"
                    )
            if not filled_this_round:
                break
        result.steps.append(StepResult(
            "开口：逐个补洞",
            applied > 0,
            f"补洞 {applied} 个；拒绝 {refused} 个（原有面全部保留）",
            recs_all,
        ))

    result.invariants_after = measure(session.mesh)
    result.hash_after = mesh_hash(session.mesh)
    result.mesh_changed = result.hash_before != result.hash_after
    result.attributes = attribute_delta(mesh, session.mesh)
    return result, session


def _loop_fingerprint(mesh: Mesh, vertices: list[int]) -> str:
    from .verify import loop_fingerprint

    return loop_fingerprint(mesh, vertices)


# ---------------------------------------------------------------------------
# Route B: valid topology first, then patch
# ---------------------------------------------------------------------------


@dataclass
class ScaffoldLoop:
    vertices: list[int]
    perimeter: float
    closure_ok: bool
    fillable: bool
    reason: str = ""


@dataclass
class Scaffold:
    """A connectivity plan built before geometry is patched."""

    manifold: bool
    plan_session: RepairSession
    loops: list[ScaffoldLoop]
    junctions_split: int
    nonmanifold_edges: int
    problems: list[str] = field(default_factory=list)


def build_scaffold(
    mesh: Mesh,
    *,
    unit: str = "mm",
    limits: holes.HoleFillLimits | None = None,
) -> Scaffold:
    """Split junctions and inventory every loop *before* filling any of them.

    The scaffold lives on its own session copy; if building it fails, the
    caller's mesh is untouched because nothing was applied to it.  Duplicate
    faces are removed first: they are redundant copies, not extra sheets, so
    splitting them as junctions would keep the duplication as if it were
    legitimate topology.
    """
    limits = limits or _default_limits(mesh)
    plan = RepairSession(mesh, unit=unit)
    problems: list[str] = []
    split_count = 0
    split_failed = False
    try:
        dup = analyze(plan.mesh, unit=unit, dup_tolerance=0.01,
                      degenerate_edge=0.001).groups.get("duplicate_faces")
        if dup:
            repair.remove_duplicate_faces(plan, dup.instances)
        recs = repair.repair_nonmanifold_junctions(
            plan, reason="路线B：先建立有效拓扑"
        )
        split_count = len(recs)
    except RepairError as exc:
        split_failed = True
        problems.append(f"非流形拆分未收敛：{exc}")

    inc = build_edge_incidence(plan.mesh.faces)
    nm = len(inc.nonmanifold_edges())
    loops_before = analyze(
        plan.mesh, unit=unit, dup_tolerance=0.01, degenerate_edge=0.001
    ).groups.get("open_boundaries")

    scaffold_loops: list[ScaffoldLoop] = []
    if loops_before:
        for inst in loops_before.instances:
            vs = [int(v) for v in inst.get("vertices", [])]
            plan_res = _loop_plan_or_none(plan, vs, limits)
            closure_ok = bool(inst.get("closed"))
            scaffold_loops.append(ScaffoldLoop(
                vertices=vs,
                perimeter=float(inst.get("perimeter", 0.0)),
                closure_ok=closure_ok,
                fillable=bool(plan_res.get("ok")),
                reason="" if plan_res.get("ok") else (
                    plan_res.get("message")
                    or plan_res.get("check", {}).get("message", "超出补洞限制")
                ),
            ))
    if nm:
        problems.append(f"拆分后仍有 {nm} 条非流形边")
    return Scaffold(
        manifold=(nm == 0 and not split_failed),
        plan_session=plan,
        loops=scaffold_loops,
        junctions_split=split_count,
        nonmanifold_edges=nm,
        problems=problems,
    )


def route_topology_first(
    mesh: Mesh,
    *,
    unit: str = "mm",
    limits: holes.HoleFillLimits | None = None,
    require_all_loops: bool = True,
) -> tuple[RouteResult, RepairSession | None]:
    """Build a manifold scaffold, then patch all loops or apply nothing.

    ``require_all_loops`` makes the route transactional: if a single loop
    is over the limit, the whole patch phase is abandoned and the returned
    session is ``None``.  This is the deliberate contrast with Route A,
    which would still fix the other loops.
    """
    limits = limits or _default_limits(mesh)
    result = RouteResult(
        route="topology_first",
        invariants_before=measure(mesh),
        hash_before=mesh_hash(mesh),
    )
    scaffold = build_scaffold(mesh, unit=unit, limits=limits)
    result.steps.append(StepResult(
        "建立有效拓扑（拆分交汇 + 清点环）",
        scaffold.junctions_split > 0 or not scaffold.problems,
        f"拆分 {scaffold.junctions_split} 条非流形边；"
        f"清点 {len(scaffold.loops)} 个环；剩余非流形边 {scaffold.nonmanifold_edges}",
        [],
    ))
    result.refused.extend(scaffold.problems)

    unfillable = [lp for lp in scaffold.loops if not lp.fillable]
    if not scaffold.manifold:
        result.refused.append("拓扑未达到流形，拒绝进入修补阶段")
        result.invariants_after = measure(mesh)
        result.hash_after = mesh_hash(mesh)
        result.attributes = attribute_delta(mesh, mesh)
        return result, None
    if require_all_loops and unfillable:
        for lp in unfillable:
            result.refused.append(
                f"{len(lp.vertices)} 顶点 / 周长 {lp.perimeter:.4g} 的环不可补：{lp.reason}"
            )
        result.steps.append(StepResult(
            "修补阶段",
            False,
            f"{len(unfillable)} 个环超限；按事务语义整体放弃，网格未改动",
        ))
        result.invariants_after = measure(mesh)
        result.hash_after = mesh_hash(mesh)
        result.attributes = attribute_delta(mesh, mesh)
        return result, None

    session = scaffold.plan_session
    records: list[dict] = []
    applied = 0
    # patch loop by loop; vertices are stable inside the scaffold session, but
    # filling changes only face rows, so the recorded loops stay valid.
    for lp in scaffold.loops:
        if not lp.fillable:
            continue
        try:
            rec = holes.fill_hole(
                session, lp.vertices, limits, reason="路线B：拓扑校验后统一修补"
            )
            applied += 1
            records.append(rec.to_dict())
        except RepairError as exc:
            result.refused.append(f"补洞失败：{exc}")
    result.steps.append(StepResult(
        "修补阶段",
        applied > 0,
        f"按已清点计划补洞 {applied} 个",
        records,
    ))
    result.invariants_after = measure(session.mesh)
    result.hash_after = mesh_hash(session.mesh)
    result.mesh_changed = result.hash_before != result.hash_after
    result.attributes = attribute_delta(mesh, session.mesh)
    return result, session


ROUTES = {
    "local_connections": route_local_connections,
    "topology_first": route_topology_first,
}
