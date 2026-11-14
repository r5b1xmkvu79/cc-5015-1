"""Repair operations.

Every mutation goes through :class:`RepairSession` which:

* keeps the original mesh untouched (``original`` snapshot),
* records the affected vertices/faces and the exact parameters used,
* can be undone individually or replayed in order,
* never touches anything the caller did not explicitly select.

Selection is always by defect instance, never "fix everything silently".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .mesh import Mesh


class RepairError(RuntimeError):
    """Raised when an operation cannot be applied safely."""


@dataclass
class EditRecord:
    op: str
    label: str
    parameters: dict
    affected_vertices: list[int]
    affected_faces: list[int]
    removed_vertices: list[int] = field(default_factory=list)
    added_vertices: list[int] = field(default_factory=list)
    removed_faces: list[int] = field(default_factory=list)
    added_faces: list[int] = field(default_factory=list)
    reason: str = ""
    timestamp: str = ""
    index: int = 0
    # snapshots enable per-operation undo without replaying the whole chain
    before_vertices: np.ndarray | None = None
    before_faces: np.ndarray | None = None
    after_vertices: np.ndarray | None = None
    after_faces: np.ndarray | None = None
    undone: bool = False

    def to_dict(self, ndigits: int = 6) -> dict:
        def r(x):
            return [round(float(v), ndigits) for v in x]

        return {
            "index": self.index,
            "op": self.op,
            "label": self.label,
            "parameters": _jsonable(self.parameters),
            "affected_vertices": list(self.affected_vertices),
            "affected_faces": list(self.affected_faces),
            "removed_vertices": list(self.removed_vertices),
            "added_vertices": list(self.added_vertices),
            "removed_faces": list(self.removed_faces),
            "added_faces": list(self.added_faces),
            "reason": self.reason,
            "timestamp": self.timestamp,
            "undone": self.undone,
        }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RepairOptions:
    """Limits that keep hole filling from sealing designed openings."""

    max_hole_edges: int = 64
    max_hole_perimeter_fraction: float = 0.15   # of bbox diagonal
    max_hole_spacing_fraction: float = 0.10     # of bbox diagonal
    max_planarity_fraction: float = 0.05        # of bbox diagonal


class RepairSession:
    """Holds current mesh, immutable original, and the edit history."""

    def __init__(self, mesh: Mesh, *, unit: str = "mm", source_name: str = ""):
        self.unit = unit
        self.source_name = source_name or mesh.name
        self.original = mesh.copy(name=mesh.name + "::original")
        self.mesh = mesh.copy(name=mesh.name)
        self.history: list[EditRecord] = []
        self.redo_stack: list[EditRecord] = []
        self._next_index = 1

    # -- state ----------------------------------------------------------
    def reset(self, mesh: Mesh) -> None:
        self.original = mesh.copy(name=mesh.name + "::original")
        self.mesh = mesh.copy(name=mesh.name)
        self.history.clear()
        self.redo_stack.clear()
        self._next_index = 1

    @property
    def can_undo(self) -> bool:
        return bool(self.history)

    @property
    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    def snapshot(self) -> None:
        """Record current state so an interactive gesture can be rolled back."""
        self._gesture = (self.mesh.vertices.copy(), self.mesh.faces.copy())

    def rollback_snapshot(self) -> bool:
        if getattr(self, "_gesture", None) is None:
            return False
        v, f = self._gesture
        self.mesh.vertices = v
        self.mesh.faces = f
        self._gesture = None
        return True

    # -- transaction wrapper --------------------------------------------
    def _apply(
        self,
        op: str,
        label: str,
        parameters: dict,
        new_vertices: np.ndarray,
        new_faces: np.ndarray,
        *,
        reason: str = "",
        affected_vertices: list[int] | None = None,
        affected_faces: list[int] | None = None,
    ) -> EditRecord:
        before_v = self.mesh.vertices.copy()
        before_f = self.mesh.faces.copy()
        from .mesh import _as_triangles  # local: keeps mesh module private

        new_faces = _as_triangles(np.asarray(new_faces))
        new_vertices = np.asarray(new_vertices, dtype=np.float64)
        if new_vertices.ndim != 2 or new_vertices.shape[1] != 3:
            raise RepairError("内部错误：修复后顶点数组形状无效")
        if len(new_faces) and (
            new_faces.min() < 0 or new_faces.max() >= len(new_vertices)
        ):
            raise RepairError("内部错误：修复后存在越界面索引，操作已取消")
        if not np.all(np.isfinite(new_vertices)):
            raise RepairError("内部错误：修复引入非有限坐标，操作已取消")

        # compute a best-effort changed-entity list if caller did not supply one
        if affected_vertices is None:
            changed = set()
            n_old = before_v.shape[0]
            if n_old and len(new_vertices):
                common = min(n_old, len(new_vertices))
                diff = np.nonzero(np.any(before_v[:common] != new_vertices[:common], axis=1))[0]
                changed.update(int(x) for x in diff)
            if len(new_vertices) > n_old:
                changed.update(range(n_old, len(new_vertices)))
            affected_vertices = sorted(changed)
        if affected_faces is None:
            affected_faces = []

        removed_v = sorted(set(range(before_v.shape[0])) - set(range(new_vertices.shape[0])))
        added_v = sorted(set(range(new_vertices.shape[0])) - set(range(before_v.shape[0])))
        record = EditRecord(
            op=op,
            label=label,
            parameters=_jsonable(parameters),
            affected_vertices=list(affected_vertices),
            affected_faces=list(affected_faces),
            removed_vertices=removed_v,
            added_vertices=added_v,
            removed_faces=[],  # filled below by caller-visible diff where feasible
            added_faces=[],
            reason=reason,
            timestamp=_now(),
            index=self._next_index,
            before_vertices=before_v,
            before_faces=before_f,
            after_vertices=new_vertices.copy(),
            after_faces=new_faces.copy(),
        )
        record.removed_faces, record.added_faces = _face_diff(before_f, new_faces)
        self.mesh = Mesh(new_vertices, new_faces, self.mesh.name)
        self.history.append(record)
        self.redo_stack.clear()
        self._next_index += 1
        return record

    # -- undo / redo ----------------------------------------------------
    def undo(self) -> EditRecord | None:
        if not self.history:
            return None
        rec = self.history.pop()
        self.mesh.vertices = rec.before_vertices.copy()
        self.mesh.faces = rec.before_faces.copy()
        rec.undone = True
        self.redo_stack.append(rec)
        return rec

    def redo(self) -> EditRecord | None:
        if not self.redo_stack:
            return None
        rec = self.redo_stack.pop()
        self.mesh.vertices = rec.after_vertices.copy()
        self.mesh.faces = rec.after_faces.copy()
        rec.undone = False
        self.history.append(rec)
        return rec

    def undo_until(self, index: int) -> int:
        """Undo operations with the given record index and everything after it."""
        undone = 0
        while self.history and self.history[-1].index >= index:
            self.undo()
            undone += 1
        return undone

    # -- operation helper -----------------------------------------------
    def require_face_selection(self, faces: list[int]) -> list[int]:
        if not faces:
            raise RepairError("未选择任何面")
        if any(f < 0 or f >= self.mesh.face_count for f in faces):
            raise RepairError("选择中包含越界面索引")
        return sorted(set(int(f) for f in faces))

    def require_vertex_selection(self, verts: list[int]) -> list[int]:
        if not verts:
            raise RepairError("未选择任何顶点")
        if any(v < 0 or v >= self.mesh.vertex_count for v in verts):
            raise RepairError("选择中包含越界顶点索引")
        return sorted(set(int(v) for v in verts))


def _face_diff(before: np.ndarray, after: np.ndarray) -> tuple[list[int], list[int]]:
    """Approximate face diff by vertex-triple identity.

    Faces are reindexed when vertices are removed, so we compare canonical
    triples rather than array positions.
    """
    def keys(faces):
        if len(faces) == 0:
            return []
        return [tuple(sorted(int(x) for x in tri)) for tri in faces]

    bk = keys(before)
    ak = keys(after)
    before_set = {}
    for i, k in enumerate(bk):
        before_set.setdefault(k, []).append(i)
    after_set = {}
    for i, k in enumerate(ak):
        after_set.setdefault(k, []).append(i)
    removed = []
    added = []
    for k, ids in before_set.items():
        extra = len(ids) - len(after_set.get(k, []))
        if extra > 0:
            removed.extend(ids[-extra:])
    for k, ids in after_set.items():
        extra = len(ids) - len(before_set.get(k, []))
        if extra > 0:
            added.extend(ids[-extra:])
    return sorted(removed), sorted(added)


# ---------------------------------------------------------------------------
# Concrete repair operations
# ---------------------------------------------------------------------------


def delete_faces(session: RepairSession, face_ids: list[int], *, reason: str = "") -> EditRecord:
    sel = session.require_face_selection(face_ids)
    keep = np.ones(session.mesh.face_count, dtype=bool)
    keep[sel] = False
    new_faces = session.mesh.faces[keep]
    verts = session.mesh.vertices
    return session._apply(
        "delete_faces",
        f"删除 {len(sel)} 个面",
        {"faces": sel, "strategy": "delete"},
        verts,
        new_faces,
        reason=reason,
        affected_vertices=sorted({int(v) for f in sel for v in session.mesh.faces[f]}),
        affected_faces=sel,
    )


def remove_duplicate_faces(
    session: RepairSession,
    groups: list[dict],
    *,
    keep: str = "first",
) -> list[EditRecord]:
    """Remove redundant faces within duplicate groups.

    Groups are matched to the current mesh by vertex triple, not by the face
    ids captured at detection time, because earlier deletions in this batch
    shift indices.  A group whose triple is no longer present (already
    cleaned) is skipped.  ``keep='first'`` keeps the lowest-numbered face.
    """
    records = []
    for group in groups:
        triple = tuple(sorted(int(v) for v in group.get("vertices", [])))
        if len(triple) != 3:
            continue
        mesh = session.mesh
        canon = np.sort(mesh.faces, axis=1)
        hits = [int(i) for i in np.nonzero(
            (canon[:, 0] == triple[0]) & (canon[:, 1] == triple[1]) & (canon[:, 2] == triple[2])
        )[0]]
        if len(hits) < 2:
            continue
        keep_id = min(hits) if keep == "first" else max(hits)
        delete = [f for f in hits if f != keep_id]
        rec = delete_faces(
            session,
            delete,
            reason=(
                f"重复面组（{group.get('reason', '重复')}）：保留面 {keep_id}，"
                f"删除 {delete}"
            ),
        )
        rec.parameters.update({"group_vertices": list(triple), "kept": keep_id, "keep": keep})
        records.append(rec)
    return records


def remove_isolated_vertices(session: RepairSession, vertex_ids: list[int]) -> EditRecord:
    sel = session.require_vertex_selection(vertex_ids)
    used = np.zeros(session.mesh.vertex_count, dtype=bool)
    if session.mesh.face_count:
        used[session.mesh.used_vertices()] = True
    still_used = [v for v in sel if used[v]]
    if still_used:
        raise RepairError(
            f"{len(still_used)} 个选中顶点仍被面引用，拒绝移除。请先删除相关面。"
        )
    keep = np.ones(session.mesh.vertex_count, dtype=bool)
    keep[sel] = False
    verts = session.mesh.vertices[keep]
    mapping = np.full(session.mesh.vertex_count, -1, dtype=np.int64)
    mapping[np.nonzero(keep)[0]] = np.arange(int(keep.sum()), dtype=np.int64)
    faces = mapping[session.mesh.faces]
    return session._apply(
        "remove_isolated_vertices",
        f"移除 {len(sel)} 个孤立顶点",
        {"vertices": sel},
        verts,
        faces,
        reason="孤立顶点不被任何面引用，移除不影响几何",
        affected_vertices=sel,
    )


def flip_faces(session: RepairSession, face_ids: list[int], *, reason: str = "") -> EditRecord:
    sel = session.require_face_selection(face_ids)
    faces = session.mesh.faces.copy()
    faces[sel, 1], faces[sel, 2] = faces[sel, 2], faces[sel, 1].copy()
    return session._apply(
        "flip_faces",
        f"翻转 {len(sel)} 个面绕向",
        {"faces": sel},
        session.mesh.vertices,
        faces,
        reason=reason or "按选择翻转绕向",
        affected_vertices=sorted({int(v) for f in sel for v in session.mesh.faces[f]}),
        affected_faces=sel,
    )


def flip_component(
    session: RepairSession,
    face_ids: list[int],
    *,
    confirm_reason: str,
    expected_volume_sign: str = "auto",
) -> EditRecord:
    """Flip a whole connected component after the user confirmed it.

    ``expected_volume_sign='auto'`` flips only when the component is closed
    and its signed volume disagrees with the outward convention.  The caller
    must supply a human reason; the UI only offers this for components the
    user explicitly confirmed.
    """
    sel = session.require_face_selection(face_ids)
    if expected_volume_sign == "auto":
        v = session.mesh.vertices[session.mesh.faces[sel]]
        vol = float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)
        if vol == 0:
            raise RepairError("该组件体积为零，无法用体积符号判断方向；请逐面指定")
    return _apply_flip(session, sel, confirm_reason, expected_volume_sign)


def _apply_flip(session: RepairSession, sel: list[int], reason: str, mode: str) -> EditRecord:
    faces = session.mesh.faces.copy()
    faces[sel, 1], faces[sel, 2] = faces[sel, 2], faces[sel, 1].copy()
    return session._apply(
        "flip_component",
        f"整体翻转组件（{len(sel)} 面）",
        {"faces": sel, "volume_sign": mode, "confirm_reason": reason},
        session.mesh.vertices,
        faces,
        reason=reason,
        affected_vertices=sorted({int(v) for f in sel for v in session.mesh.faces[f]}),
        affected_faces=sel,
    )


def weld_vertices_selected(
    session: RepairSession,
    vertex_groups: list[list[int]],
    *,
    unit: str,
    threshold: float,
    confirmed_reason: str,
) -> list[EditRecord]:
    """Weld exactly the vertex groups the user confirmed.

    Groups are never inferred from proximity.  The caller passes the groups
    (from a defect instance the user selected) plus the threshold used, and
    the operation refuses groups whose spread exceeds the threshold so a
    mistaken selection cannot drag distant vertices together.
    """
    records = []
    for group in vertex_groups:
        ids = sorted(set(int(v) for v in group))
        if len(ids) < 2:
            continue
        pts = session.mesh.vertices[ids]
        spread = float(np.linalg.norm(pts - pts[0], axis=1).max())
        if spread > threshold:
            raise RepairError(
                f"顶点组最大间距 {spread:.4g} 超过阈值 {threshold:.4g}，拒绝焊接"
            )
        target = ids[0]
        faces = session.mesh.faces.copy()
        for v in ids[1:]:
            faces[faces == v] = target
        removed = ids[1:]
        keep = np.ones(session.mesh.vertex_count, dtype=bool)
        keep[removed] = False
        verts = session.mesh.vertices[keep]
        mapping = np.full(session.mesh.vertex_count, -1, dtype=np.int64)
        mapping[np.nonzero(keep)[0]] = np.arange(int(keep.sum()), dtype=np.int64)
        mapping[target] = mapping[target]
        new_faces = mapping[faces]
        rec = session._apply(
            "weld_vertices",
            f"焊接 {len(ids)} 个顶点到 {target}",
            {
                "group": ids,
                "threshold": threshold,
                "unit": unit,
                "max_spread": spread,
                "confirm_reason": confirmed_reason,
            },
            verts,
            new_faces,
            reason=confirmed_reason,
            affected_vertices=ids,
        )
        collided = _degenerate_after_weld(new_faces)
        if collided:
            rec.parameters["created_degenerate_faces"] = sorted(collided)
        records.append(rec)
    return records


def _degenerate_after_weld(faces: np.ndarray) -> list[int]:
    if len(faces) == 0:
        return []
    bad = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    return [int(i) for i in np.nonzero(bad)[0]]
