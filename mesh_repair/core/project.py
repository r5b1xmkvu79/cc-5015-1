"""Project document: owns session state, analysis, and export provenance."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import io_mesh
from .mesh import Mesh
from .repair import RepairSession
from .topology import AnalysisResult, analyze
from .units import UnitContext, mm_per_unit
from .verify import verification_report


@dataclass
class AnalysisSettings:
    unit: str = "mm"
    dup_tolerance_mm: float = 0.01
    degenerate_edge_mm: float = 0.001
    aspect_max: float = 1e4

    def dup_tolerance(self) -> float:
        return UnitContext(self.unit, self.dup_tolerance_mm / mm_per_unit(self.unit)).value

    def degenerate_edge(self) -> float:
        return UnitContext(self.unit, self.degenerate_edge_mm / mm_per_unit(self.unit)).value


class ProjectDocument:
    """One loaded model plus its repair session and verification results."""

    def __init__(self) -> None:
        self.session: RepairSession | None = None
        self.settings = AnalysisSettings()
        self.source_path: Path | None = None
        self.baseline: AnalysisResult | None = None
        self.current: AnalysisResult | None = None
        self.baseline_fingerprints: dict = {}
        self.baseline_fingerprints_from_file: dict = {}
        self.origin_provenance: str = ""
        self.provenance_id = ""
        self.pending_confirmations: list[dict] = []
        self.last_verification: dict | None = None
        self.notes: list[str] = []

    # -- loading --------------------------------------------------------
    def load(self, path: str | Path, unit: str) -> Mesh:
        mesh = io_mesh.load_mesh(path)
        mesh.name = Path(path).stem
        self.session = RepairSession(mesh, unit=unit, source_name=str(path))
        self.settings.unit = unit
        self.source_path = Path(path)
        self.notes.clear()
        sidecar = io_mesh.read_sidecar(path)
        embedded = io_mesh.read_embedded_provenance(path)

        # provenance chain: every load gets its own id, but remembers the
        # origin (the id of the first import) so repairs can be traced back.
        origin = ""
        baseline_fps = None
        if sidecar:
            origin = str(sidecar.get("origin_provenance") or sidecar.get("provenance_id") or "")
            baseline_fps = sidecar.get("baseline_fingerprints")
            self.notes.append(
                f"检测到审计记录：本次加载来源 {sidecar.get('provenance_id', '未知')}，"
                f"最初导入来源 {origin or '未知'}"
            )
            if sidecar.get("operations"):
                self.notes.append(f"该文件此前已有 {len(sidecar['operations'])} 条修复记录")
        elif embedded:
            origin = embedded
            self.notes.append(f"文件内嵌来源标记：{embedded}（无审计记录）")

        self.origin_provenance = origin
        self.provenance_id = uuid.uuid4().hex[:16]
        self.reanalyze(update_baseline=True)
        if baseline_fps:
            # keep the recorded fingerprints so the UI can show which original
            # defects the freshly loaded asset still contains.
            self.baseline_fingerprints_from_file = baseline_fps
            self.notes.append(
                f"审计记录包含 {sum(len(v) for v in baseline_fps.values())} 条原始缺陷指纹，"
                "可用于核对仍存在的问题"
            )
        return self.session.mesh

    def close(self) -> None:
        self.session = None
        self.baseline = None
        self.current = None
        self.baseline_fingerprints = {}
        self.last_verification = None
        self.pending_confirmations.clear()
        self.notes.clear()

    # -- analysis -------------------------------------------------------
    def analyzer(self, mesh: Mesh | None = None) -> AnalysisResult:
        if self.session is None:
            raise RuntimeError("尚未加载模型")
        s = self.settings
        return analyze(
            mesh or self.session.mesh,
            unit=s.unit,
            dup_tolerance=s.dup_tolerance(),
            degenerate_edge=s.degenerate_edge(),
            aspect_max=s.aspect_max,
        )

    def reanalyze(self, *, update_baseline: bool = False) -> AnalysisResult:
        res = self.analyzer()
        self.current = res
        if update_baseline or self.baseline is None:
            from .verify import all_fingerprints

            self.baseline = self.analyzer(self.session.original)
            self.baseline_fingerprints = all_fingerprints(self.session.original, self.baseline)
            self._baseline_original_fps = self.baseline_fingerprints
        self._refresh_pending_confirmations(res)
        return res

    def _refresh_pending_confirmations(self, res: AnalysisResult) -> None:
        """Summarise the defects that the tool cannot decide on its own.

        These are the items a human must confirm before any repair: each
        open boundary (designed vs damage), each inward component (cavity vs
        error), each near-coincident vertex group (same surface vs two thin
        walls), every open shell whose orientation is unknowable, and each
        non-manifold entity the tool refuses to split automatically.
        """
        pending: list[dict] = []
        boundaries = res.groups.get("open_boundaries")
        if boundaries:
            for i, inst in enumerate(boundaries.instances):
                pending.append({
                    "kind": "open_boundaries",
                    "index": i,
                    "question": "该开口是破损还是设计特征？补洞前必须人工确认。",
                    "perimeter": inst.get("perimeter"),
                    "vertex_count": len(inst.get("vertices", [])),
                    "closed": inst.get("closed"),
                })
        inward = res.groups.get("inward_components")
        if inward:
            for i, inst in enumerate(inward.instances):
                pending.append({
                    "kind": "inward_components",
                    "index": i,
                    "question": "该封闭组件朝内是有意设计（内腔/模具）还是错误？",
                    "component": inst.get("component"),
                    "face_count": len(inst.get("faces", [])),
                })
        close = res.groups.get("duplicate_vertices")
        if close:
            near = [inst for inst in close.instances if not inst.get("exact")]
            if near:
                pending.append({
                    "kind": "duplicate_vertices",
                    "index": 0,
                    "question": (
                        f"{len(near)} 组非零间距的邻近顶点；它们是同一表面上的重复点，"
                        "还是相邻薄壁的两侧？默认不焊接。"
                    ),
                    "count": len(near),
                })
        for kind, question in (
            ("nonmanifold_edges", "非流形边不能自动拆分，需人工定位并指定处理方式。"),
            ("nonmanifold_vertices", "非流形顶点不能自动修复，需人工定位。"),
        ):
            group = res.groups.get(kind)
            if group and len(group):
                pending.append({
                    "kind": kind,
                    "index": 0,
                    "question": question,
                    "count": len(group),
                })
        orient = res.metrics.get("orientation", {})
        for comp in orient.get("components", []):
            if not comp.get("closed"):
                pending.append({
                    "kind": "orientation_unresolved",
                    "index": comp["id"],
                    "question": "该组件不封闭，无法用体积判断内外方向；请人工指定。",
                    "face_count": comp.get("face_count"),
                })
        self.pending_confirmations = pending

    def verify(self) -> dict:
        if self.session is None:
            raise RuntimeError("尚未加载模型")
        if self.baseline is None:
            self.reanalyze(update_baseline=True)
        after = self.analyzer()
        self.current = after
        self._refresh_pending_confirmations(after)
        report = verification_report(
            self.session.original, self.baseline, self.session.mesh, after
        )
        self.last_verification = report
        return report

    # -- tracing --------------------------------------------------------
    def trace_baseline_defects_on(self, mesh: Mesh) -> dict:
        """Recompute baseline fingerprints against a freshly loaded mesh.

        Used when a repaired asset is reloaded: we re-detect and compare to
        the fingerprint set recorded from the original import.
        """
        from .verify import all_fingerprints

        result = self.analyzer(mesh)
        fps = all_fingerprints(mesh, result)
        return {"analysis": result, "fingerprints": fps}

    # -- persistence ----------------------------------------------------
    def audit_payload(self) -> dict:
        if self.session is None:
            return {}
        s = self.session
        if self.current is not None:
            self._refresh_pending_confirmations(self.current)
        return {
            "tool": "mesh_repair",
            "version": "0.1.0",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provenance_id": self.provenance_id,
            "origin_provenance": self.origin_provenance or self.provenance_id,
            "source": str(self.source_path) if self.source_path else "",
            "unit": s.unit,
            "mm_per_unit": mm_per_unit(s.unit),
            "settings": {
                "dup_tolerance_mm": self.settings.dup_tolerance_mm,
                "dup_tolerance_model": self.settings.dup_tolerance(),
                "degenerate_edge_mm": self.settings.degenerate_edge_mm,
                "degenerate_edge_model": self.settings.degenerate_edge(),
                "aspect_max": self.settings.aspect_max,
            },
            "baseline_defects": self._baseline_summary(),
            "baseline_fingerprints": getattr(self, "_baseline_original_fps", {}),
            "operations": [r.to_dict() for r in s.history],
            "pending_confirmations": list(self.pending_confirmations),
            "baseline_metrics": self.baseline.metrics if self.baseline else {},
            "current_metrics": self.current.metrics if self.current else {},
            "verification": self.last_verification,
            "origin_defect_check": self.check_origin_defects(),
            "notes": list(self.notes),
        }

    def check_origin_defects(self) -> dict:
        """Which fingerprints recorded in the original audit are still present.

        This is the "reloaded model can be traced back to the original
        defect" requirement: fingerprints recorded at first import are
        recomputed against the current mesh and matched.
        """
        recorded = self.baseline_fingerprints_from_file
        if not recorded:
            return {"available": False}
        from .verify import all_fingerprints

        if self.current is None or self.session is None:
            return {"available": False}
        current = all_fingerprints(self.session.mesh, self.current)
        out = {"available": True, "kinds": {}}
        for kind, recorded_insts in recorded.items():
            cur = current.get(kind, [])
            cur_sums = {i.get("summary") for i in cur if i.get("summary")}
            rec_sums = {i.get("summary") for i in recorded_insts if i.get("summary")}
            still = sorted(rec_sums & cur_sums)
            gone = sorted(rec_sums - cur_sums)
            out["kinds"][kind] = {
                "recorded": len(rec_sums),
                "still_present": len(still),
                "resolved": len(gone),
                "still_present_fingerprints": still,
            }
        return out

    def _baseline_summary(self) -> dict:
        if self.baseline is None:
            return {}
        out = {}
        for kind, group in self.baseline.groups.items():
            out[kind] = {
                "label": group.label,
                "count": len(group),
                "severity": group.severity,
                "note": group.note,
                "auto_repairable": group.auto_repairable,
            }
        return out

    def export(self, path: str | Path, *, comment: str = "") -> dict:
        if self.session is None:
            raise RuntimeError("尚未加载模型")
        io_mesh.save_mesh(
            self.session.mesh, path, provenance_id=self.provenance_id, comment=comment
        )
        payload = self.audit_payload()
        payload["exports"] = [str(path)]
        side = io_mesh.write_sidecar(path, payload)
        return {"mesh": str(path), "sidecar": str(side), "audit": payload}

    def save_audit_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(
            json.dumps(self.audit_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return p
