"""Defect list, selection model, and repair actions for the main window.

The panel is the only place that can trigger a mutation.  Every action:

* requires an explicit selection (face/vertex/loop/group instance),
* shows what will change and the parameters in use,
* asks for a written or checked confirmation when the operation is a
  judgement call (weld, flip inward component, fill a large hole),
* refuses operations whose preconditions fail, with the reason shown.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core import holes, production, repair
from ..core.repair import RepairError, RepairSession
from ..core.topology import AnalysisResult
from ..core.units import format_mm, to_mm

KIND_ORDER = [
    "degenerate_faces", "duplicate_faces", "nonmanifold_edges",
    "nonmanifold_vertices", "winding_conflicts", "inward_components",
    "duplicate_vertices", "open_boundaries", "unreferenced_vertices",
]

SEVERITY_COLORS = {
    "error": QColor(220, 80, 80),
    "warning": QColor(230, 170, 60),
    "info": QColor(120, 170, 220),
}


class ConfirmDialog(QDialog):
    """Free-text justification plus optional acknowledgement checkbox."""

    auto_accept = False

    def __init__(self, parent, title: str, body: str, required_text: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        label = QLabel(body)
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(QLabel(required_text or "请说明为何此操作对当前模型是正确的："))
        self.reason = QPlainTextEdit()
        self.reason.setPlaceholderText("例如：该开口为破损，不是设计特征；相邻薄壁间距 0.4 mm，焊接阈值 0.4 mm")
        self.reason.setMaximumHeight(100)
        self._preset_reason = ""
        layout.addWidget(self.reason)
        self.ack = QCheckBox("我已核对选择范围与参数，确认执行")
        layout.addWidget(self.ack)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._buttons = buttons
        buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.reason.textChanged.connect(self._validate)
        self.ack.toggled.connect(self._validate)

    def set_preset(self, reason: str) -> None:
        """Used by headless tests: pre-fill the justification and ack."""
        self._preset_reason = reason
        self.reason.setPlainText(reason)
        self.ack.setChecked(True)

    def exec(self) -> int:
        if type(self).auto_accept:
            if not self._preset_reason:
                self.set_preset("自动化测试：已核对参数与选择范围")
            return QDialog.Accepted
        return super().exec()

    def _validate(self) -> None:
        ok = bool(self.reason.toPlainText().strip()) and self.ack.isChecked()
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(ok)

    def values(self) -> tuple[str, bool]:
        return self.reason.toPlainText().strip(), self.ack.isChecked()


class DefectPanel(QWidget):
    defectActivated = Signal(str, int)      # kind, instance index
    selectionRequested = Signal(str, int)   # kind, instance index
    analysisRefresh = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: AnalysisResult | None = None
        self._session: RepairSession | None = None
        self._unit = "mm"
        self._pending: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("缺陷列表")
        header.setStyleSheet("font-weight:600; padding:4px;")
        layout.addWidget(header)
        self.summary = QLabel("尚未分析")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color:#9aa4b2; padding:0 4px 4px 4px;")
        layout.addWidget(self.summary)

        self.tree = QListWidget()
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.itemClicked.connect(self._on_clicked)
        self.tree.itemDoubleClicked.connect(self._on_double)
        layout.addWidget(self.tree, 1)

        self.warn_box = QPlainTextEdit()
        self.warn_box.setReadOnly(True)
        self.warn_box.setMaximumHeight(110)
        self.warn_box.setPlaceholderText("待确认事项会显示在这里")
        layout.addWidget(self.warn_box)

        self.pending = QCheckBox("仅显示待确认/未解决")
        self.pending.toggled.connect(self._rebuild)
        layout.addWidget(self.pending)

    def set_context(self, session: RepairSession | None, result: AnalysisResult | None) -> None:
        self._session = session
        self._result = result
        self._rebuild()

    def set_unit(self, unit: str) -> None:
        self._unit = unit

    def set_pending(self, pending: list[dict]) -> None:
        self._pending = list(pending)
        self._render_pending()

    def _render_pending(self) -> None:
        self.warn_box.clear()
        if self._pending:
            self.warn_box.appendPlainText("—— 待确认（工具不自动处理）——")
            for item in self._pending:
                self.warn_box.appendPlainText(f"• [{item.get('kind')}] {item.get('question', '')}")
        if self._result is not None:
            if self._pending:
                self.warn_box.appendPlainText("")
            for w in self._result.warnings:
                self.warn_box.appendPlainText("• " + w)

    def _rebuild(self) -> None:
        self.tree.clear()
        if self._result is None:
            self.warn_box.clear()
            self.summary.setText("尚未分析")
            return
        total = self._result.total_instances()
        self.summary.setText(
            f"共 {total} 个缺陷实例，涉及 {len(self._result.groups)} 类；"
            "双击定位并选中，单号前色块表示严重度"
        )
        only_pending = self.pending.isChecked()
        for kind in KIND_ORDER + [k for k in self._result.groups if k not in KIND_ORDER]:
            group = self._result.groups.get(kind)
            if group is None:
                continue
            header = QListWidgetItem(f"{group.label}  [{len(group)}]")
            header.setFlags(Qt.ItemIsEnabled)
            header.setForeground(SEVERITY_COLORS.get(group.severity, QColor(200, 200, 200)))
            font = header.font()
            font.setBold(True)
            header.setFont(font)
            header.setToolTip(group.note)
            self.tree.addItem(header)
            for idx, inst in enumerate(group.instances):
                if only_pending and kind not in ("open_boundaries", "inward_components",
                                                 "duplicate_vertices", "nonmanifold_vertices",
                                                 "nonmanifold_edges"):
                    continue
                text = self._instance_text(kind, idx, inst)
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, (kind, idx))
                item.setToolTip(self._instance_tooltip(kind, inst))
                if kind in ("open_boundaries", "inward_components", "duplicate_vertices"):
                    item.setForeground(QColor(235, 200, 120))
                self.tree.addItem(item)
        for w in self._result.warnings:
            self.warn_box.appendPlainText("• " + w)
        self._render_pending()

    def _instance_text(self, kind: str, idx: int, inst: dict) -> str:
        if kind in ("degenerate_faces", "duplicate_faces", "winding_conflicts"):
            return f"  #{idx} 面 {self._short(inst.get('faces', []))}  {inst.get('reason','')}"
        if kind == "inward_components":
            return f"  #{idx} 组件 {inst.get('component','?')} {inst.get('reason','')}"
        if kind == "nonmanifold_edges":
            return f"  #{idx} 边顶点 {inst.get('vertices', [])}  {inst.get('reason','')}"
        if kind == "nonmanifold_vertices":
            return f"  #{idx} 顶点 {inst.get('vertices', [])}  {inst.get('reason','')}"
        if kind == "duplicate_vertices":
            spread = inst.get("max_spread", 0.0)
            tag = "精确重合" if inst.get("exact") else f"间距 {format_mm(spread, self._unit, 3)}"
            return f"  #{idx} {len(inst.get('vertices', []))} 顶点 {tag}"
        if kind == "open_boundaries":
            closed = "闭合" if inst.get("closed") else "未闭合"
            return (f"  #{idx} {closed}环 {inst.get('vertex_count', len(inst.get('vertices', [])))} 顶点 "
                    f"周长 {format_mm(inst.get('perimeter', 0.0), self._unit, 3)}")
        if kind == "unreferenced_vertices":
            return f"  #{idx} {len(inst.get('vertices', []))} 个孤立顶点"
        return f"  #{idx}"

    def _instance_tooltip(self, kind: str, inst: dict) -> str:
        parts = [inst.get("reason", "")]
        if kind == "open_boundaries":
            parts.append(
                f"最大段长 {format_mm(inst.get('max_spacing', 0), self._unit, 3)}\n"
                "设计开口与破损外观相同；补洞仅在尺寸/形状限制内允许。"
            )
        if kind == "duplicate_vertices":
            parts.append("默认不焊接。焊接会改变拓扑，须确认这些点确实属于同一表面。")
        if kind == "inward_components":
            parts.append("仅封闭组件可用体积判断；翻转前请确认该组件不是内腔或模具。")
        return "\n".join(p for p in parts if p)

    @staticmethod
    def _short(ids, limit=8) -> str:
        ids = list(ids)
        if len(ids) <= limit:
            return str(ids)
        return str(ids[:limit]) + f" …(+{len(ids)-limit})"

    def _on_clicked(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if data:
            self.selectionRequested.emit(data[0], data[1])

    def _on_double(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.UserRole)
        if data:
            self.defectActivated.emit(data[0], data[1])

    def current_defect(self) -> tuple[str, dict] | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        data = item.data(Qt.UserRole)
        if not data or self._result is None:
            return None
        kind, idx = data
        group = self._result.groups.get(kind)
        if group is None or idx >= len(group.instances):
            return None
        return kind, group.instances[idx]

    def select_defect(self, kind: str, index: int) -> bool:
        """Programmatically focus an instance; keeps list and canvas in sync."""
        for i in range(self.tree.count()):
            item = self.tree.item(i)
            if item is None:
                continue
            data = item.data(Qt.UserRole)
            if data == (kind, index):
                self.tree.setCurrentItem(item)
                return True
        return False


class RepairPanel(QWidget):
    """Buttons that mutate the mesh; all go through the session."""

    changed = Signal(str)          # human readable status
    requestRepaint = Signal()
    requestRefreshAnalysis = Signal()
    repairApplied = Signal(object)  # RepairOutcome, for before/after display
    repairFailed = Signal(str)      # failure text, fed back to the panel

    def __init__(self, defect_panel: DefectPanel, parent=None):
        super().__init__(parent)
        self.defect_panel = defect_panel
        self.session: RepairSession | None = None
        self.unit = "mm"
        self._result: AnalysisResult | None = None
        self._selected_faces: list[int] = []
        self._selected_vertices: list[int] = []
        self._selected_loop: list[int] | None = None
        self.headless_auto_confirm = False
        self.last_error = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        face_box = QGroupBox("面操作（需选择）")
        fv = QVBoxLayout(face_box)
        self.btn_repair_pick = QPushButton("修复选中的坏面（内核驱动）")
        self.btn_repair_pick.setToolTip(
            "按画布中选中的面定位缺陷实例，调用修复内核；"
            "前后拓扑不变量与属性变化会显示在右侧"
        )
        self.btn_del = QPushButton("删除选中面")
        self.btn_del.setToolTip("仅删除画布中明确选中的面")
        self.btn_flip = QPushButton("翻转选中面绕向")
        self.btn_fix_winding = QPushButton("按检测结果翻转绕向冲突面")
        self.btn_del_dupes = QPushButton("删除选中重复面组")
        self.btn_flip_comp = QPushButton("翻转选中的整体朝内组件…")
        for b in (self.btn_repair_pick, self.btn_del, self.btn_flip,
                  self.btn_fix_winding, self.btn_del_dupes, self.btn_flip_comp):
            fv.addWidget(b)
        layout.addWidget(face_box)

        vert_box = QGroupBox("顶点操作（需选择）")
        vv = QFormLayout(vert_box)
        self.weld_tol = QDoubleSpinBox()
        self.weld_tol.setDecimals(4)
        self.weld_tol.setRange(0.0, 1e6)
        self.weld_tol.setValue(0.05)
        self.weld_unit_label = QLabel("mm")
        tol_row = QHBoxLayout()
        tol_row.addWidget(self.weld_tol)
        tol_row.addWidget(self.weld_unit_label)
        vv.addRow("合并阈值", tol_row)
        self.weld_hint = QLabel("阈值以模型单位输入；焊接只作用于明确选中的顶点组")
        self.weld_hint.setWordWrap(True)
        vv.addRow(self.weld_hint)
        self.btn_weld = QPushButton("焊接选中顶点组…")
        self.btn_drop_iso = QPushButton("移除选中孤立顶点")
        vv.addRow(self.btn_weld)
        vv.addRow(self.btn_drop_iso)
        layout.addWidget(vert_box)

        hole_box = QGroupBox("开口处理")
        hv = QFormLayout(hole_box)
        self.max_edges = QSpinBox()
        self.max_edges.setRange(3, 4096)
        self.max_edges.setValue(64)
        self.perim_frac = QDoubleSpinBox()
        self.perim_frac.setRange(0.001, 1.0)
        self.perim_frac.setDecimals(3)
        self.perim_frac.setValue(0.15)
        self.spacing_frac = QDoubleSpinBox()
        self.spacing_frac.setRange(0.001, 1.0)
        self.spacing_frac.setDecimals(3)
        self.spacing_frac.setValue(0.10)
        self.planarity_frac = QDoubleSpinBox()
        self.planarity_frac.setRange(0.001, 1.0)
        self.planarity_frac.setDecimals(3)
        self.planarity_frac.setValue(0.05)
        hv.addRow("最大边界边数", self.max_edges)
        hv.addRow("周长上限(×对角线)", self.perim_frac)
        hv.addRow("段长上限(×对角线)", self.spacing_frac)
        hv.addRow("平面偏差上限(×对角)", self.planarity_frac)
        self.btn_check_loop = QPushButton("检查选中的开口环")
        self.btn_fill = QPushButton("补洞（需复核限制）…")
        hv.addRow(self.btn_check_loop)
        hv.addRow(self.btn_fill)
        self.loop_status = QLabel("未选择开口")
        self.loop_status.setWordWrap(True)
        self.loop_status.setStyleSheet("color:#9aa4b2;")
        hv.addRow(self.loop_status)
        layout.addWidget(hole_box)

        misc_box = QGroupBox("全局")
        mv = QVBoxLayout(misc_box)
        self.btn_compact = QPushButton("清理所有孤立顶点…")
        mv.addWidget(self.btn_compact)
        layout.addWidget(misc_box)

        layout.addStretch(1)

        self.btn_repair_pick.clicked.connect(self._repair_picked_face)
        self.btn_del.clicked.connect(self._delete_faces)
        self.btn_flip.clicked.connect(self._flip_faces)
        self.btn_fix_winding.clicked.connect(self._fix_winding)
        self.btn_del_dupes.clicked.connect(self._delete_duplicate_group)
        self.btn_flip_comp.clicked.connect(self._flip_component)
        self.btn_weld.clicked.connect(self._weld)
        self.btn_drop_iso.clicked.connect(self._drop_isolated)
        self.btn_check_loop.clicked.connect(self._check_loop)
        self.btn_fill.clicked.connect(self._fill_hole)
        self.btn_compact.clicked.connect(self._compact_all)
        self.weld_tol.valueChanged.connect(self._update_weld_hint)
        self._update_weld_hint()

    def set_context(self, session: RepairSession | None, result: AnalysisResult | None, unit: str) -> None:
        self.session = session
        self._result = result
        self.unit = unit
        self.weld_unit_label.setText(unit)
        self._update_weld_hint()
        self._set_enabled(session is not None)

    def _set_enabled(self, on: bool) -> None:
        for b in (self.btn_repair_pick, self.btn_del, self.btn_flip,
                  self.btn_fix_winding, self.btn_del_dupes, self.btn_flip_comp,
                  self.btn_weld, self.btn_drop_iso, self.btn_check_loop,
                  self.btn_fill, self.btn_compact):
            b.setEnabled(on)

    def set_selection(self, faces, vertices, loop) -> None:
        self._selected_faces = sorted(int(f) for f in faces)
        self._selected_vertices = sorted(int(v) for v in vertices)
        self._selected_loop = [int(v) for v in loop] if loop else None
        self.btn_check_loop.setEnabled(self.session is not None and bool(self._selected_loop))
        self.btn_fill.setEnabled(self.session is not None and bool(self._selected_loop))
        if self._selected_loop:
            self.loop_status.setText(f"已选开口环 {len(self._selected_loop)} 顶点")

    def _update_weld_hint(self) -> None:
        v = self.weld_tol.value()
        self.weld_hint.setText(
            f"{v:.4g} {self.unit} ≈ {to_mm(v, self.unit):.4g} mm；"
            "只焊接明确选中的顶点组，阈值外的组会被拒绝"
        )

    def _require_session(self) -> RepairSession:
        if self.session is None:
            raise RepairError("请先加载模型")
        return self.session

    def _fail(self, message: str) -> None:
        self.last_error = message
        if self.headless_auto_confirm:
            return
        QMessageBox.warning(self, "无法执行", message)

    def _inform(self, title: str, message: str) -> None:
        if self.headless_auto_confirm:
            return
        QMessageBox.information(self, title, message)

    def _confirm(self, title: str, body: str) -> tuple[bool, str]:
        if self.headless_auto_confirm:
            return True, "自动确认（无头模式）"
        dlg = ConfirmDialog(self, title, body)
        if dlg.exec() != QDialog.Accepted:
            return False, ""
        reason, _ack = dlg.values()
        return True, reason

    def _done(self, msg: str) -> None:
        self.changed.emit(msg)
        self.requestRepaint.emit()
        self.requestRefreshAnalysis.emit()

    # -- face ops -------------------------------------------------------
    def _repair_picked_face(self) -> None:
        """Pick bad faces on the canvas, repair via the shared kernel."""
        if not self._selected_faces:
            self._fail("请先在画布中选择要修复的坏面")
            return
        if self._result is None or self.session is None:
            self._fail("尚无检测结果，请先检测")
            return
        found = production.find_defect_for_faces(self._result, self._selected_faces)
        if found is None:
            self._fail(
                "选中的面不属于任何可自动修复的缺陷实例。\n"
                "请用缺陷列表定位，或确认所选面确实属于重复/退化/绕向/非流形缺陷。"
            )
            return
        kind, idx = found
        if kind in ("inward_components", "duplicate_vertices", "open_boundaries"):
            ok, reason = self._confirm(
                "确认修复",
                f"选中的面属于“{kind}”实例 #{idx}。\n"
                "该操作会改变连通性或几何，请确认选择范围与参数正确。",
            )
            if not ok:
                return
        else:
            reason = f"按画布选中面修复 {kind}#{idx}"
        try:
            outcome = production.apply_to_instance(
                self.session, kind, idx, reason=reason, limits=self._limits()
            )
        except RepairError as exc:
            self.last_error = str(exc)
            self.repairFailed.emit(str(exc))
            self._fail(str(exc))
            return
        self.repairApplied.emit(outcome)
        self._done(outcome.message + "；前后变化已显示在“修复对照”")

    def _delete_faces(self) -> None:
        if not self._selected_faces:
            self._fail("请先在画布中选择要删除的面")
            return
        try:
            rec = repair.delete_faces(self._require_session(), self._selected_faces,
                                      reason="用户明确选择删除")
        except RepairError as exc:
            self._fail(str(exc))
            return
        self._done(f"{rec.label}；已记录受影响面 {len(rec.affected_faces)} 个")

    def _flip_faces(self) -> None:
        if not self._selected_faces:
            self._fail("请先选择要翻转的面")
            return
        try:
            rec = repair.flip_faces(self._require_session(), self._selected_faces)
        except RepairError as exc:
            self._fail(str(exc))
            return
        self._done(rec.label)

    def _fix_winding(self) -> None:
        if self._result is None:
            return
        group = self._result.groups.get("winding_conflicts")
        if group is None:
            self._done("未检测到绕向冲突")
            return
        faces = sorted({int(f) for inst in group.instances for f in inst.get("faces", [])})
        if not faces:
            return
        ok, reason = self._confirm(
            "翻转绕向冲突面",
            f"将翻转 {len(faces)} 个被检测为绕向冲突的面。\n"
            "这只是让相邻面在共享边上方向相反，不保证整体朝外。",
        )
        if not ok:
            return
        try:
            rec = repair.flip_faces(self._require_session(), faces, reason=reason)
        except RepairError as exc:
            self._fail(str(exc))
            return
        self._done(rec.label)

    def _delete_duplicate_group(self) -> None:
        current = self.defect_panel.current_defect()
        if current is None:
            self._fail("请在缺陷列表中选择一个重复面组")
            return
        kind, inst = current
        if kind != "duplicate_faces":
            self._fail("当前选择不是重复面组")
            return
        try:
            recs = repair.remove_duplicate_faces(self._require_session(), [inst])
        except RepairError as exc:
            self._fail(str(exc))
            return
        if recs:
            self._done(f"删除重复面 {len(recs[0].removed_faces)} 个，保留面已记录")

    def _flip_component(self) -> None:
        current = self.defect_panel.current_defect()
        if current is None or current[0] != "inward_components":
            self._fail("请在缺陷列表中选择一个“整体朝内组件”")
            return
        _kind, inst = current
        faces = [int(f) for f in inst.get("faces", [])]
        vol = inst.get("reason", "")
        ok, reason = self._confirm(
            "翻转整体朝内组件",
            f"组件 {inst.get('component')} 共 {len(faces)} 面，{vol}。\n"
            "封闭组件的负体积通常意味着整体朝内，但内腔/模具可能本就朝内。\n"
            "请确认这不是有意设计的内向表面。",
        )
        if not ok:
            return
        try:
            rec = repair.flip_component(self._require_session(), faces,
                                        confirm_reason=reason)
        except RepairError as exc:
            self._fail(str(exc))
            return
        self._done(rec.label)

    # -- vertex ops -----------------------------------------------------
    def _current_duplicate_group(self) -> list[int] | None:
        current = self.defect_panel.current_defect()
        if current and current[0] == "duplicate_vertices":
            return [int(v) for v in current[1].get("vertices", [])]
        return None

    def _weld(self) -> None:
        group = self._current_duplicate_group()
        if group is None:
            if len(self._selected_vertices) >= 2:
                group = list(self._selected_vertices)
            else:
                self._fail("请在缺陷列表中选择一个重合顶点组，或在画布中选择 ≥2 个顶点")
                return
        threshold = self.weld_tol.value()
        spread = 0.0
        if self.session is not None and group:
            pts = self.session.mesh.vertices[group]
            spread = float(np.linalg.norm(pts - pts[0], axis=1).max())
        if spread > threshold:
            self._fail(
                f"该组最大间距 {format_mm(spread, self.unit)} 超过阈值 "
                f"{format_mm(threshold, self.unit)}。请先确认单位与阈值；"
                "薄壁两侧重合点不应焊接。"
            )
            return
        ok, reason = self._confirm(
            "焊接顶点组",
            f"将把 {len(group)} 个顶点焊为 1 个：{group[:12]}{'…' if len(group) > 12 else ''}\n"
            f"组内最大间距 {format_mm(spread, self.unit)}，阈值 {format_mm(threshold, self.unit)}。\n"
            "若这些点分属相邻薄壁的两侧，焊接会把两层粘成一层。",
        )
        if not ok:
            return
        try:
            recs = repair.weld_vertices_selected(
                self._require_session(), [group], unit=self.unit,
                threshold=threshold, confirmed_reason=reason,
            )
        except RepairError as exc:
            self._fail(str(exc))
            return
        extra = ""
        if recs and recs[0].parameters.get("created_degenerate_faces"):
            extra = f"；新增退化面 {len(recs[0].parameters['created_degenerate_faces'])} 个，请复核"
        self._done(f"焊接完成{extra}")

    def _drop_isolated(self) -> None:
        if not self._selected_vertices:
            self._fail("请先选择孤立顶点")
            return
        try:
            rec = repair.remove_isolated_vertices(self._require_session(), self._selected_vertices)
        except RepairError as exc:
            self._fail(str(exc))
            return
        self._done(rec.label)

    def _compact_all(self) -> None:
        session = self._require_session()
        if session.mesh.face_count == 0:
            self._fail("网格为空")
            return
        used = np.zeros(session.mesh.vertex_count, dtype=bool)
        used[session.mesh.used_vertices()] = True
        dangling = [int(v) for v in np.nonzero(~used)[0]]
        if not dangling:
            self._done("没有孤立顶点")
            return
        if not self.headless_auto_confirm:
            ans = QMessageBox.question(
                self, "清理孤立顶点",
                f"将移除 {len(dangling)} 个未被任何面引用的顶点。\n此操作可撤销。",
            )
            if ans != QMessageBox.Yes:
                return
        try:
            rec = repair.remove_isolated_vertices(session, dangling)
        except RepairError as exc:
            self._fail(str(exc))
            return
        self._done(rec.label)

    # -- holes ----------------------------------------------------------
    def _limits(self) -> holes.HoleFillLimits:
        return holes.HoleFillLimits(
            max_edges=self.max_edges.value(),
            perimeter_fraction=self.perim_frac.value(),
            spacing_fraction=self.spacing_frac.value(),
            planarity_fraction=self.planarity_frac.value(),
        )

    def _check_loop(self) -> None:
        if not self._selected_loop or self.session is None:
            self._fail("请先在画布中选中一个开口环（单击缺陷列表中的开口）")
            return
        check = holes.check_loop(self.session.mesh, self._selected_loop, self._limits())
        if check["ok"]:
            self.loop_status.setText(
                f"可通过限制：{check['vertex_count']} 顶点，"
                f"周长 {format_mm(check['perimeter'], self.unit)}，"
                f"平面偏差 {format_mm(check['plane_deviation'], self.unit)}"
            )
            self._inform("开口检查通过",
                         self.loop_status.text() + "\n\n仍请确认这不是设计开口。")
        else:
            self.loop_status.setText("未通过：" + check["message"])
            self._fail("开口不满足补洞限制：" + check["message"])

    def _fill_hole(self) -> None:
        if not self._selected_loop or self.session is None:
            self._fail("请先选择一个开口环")
            return
        plan = holes.plan_fill(self.session, self._selected_loop, self._limits())
        if not plan["ok"]:
            msg = plan.get("message") or plan["check"].get("message", "校验失败")
            self.loop_status.setText("未通过：" + msg)
            self._fail("拒绝补洞：" + msg)
            return
        chk = plan["check"]
        ok, reason = self._confirm(
            "确认补洞",
            f"将为 {len(self._selected_loop)} 顶点开口添加 {plan['added_faces']} 个面。\n"
            f"周长 {format_mm(chk['perimeter'], self.unit)}，"
            f"平面偏差 {format_mm(chk['plane_deviation'], self.unit)}，"
            f"新增面积 {format_mm(plan['added_area'], self.unit)}²。\n"
            "设计开口与破损开口外观相同，请确认此开口确属破损。",
        )
        if not ok:
            return
        try:
            rec = holes.fill_hole(self.session, self._selected_loop, self._limits(),
                                  reason=reason)
        except RepairError as exc:
            self._fail(str(exc))
            return
        self._done(rec.label + "；已记录新增面与限制参数")
