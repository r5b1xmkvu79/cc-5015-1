"""Main application window."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import io_mesh
from ..core.project import AnalysisSettings, ProjectDocument
from ..core.units import UNIT_NAMES, format_mm, unit_label
from .panels import DefectPanel, RepairPanel
from .viewport import MeshViewport

FILE_FILTER = "网格文件 (*.stl *.obj *.ply);;所有文件 (*)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("机械网格修复工具")
        self.resize(1440, 900)
        self.doc = ProjectDocument()
        self._loop_selection: list[int] | None = None
        self._active_defect: tuple[str, dict] | None = None
        # When true, modal dialogs are suppressed (used by headless tests and
        # automated benchmarks).  Never enabled from the normal UI.
        self.headless_auto_confirm = False
        self._suppress_exit_prompt = False

        self.viewport = MeshViewport(self)
        self.setCentralWidget(self.viewport)
        self._build_actions()
        self._build_docks()
        self.setStatusBar(QStatusBar())
        self._update_actions()
        self._status("就绪。请导入网格，或在“基准”菜单生成测试件。")

        self.viewport.facePicked.connect(self._on_face_picked)
        self.viewport.vertexPicked.connect(self._on_vertex_picked)
        self.viewport.selectionCleared.connect(self._clear_selection)

    # -- construction ---------------------------------------------------
    def _build_actions(self) -> None:
        mb = self.menuBar()
        file_menu = mb.addMenu("文件(&F)")
        self.act_open = QAction("导入网格…", self, shortcut=QKeySequence.Open)
        self.act_open.triggered.connect(self._open)
        file_menu.addAction(self.act_open)
        self.act_export = QAction("导出修复结果…", self, shortcut=QKeySequence.Save)
        self.act_export.triggered.connect(self._export)
        file_menu.addAction(self.act_export)
        self.act_audit = QAction("导出审计 JSON…", self)
        self.act_audit.triggered.connect(self._export_audit)
        file_menu.addAction(self.act_audit)
        self.act_shots = QAction("导出视图截图…", self)
        self.act_shots.triggered.connect(self._export_view)
        file_menu.addAction(self.act_shots)
        file_menu.addSeparator()
        self.act_quit = QAction("退出", self, shortcut=QKeySequence.Quit)
        self.act_quit.triggered.connect(self.close)
        file_menu.addAction(self.act_quit)

        edit_menu = mb.addMenu("编辑(&E)")
        self.act_undo = QAction("撤销", self, shortcut=QKeySequence.Undo)
        self.act_undo.triggered.connect(self._undo)
        edit_menu.addAction(self.act_undo)
        self.act_redo = QAction("重做", self, shortcut=QKeySequence.Redo)
        self.act_redo.triggered.connect(self._redo)
        edit_menu.addAction(self.act_redo)
        edit_menu.addSeparator()
        self.act_reanalyze = QAction("重新检测", self, shortcut="F5")
        self.act_reanalyze.triggered.connect(lambda: self._reanalyze())
        edit_menu.addAction(self.act_reanalyze)
        self.act_verify = QAction("复核修复结果", self, shortcut="F6")
        self.act_verify.triggered.connect(self._verify)
        edit_menu.addAction(self.act_verify)

        bench = mb.addMenu("基准(&B)")
        for label, name in [
            ("薄壁双层", "thin_wall"),
            ("重复面盒体", "duplicate_faces"),
            ("带孔板（含设计开口）", "holed_plate"),
            ("相反方向组件", "mixed_orientation"),
            ("缺面开放盒", "open_shell"),
            ("综合压力样本", "combined_stress"),
        ]:
            act = QAction(label, self)
            act.triggered.connect(lambda _=False, n=name: self._load_benchmark(n))
            bench.addAction(act)
        bench.addSeparator()
        act_write = QAction("导出全部基准文件到目录…", self)
        act_write.triggered.connect(self._write_benchmarks)
        bench.addAction(act_write)

        view_menu = mb.addMenu("视图(&V)")
        self.act_edges = QAction("显示边", self, checkable=True, checked=True)
        self.act_edges.toggled.connect(self._toggle_edges)
        view_menu.addAction(self.act_edges)
        self.act_verts = QAction("显示顶点", self, checkable=True, checked=False)
        self.act_verts.toggled.connect(self._toggle_vertices)
        view_menu.addAction(self.act_verts)
        act_frame = QAction("适配视图", self, shortcut="F")
        act_frame.triggered.connect(self.viewport.frame_model)
        view_menu.addAction(act_frame)

    def set_headless_auto_confirm(self, enabled: bool) -> None:
        """Suppress modal dialogs for automated runs (tests, batch checks)."""
        self.headless_auto_confirm = enabled
        self.repair_panel.headless_auto_confirm = enabled

    def _build_docks(self) -> None:
        # left: defects + repair
        self.defect_panel = DefectPanel()
        self.repair_panel = RepairPanel(self.defect_panel)

        left = QTabWidget()
        left.addTab(self.defect_panel, "缺陷")
        left.addTab(self.repair_panel, "修复")
        self.left_tabs = left
        dock_left = QDockWidget("检测与修复", self)
        dock_left.setWidget(left)
        dock_left.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock_left.setMinimumWidth(380)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock_left)

        # right: history / verification / details
        self.history_list = QListWidget()
        self.history_list.setAlternatingRowColors(True)
        self.history_list.itemDoubleClicked.connect(self._on_history_double_click)
        hist_widget = QWidget()
        hv = QVBoxLayout(hist_widget)
        hv.setContentsMargins(0, 0, 0, 0)
        hv.addWidget(QLabel("每次修改记录：操作、参数、受影响顶点/面。双击可回退到该步之前。"))
        hv.addWidget(self.history_list, 1)
        row = QHBoxLayout()
        self.btn_undo = QPushButton("撤销上一步")
        self.btn_undo.clicked.connect(self._undo)
        self.btn_redo = QPushButton("重做")
        self.btn_redo.clicked.connect(self._redo)
        self.btn_history_clear = QPushButton("清空历史")
        self.btn_history_clear.clicked.connect(self._clear_history)
        row.addWidget(self.btn_undo)
        row.addWidget(self.btn_redo)
        row.addWidget(self.btn_history_clear)
        hv.addLayout(row)

        self.detail_view = QPlainTextEdit()
        self.detail_view.setReadOnly(True)
        self.detail_view.setPlaceholderText("选择历史记录或缺陷以查看详细信息与参数")

        self.verify_view = QPlainTextEdit()
        self.verify_view.setReadOnly(True)

        self.settings_widget = self._build_settings()

        right = QTabWidget()
        right.addTab(hist_widget, "修复历史")
        right.addTab(self.detail_view, "详情")
        right.addTab(self.verify_view, "复核报告")
        right.addTab(self.settings_widget, "检测设置")
        self.right_tabs = right
        dock_right = QDockWidget("审计与复核", self)
        dock_right.setWidget(right)
        dock_right.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock_right.setMinimumWidth(420)
        self.addDockWidget(Qt.RightDockWidgetArea, dock_right)

        self.defect_panel.selectionRequested.connect(self._focus_defect)
        self.defect_panel.defectActivated.connect(self._focus_defect)
        self.repair_panel.changed.connect(self._status)
        self.repair_panel.requestRepaint.connect(self._refresh_viewport)
        self.repair_panel.requestRefreshAnalysis.connect(self._after_repair)
        self.repair_panel.headless_auto_confirm = self.headless_auto_confirm

    def _build_settings(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.unit_combo = QComboBox()
        for u in UNIT_NAMES:
            self.unit_combo.addItem(f"{u}（{unit_label(u)}）", u)
        self.unit_combo.currentIndexChanged.connect(self._unit_changed)
        form.addRow("模型单位", self.unit_combo)

        self.dup_tol = QDoubleSpinBox()
        self.dup_tol.setDecimals(4)
        self.dup_tol.setRange(0.0, 1e6)
        self.dup_tol.setValue(0.01)
        self.dup_tol.setSuffix(" mm")
        form.addRow("重合顶点阈值", self.dup_tol)

        self.degen = QDoubleSpinBox()
        self.degen.setDecimals(5)
        self.degen.setRange(0.0, 1e3)
        self.degen.setValue(0.001)
        self.degen.setSuffix(" mm")
        form.addRow("退化面尺寸阈值", self.degen)

        self.aspect = QDoubleSpinBox()
        self.aspect.setDecimals(0)
        self.aspect.setRange(10.0, 1e9)
        self.aspect.setValue(1e4)
        form.addRow("狭长面长宽比上限", self.aspect)

        info = QLabel(
            "阈值以物理长度（mm）输入，按模型单位换算。\n"
            "退化面阈值按边长的平方参与面积判定。\n"
            "重合顶点只报告，不自动焊接。"
        )
        info.setWordWrap(True)
        form.addRow(info)
        btn = QPushButton("应用并重新检测")
        btn.clicked.connect(lambda: self._reanalyze(settings_changed=True))
        form.addRow(btn)
        return w

    # -- helpers --------------------------------------------------------
    def _status(self, text: str) -> None:
        self.statusBar().showMessage(text, 8000)

    def _settings_from_ui(self) -> AnalysisSettings:
        return AnalysisSettings(
            unit=self.unit_combo.currentData() or "mm",
            dup_tolerance_mm=self.dup_tol.value(),
            degenerate_edge_mm=self.degen.value(),
            aspect_max=self.aspect.value(),
        )

    def _update_actions(self) -> None:
        has = self.doc.session is not None
        self.act_export.setEnabled(has)
        self.act_audit.setEnabled(has)
        self.act_verify.setEnabled(has)
        self.btn_undo.setEnabled(has and self.doc.session.can_undo)
        self.btn_redo.setEnabled(has and self.doc.session.can_redo)
        self.btn_history_clear.setEnabled(has and bool(self.doc.session.history))

    # -- loading --------------------------------------------------------
    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入网格", "", FILE_FILTER)
        if path:
            self.load_path(path)

    def load_path(self, path: str, unit: str = "mm") -> None:
        try:
            self.doc.settings = AnalysisSettings(
                unit=unit,
                dup_tolerance_mm=self.dup_tol.value(),
                degenerate_edge_mm=self.degen.value(),
                aspect_max=self.aspect.value(),
            )
            self.doc.settings.unit = unit
            self.unit_combo.blockSignals(True)
            self.unit_combo.setCurrentIndex(max(0, self.unit_combo.findData(unit)))
            self.unit_combo.blockSignals(False)
            self.doc.load(path, unit)
        except (io_mesh.MeshIOError, ValueError, OSError) as exc:
            QMessageBox.critical(self, "导入失败", str(exc))
            return
        self._after_load(f"已导入 {Path(path).name}")

    def _load_benchmark(self, name: str) -> None:
        from ..benchmarks import generators

        try:
            mesh, meta = generators.build(name)
        except Exception as exc:  # benchmark construction is trusted but defensive
            QMessageBox.critical(self, "基准生成失败", str(exc))
            return
        self._reset_interaction_state()
        self.doc.close()
        from ..core.repair import RepairSession

        self.doc.session = RepairSession(mesh, unit=self.doc.settings.unit, source_name=f"benchmark:{name}")
        self.doc.source_path = None
        self.doc.notes = [meta.get("notes", "")]
        self.doc.reanalyze(update_baseline=True)
        self.viewport.set_mesh(self.doc.session.mesh)
        self._refresh_all()
        self._status(f"已加载基准：{name}；{meta.get('notes','')}")

    def _reset_interaction_state(self) -> None:
        """Drop selections and highlights that refer to the previous mesh."""
        self._loop_selection = None
        self._active_defect = None
        self.viewport.selected_faces.clear()
        self.viewport.selected_vertices.clear()
        self.viewport.set_defect_highlight()
        self.viewport.set_highlight_loops([])
        self.viewport._projected = None

    def _write_benchmarks(self) -> None:
        from ..benchmarks import generators

        directory = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not directory:
            return
        try:
            paths = generators.write_all(directory)
        except Exception as exc:
            QMessageBox.critical(self, "导出基准失败", str(exc))
            return
        QMessageBox.information(self, "基准已导出", "\n".join(str(p) for p in paths))

    def _after_load(self, message: str) -> None:
        self.viewport.set_mesh(self.doc.session.mesh)
        self._refresh_all()
        self._status(message + "；已检测并记录基线缺陷与指纹")

    # -- analysis / views ----------------------------------------------
    def _reanalyze(self, *, settings_changed: bool = False) -> None:
        if self.doc.session is None:
            return
        if settings_changed:
            self.doc.settings = self._settings_from_ui()
            self.defect_panel.set_unit(self.doc.settings.unit)
            self.repair_panel.set_context(self.doc.session, self.doc.current, self.doc.settings.unit)
        else:
            self.doc.settings = self._settings_from_ui()
        res = self.doc.reanalyze()
        self._refresh_all()
        self._status(f"重新检测完成：{res.total_instances()} 个缺陷实例")

    def _after_repair(self) -> None:
        if self.doc.session is None:
            return
        self.doc.reanalyze()
        self._refresh_all()
        self._status("修复已应用；请运行“复核修复结果”以确认拓扑与属性")

    def _refresh_all(self) -> None:
        session = self.doc.session
        result = self.doc.current
        self.defect_panel.set_unit(self.doc.settings.unit)
        self.defect_panel.set_context(session, result)
        self.defect_panel.set_pending(self.doc.pending_confirmations)
        self.repair_panel.set_context(session, result, self.doc.settings.unit)
        self.repair_panel.set_selection(
            self.viewport.selected_faces,
            self.viewport.selected_vertices,
            self._loop_selection,
        )
        self._refresh_history()
        self._refresh_viewport()
        self._update_actions()

    def _refresh_viewport(self) -> None:
        if self.doc.session is None:
            self.viewport.set_mesh(None)
            return
        self.viewport.refresh_mesh(self.doc.session.mesh)
        if self._active_defect is not None:
            kind, inst = self._active_defect
            faces = []
            verts = []
            if kind in ("degenerate_faces", "duplicate_faces", "winding_conflicts", "inward_components"):
                faces = inst.get("faces", [])
            elif kind == "nonmanifold_edges":
                verts = inst.get("vertices", [])
            elif kind in ("open_boundaries", "duplicate_vertices", "nonmanifold_vertices", "unreferenced_vertices"):
                verts = inst.get("vertices", [])
            self.viewport.set_defect_highlight(faces, verts)
            if kind == "open_boundaries":
                self.viewport.set_highlight_loops([inst.get("vertices", [])])
            else:
                self.viewport.set_highlight_loops([])
        # newly added faces from the last operation
        session = self.doc.session
        if session.history:
            self.viewport.set_added_faces(session.history[-1].added_faces)

    def _refresh_history(self) -> None:
        self.history_list.clear()
        session = self.doc.session
        if session is None:
            return
        for rec in session.history:
            item = QListWidgetItem(
                f"#{rec.index} {rec.label}  受影响面 {len(rec.affected_faces)} / 顶点 {len(rec.affected_vertices)}"
            )
            item.setData(Qt.UserRole, rec.index)
            self.history_list.addItem(item)
        if not session.history:
            self.history_list.addItem("（尚无修改）")
        self._update_actions()

    def _refresh_verify_view(self, report: dict | None = None) -> None:
        report = report or self.doc.last_verification
        if not report:
            self.verify_view.setPlainText("尚未复核。运行“复核修复结果”(F6)。")
            return
        self.verify_view.setPlainText(_format_verification(report, self.doc.settings.unit))

    # -- selection ------------------------------------------------------
    def _on_face_picked(self, face_id: int, additive: bool) -> None:
        if not additive:
            self.viewport.selected_faces.clear()
            self.viewport.selected_vertices.clear()
        self.viewport.selected_faces.add(int(face_id))
        self._loop_selection = None
        self._sync_selection()
        self._show_face_details(face_id)

    def _on_vertex_picked(self, vertex_id: int, additive: bool) -> None:
        if not additive:
            self.viewport.selected_vertices.clear()
            self.viewport.selected_faces.clear()
        self.viewport.selected_vertices.add(int(vertex_id))
        self._loop_selection = None
        self._sync_selection()
        self.detail_view.setPlainText(f"顶点 {vertex_id}: {self.doc.session.mesh.vertices[vertex_id]}")

    def _clear_selection(self) -> None:
        self.viewport.selected_faces.clear()
        self.viewport.selected_vertices.clear()
        self._loop_selection = None
        self._active_defect = None
        self.viewport.set_defect_highlight()
        self.viewport.set_highlight_loops([])
        self._sync_selection()
        self.detail_view.clear()

    def _sync_selection(self) -> None:
        self.viewport.update()
        self.repair_panel.set_selection(
            self.viewport.selected_faces,
            self.viewport.selected_vertices,
            self._loop_selection,
        )

    def _focus_defect(self, kind: str, index: int) -> None:
        if self.doc.session is None or self.doc.current is None:
            return
        group = self.doc.current.groups.get(kind)
        if group is None or index >= len(group.instances):
            return
        self.defect_panel.select_defect(kind, index)
        inst = group.instances[index]
        self._active_defect = (kind, inst)
        faces = [int(f) for f in inst.get("faces", [])]
        verts = [int(v) for v in inst.get("vertices", [])]
        if kind in ("duplicate_vertices",):
            self.viewport.selected_vertices = set(verts)
            self.viewport.show_vertices = True
            self.act_verts.setChecked(True)
            self.viewport.selected_faces.clear()
        elif kind == "open_boundaries":
            self.viewport.selected_faces.clear()
            self.viewport.selected_vertices = set(verts)
            self._loop_selection = list(verts)
            self.viewport.set_highlight_loops([verts])
        else:
            self.viewport.selected_faces = set(faces)
            self.viewport.selected_vertices = set(verts)
            self._loop_selection = None
            self.viewport.set_highlight_loops([])
        self.viewport.set_defect_highlight(faces, verts)
        self.repair_panel.set_selection(
            self.viewport.selected_faces, self.viewport.selected_vertices, self._loop_selection
        )
        self.viewport.update()
        self._show_defect_details(kind, inst)
        self._refresh_history()
        self._update_actions()

    def _show_defect_details(self, kind: str, inst: dict) -> None:
        session = self.doc.session
        unit = self.doc.settings.unit
        lines = [f"缺陷类型: {kind}", f"原因: {inst.get('reason','')}"]
        if "faces" in inst:
            lines.append(f"相关面: {inst.get('faces')}")
        if "vertices" in inst:
            vs = inst["vertices"]
            lines.append(f"相关顶点 ({len(vs)}): {vs[:40]}{' …' if len(vs)>40 else ''}")
            if session is not None and vs:
                pts = session.mesh.vertices[vs]
                lines.append(f"坐标范围 min={pts.min(axis=0)} max={pts.max(axis=0)}")
        if kind == "open_boundaries":
            lines.append(f"周长: {format_mm(inst.get('perimeter', 0.0), unit)}")
            lines.append(f"最大段长: {format_mm(inst.get('max_spacing', 0.0), unit)}")
            lines.append(f"闭合: {inst.get('closed')}")
            lines.append("提示：设计开口与破损开口外观相同；补洞前请先“检查选中的开口环”。")
        if kind == "duplicate_vertices":
            lines.append("提示：默认不焊接。若这些点属于薄壁两侧，焊接会粘合两层。")
        if kind == "inward_components":
            lines.append("提示：仅封闭组件可由体积判断方向；内腔/模具可能本就朝内。")
        self.detail_view.setPlainText("\n".join(lines))

    def _show_face_details(self, face_id: int) -> None:
        session = self.doc.session
        if session is None:
            return
        mesh = session.mesh
        if face_id >= mesh.face_count:
            return
        tri = mesh.faces[face_id]
        normal = mesh.face_normals()[face_id]
        centroid = mesh.face_centroids()[face_id]
        area = mesh.triangle_areas()[face_id]
        lines = [
            f"面 {face_id}: 顶点 {list(int(x) for x in tri)}",
            f"质心 {np_str(centroid)}",
            f"法线 {np_str(normal)}",
            f"面积 {format_mm(float(area), self.doc.settings.unit)}²",
        ]
        self.detail_view.setPlainText("\n".join(lines))

    # -- undo/redo ------------------------------------------------------
    def _undo(self) -> None:
        if self.doc.session is None:
            return
        rec = self.doc.session.undo()
        if rec is None:
            return
        self._after_repair()
        self._status(f"已撤销 #{rec.index} {rec.label}")

    def _redo(self) -> None:
        if self.doc.session is None:
            return
        rec = self.doc.session.redo()
        if rec is None:
            return
        self._after_repair()
        self._status(f"已重做 #{rec.index} {rec.label}")

    def _clear_history(self) -> None:
        if self.doc.session is None:
            return
        ans = QMessageBox.question(
            self, "清空历史",
            "清空后无法再撤销此前的修改，但当前网格保留。是否继续？",
        )
        if ans != QMessageBox.Yes:
            return
        self.doc.session.history.clear()
        self.doc.session.redo_stack.clear()
        self._refresh_all()
        self._status("历史已清空；撤销不再可用")

    def _on_history_double_click(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.UserRole)
        if idx is None or self.doc.session is None:
            return
        session = self.doc.session
        rec = next((r for r in session.history if r.index == idx), None)
        if rec is None:
            return
        ans = QMessageBox.question(
            self, "回退到该步之前",
            f"将撤销 #{idx} 及之后共 {sum(1 for r in session.history if r.index >= idx)} 步。继续？",
        )
        if ans != QMessageBox.Yes:
            return
        n = session.undo_until(idx)
        self._after_repair()
        self._status(f"已回退 {n} 步")

    # -- verification ---------------------------------------------------
    def _verify(self) -> None:
        if self.doc.session is None:
            return
        try:
            report = self.doc.verify()
        except Exception as exc:
            QMessageBox.critical(self, "复核失败", str(exc))
            return
        self._refresh_verify_view(report)
        self.defect_panel.set_pending(self.doc.pending_confirmations)
        self.right_tabs.setCurrentWidget(self.verify_view)
        self._status("复核完成：" + ("无新增/遗留问题" if report["clean"] else "存在遗留或新增问题"))

    # -- export ---------------------------------------------------------
    def _export(self) -> None:
        if self.doc.session is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出修复结果", "", "STL (*.stl);;OBJ (*.obj);;PLY (*.ply)"
        )
        if not path:
            return
        try:
            info = self.doc.export(path, comment="导出自动生成")
        except (io_mesh.MeshIOError, OSError, ValueError) as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self._status(f"已导出网格与审计：{path} / {info['sidecar']}")

    def _export_audit(self) -> None:
        if self.doc.session is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出审计 JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            p = self.doc.save_audit_json(path)
        except OSError as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        self._status(f"审计已导出：{p}")

    def _export_view(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存视图截图", "", "PNG (*.png)")
        if not path:
            return
        if self.viewport.grab_png(path):
            self._status(f"截图已保存：{path}")
        else:
            QMessageBox.warning(self, "保存失败", "无法写入截图")

    # -- misc -----------------------------------------------------------
    def _unit_changed(self) -> None:
        if self.doc.session is None:
            return
        self.doc.settings = self._settings_from_ui()
        self._reanalyze(settings_changed=True)

    def _toggle_edges(self, on: bool) -> None:
        self.viewport.show_edges = on
        self.viewport.update()

    def _toggle_vertices(self, on: bool) -> None:
        self.viewport.show_vertices = on
        self.viewport.update()

    def closeEvent(self, event) -> None:
        if getattr(self, "_suppress_exit_prompt", False):
            event.accept()
            return
        if self.doc.session is not None and self.doc.session.history:
            ans = QMessageBox.question(
                self, "退出",
                "存在未导出的修复历史。退出前是否导出审计 JSON？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if ans == QMessageBox.Cancel:
                event.ignore()
                return
            if ans == QMessageBox.Save:
                self._export_audit()
        event.accept()


def np_str(v) -> str:
    return "(" + ", ".join(f"{float(x):.4g}" for x in v) + ")"


def _format_verification(report: dict, unit: str) -> str:
    lines = ["== 复核结果 ==", ""]
    lines.append("—— 缺陷类别 ——")
    for kind, info in report["kinds"].items():
        state = "已完全解决" if info["fully_resolved"] else "仍有遗留"
        if info["new"]:
            state = "出现新增缺陷"
        lines.append(
            f"{kind:22s} 前 {info['before']:>4d}  后 {info['after']:>4d}  "
            f"移除 {info['removed']:>3d}  遗留 {info['remaining']:>3d}  "
            f"新增 {info['new']:>3d}   [{state}]"
        )
    lines.append("")
    lines.append("—— 几何变化 ——")
    g = report["geometry"]
    lines.append(f"面积:  {g['area_before']:.6g} -> {g['area_after']:.6g}  (Δ {g['area_delta']:+.6g})")
    lines.append(f"体积:  {g['volume_before']:.6g} -> {g['volume_after']:.6g}  (Δ {g['volume_delta']:+.6g})")
    lines.append(f"包围盒变化: {[round(x, 6) for x in g['bbox_delta']]}")
    lines.append(f"对角线: {g['diagonal_before']:.6g} -> {g['diagonal_after']:.6g}")
    lines.append("")
    lines.append("—— 拓扑指标 ——")
    for k, d in report["metrics"].items():
        lines.append(f"{k:22s} {d['before']} -> {d['after']}  (Δ {d['delta']:+})")
    if report["problems"]:
        lines.append("")
        lines.append("—— 仍存在的问题 ——")
        for p in report["problems"]:
            lines.append("• " + p)
    else:
        lines.append("")
        lines.append("未发现新增或遗留问题；但这只是对已启用检测项的复核。")
    lines.append("")
    lines.append(report.get("note", ""))
    return "\n".join(lines)
