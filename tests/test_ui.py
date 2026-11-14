"""Headless UI integration test.

Drives the real widgets (offscreen Qt) through the full workflow:

1. load a benchmark into the window,
2. click a defect in the list to select it on the canvas,
3. apply a permitted repair through the repair panel,
4. undo and redo,
5. verify,
6. export and reload, checking provenance and baseline fingerprints.

Run with:  python3 -m unittest tests.test_ui -v
(set LD_LIBRARY_PATH / QT_QPA_PLATFORM per scripts/run.sh on headless hosts)
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_QT_IMPORT_ERROR = None
try:
    import PySide6.QtWidgets  # noqa: F401
    _ = PySide6.QtWidgets
except Exception as exc:  # missing libEGL/libdbus etc. on a bare host
    _QT_IMPORT_ERROR = exc


def _app():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@unittest.skipIf(_QT_IMPORT_ERROR is not None,
                 f"Qt 运行时不可用: {_QT_IMPORT_ERROR}")
class TestUIFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _app()

    def setUp(self):
        from mesh_repair.ui.main_window import MainWindow

        self.win = MainWindow()
        self.win.show()
        self.app.processEvents()

    def tearDown(self):
        # closeEvent offers to export the audit when history exists; in tests
        # we discard rather than opening a modal dialog.
        self.win._suppress_exit_prompt = True
        self.win.close()
        self.app.processEvents()

    def _pump(self):
        for _ in range(3):
            self.app.processEvents()

    def test_benchmark_load_and_canvas_present(self):
        self.win._load_benchmark("duplicate_faces")
        self._pump()
        self.assertIsNotNone(self.win.doc.session)
        self.assertEqual(self.win.doc.current.total_instances() > 0, True)
        # list has headers + instances
        self.assertGreater(self.win.defect_panel.tree.count(), 0)

    def test_list_selection_drives_canvas_selection(self):
        self.win._load_benchmark("duplicate_faces")
        self._pump()
        tree = self.win.defect_panel.tree
        target = None
        from PySide6.QtCore import Qt

        for i in range(tree.count()):
            item = tree.item(i)
            data = item.data(Qt.UserRole)
            if data and data[0] == "duplicate_faces":
                target = item
                break
        self.assertIsNotNone(target)
        tree.setCurrentItem(target)
        self.win.defect_panel._on_double(target)
        self._pump()
        self.assertTrue(self.win.viewport.selected_faces)
        self.assertEqual(
            set(self.win.viewport.selected_faces),
            set(self.win.doc.current.groups["duplicate_faces"].instances[0]["faces"]),
        )

    def test_repair_panel_delete_duplicate_group(self):
        self.win._load_benchmark("duplicate_faces")
        self._pump()
        from PySide6.QtCore import Qt

        tree = self.win.defect_panel.tree
        for i in range(tree.count()):
            item = tree.item(i)
            data = item.data(Qt.UserRole)
            if data and data[0] == "duplicate_faces":
                tree.setCurrentItem(item)
                break
        faces_before = self.win.doc.session.mesh.face_count
        self.win.repair_panel._delete_duplicate_group()
        self._pump()
        self.assertLess(self.win.doc.session.mesh.face_count, faces_before)
        self.assertEqual(len(self.win.doc.session.history), 1)
        # history list reflects the operation
        self.assertIn("#1", self.win.history_list.item(0).text())

    def test_undo_redo_through_window(self):
        self.win._load_benchmark("duplicate_faces")
        self._pump()
        from PySide6.QtCore import Qt

        tree = self.win.defect_panel.tree
        for i in range(tree.count()):
            item = tree.item(i)
            data = item.data(Qt.UserRole)
            if data and data[0] == "duplicate_faces":
                tree.setCurrentItem(item)
                break
        original_faces = self.win.doc.session.mesh.face_count
        self.win.repair_panel._delete_duplicate_group()
        self._pump()
        self.win._undo()
        self._pump()
        self.assertEqual(self.win.doc.session.mesh.face_count, original_faces)
        self.win._redo()
        self._pump()
        self.assertLess(self.win.doc.session.mesh.face_count, original_faces)

    def test_hole_limits_enforced_in_panel(self):
        self.win._load_benchmark("holed_plate")
        self._pump()
        from PySide6.QtCore import Qt

        tree = self.win.defect_panel.tree
        big_item = small_item = None
        result = self.win.doc.current
        insts = result.groups["open_boundaries"].instances
        perims = [inst["perimeter"] for inst in insts]
        small_idx = int(np.argmin(perims))
        big_idx = int(np.argmax(perims))
        seen = 0
        for i in range(tree.count()):
            item = tree.item(i)
            data = item.data(Qt.UserRole)
            if data and data[0] == "open_boundaries":
                if seen == small_idx:
                    small_item = item
                if seen == big_idx:
                    big_item = item
                seen += 1
        self.assertIsNotNone(small_item)
        self.assertIsNotNone(big_item)
        # check the big loop via the panel's validator (no confirmation dialog)
        self.win._focus_defect("open_boundaries", big_idx)
        from mesh_repair.core import holes

        chk = holes.check_loop(self.win.doc.session.mesh,
                               self.win.repair_panel._selected_loop,
                               self.win.repair_panel._limits())
        self.assertFalse(chk["ok"])
        self.win._focus_defect("open_boundaries", small_idx)
        chk2 = holes.check_loop(self.win.doc.session.mesh,
                                self.win.repair_panel._selected_loop,
                                self.win.repair_panel._limits())
        self.assertTrue(chk2["ok"])

    def test_face_click_picks_geometry(self):
        self.win._load_benchmark("thin_wall")
        self._pump()
        vp = self.win.viewport
        vp.resize(800, 600)
        self._pump()
        pts = vp._project_points()
        self.assertIsNotNone(pts)
        # pick the centroid of face 0's projected triangle
        from PySide6.QtCore import QPoint

        tri = pts[vp.mesh.faces[5]]
        center = QPoint(int(tri[:, 0].mean()), int(tri[:, 1].mean()))
        fid = vp._face_at_point(center)
        self.assertIsNotNone(fid)
        vp._pick_at(center, additive=False)
        self._pump()
        self.assertIn(int(fid), vp.selected_faces)

    def test_verify_and_export_reload_provenance(self):
        self.win._load_benchmark("duplicate_faces")
        self._pump()
        from PySide6.QtCore import Qt

        tree = self.win.defect_panel.tree
        # delete duplicate groups one at a time; the list rebuilds after each
        guard = 0
        while guard < 20:
            guard += 1
            target = None
            for i in range(tree.count()):
                item = tree.item(i)
                if item is None:
                    continue
                data = item.data(Qt.UserRole)
                if data and data[0] == "duplicate_faces":
                    target = item
                    break
            if target is None:
                break
            tree.setCurrentItem(target)
            self.win.repair_panel._delete_duplicate_group()
            self._pump()
        self.win._verify()
        self._pump()
        report = self.win.doc.last_verification
        self.assertIsNotNone(report)
        self.assertTrue(report["kinds"]["duplicate_faces"]["fully_resolved"])
        text = self.win.verify_view.toPlainText()
        self.assertIn("复核结果", text)

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "repaired.stl"
            self.win.doc.export(path)
            self.assertTrue(path.exists())
            from mesh_repair.core import io_mesh

            prov = io_mesh.read_embedded_provenance(path)
            self.assertEqual(prov, self.win.doc.provenance_id)
            sidecar = io_mesh.read_sidecar(path)
            self.assertEqual(sidecar["provenance_id"], self.win.doc.provenance_id)
            # reload through the window and ensure baseline defects are traceable
            prior_id = self.win.doc.provenance_id
            self.win.load_path(str(path))
            self._pump()
            notes = "\n".join(self.win.doc.notes)
            self.assertIn(prior_id, notes)
            self.assertEqual(self.win.doc.origin_provenance, prior_id)
            check = self.win.doc.check_origin_defects()
            self.assertTrue(check["available"])
            self.assertIn("duplicate_faces", check["kinds"])

    def test_default_unit_switch_reanalyzes(self):
        self.win._load_benchmark("duplicate_faces")
        self._pump()
        before = self.win.doc.current.metrics["dup_tolerance"]
        idx = self.win.unit_combo.findData("in")
        self.win.unit_combo.setCurrentIndex(idx)
        self._pump()
        after = self.win.doc.current.metrics["dup_tolerance"]
        self.assertAlmostEqual(after, before / 25.4)

    def test_pending_confirmations_visible_in_panel(self):
        self.win._load_benchmark("open_shell")
        self._pump()
        self.assertTrue(self.win.doc.pending_confirmations)
        text = self.win.defect_panel.warn_box.toPlainText()
        self.assertIn("待确认", text)
        self.assertIn("open_boundaries", text)
        self.assertIn("不封闭", text)
        # and it survives a verification pass
        self.win._verify()
        self._pump()
        self.assertIn("orientation_unresolved",
                      self.win.defect_panel.warn_box.toPlainText())


if __name__ == "__main__":
    unittest.main(verbosity=2)
