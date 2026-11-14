#!/usr/bin/env python3
"""Offscreen UI drive: pick a bad face, repair via the kernel, screenshot.

Run:  LD_LIBRARY_PATH=... QT_QPA_PLATFORM=offscreen python3 scripts/window_drive.py

Loads the non-manifold junction benchmark into the real MainWindow, selects
the offending faces on the canvas exactly as a mouse pick would, runs the
kernel through the repair panel, and writes before/after screenshots plus a
log of the comparison tab.  This is the evidence that the GUI entry works on
a real mesh, not just the experiment text.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

from mesh_repair.core.topology import face_components
from mesh_repair.ui.main_window import MainWindow


def main() -> int:
    out = ROOT / "artifacts" / "strategy"
    out.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.resize(1440, 900)
    win.show()
    app.processEvents()

    win._load_benchmark("nonmanifold_junction")
    for _ in range(3):
        app.processEvents()
    vp = win.viewport
    vp.resize(900, 700)
    vp.frame_model()
    for _ in range(3):
        app.processEvents()

    mesh = win.doc.session.mesh
    inst = win.doc.current.groups["nonmanifold_edges"].instances[0]
    faces = [int(f) for f in inst["faces"]][:2]
    comp_before = face_components(mesh.faces, mesh.vertex_count)[1]

    vp.selected_faces = set(faces)
    win.repair_panel.set_selection(faces, set(), None)
    vp.update()
    app.processEvents()
    vp.grab_png(str(out / "20_pick_before.png"))

    win.repair_panel._repair_picked_face()
    for _ in range(4):
        app.processEvents()
    vp.grab_png(str(out / "21_pick_after.png"))

    comp_after = face_components(
        win.doc.session.mesh.faces, win.doc.session.mesh.vertex_count
    )[1]
    import json

    payload = {
        "benchmark": "nonmanifold_junction",
        "selected_faces": faces,
        "faces_before": mesh.face_count,
        "faces_after": win.doc.session.mesh.face_count,
        "components_before": comp_before,
        "components_after": comp_after,
        "history": [r.label for r in win.doc.session.history],
        "compare_tab": win.compare_view.toPlainText(),
        "screenshots": ["20_pick_before.png", "21_pick_after.png"],
    }
    (out / "window_run.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    win._suppress_exit_prompt = True
    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
