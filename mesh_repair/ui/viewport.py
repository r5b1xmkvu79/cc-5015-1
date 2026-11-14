"""Software-rendered 3D viewport.

Uses QPainter rather than OpenGL so the tool runs in headless/offscreen
environments (CI, remote sessions) without a GPU.  The renderer is a simple
painter's-algorithm triangle rasteriser with flat shading; enough to locate
defects visually and to screenshot for verification.

Interaction:

* left drag   orbit
* right drag  pan
* wheel       zoom
* click       pick face / vertex under the cursor
* B           box select mode (drag a rectangle)
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from ..core.mesh import Mesh

BG_TOP = QColor(34, 38, 46)
BG_BOTTOM = QColor(18, 20, 25)
FACE_BASE = np.array([0.62, 0.66, 0.72])
FACE_SELECTED = np.array([1.0, 0.45, 0.15])
FACE_DEFECT = np.array([0.85, 0.15, 0.25])
FACE_ADDED = np.array([0.25, 0.75, 0.45])
EDGE_COLOR = QColor(20, 22, 28, 150)
VERTEX_COLOR = QColor(120, 180, 255)
VERTEX_SELECTED = QColor(255, 140, 30)
LOOP_COLOR = QColor(255, 210, 60)


class Camera:
    def __init__(self) -> None:
        self.yaw = 35.0
        self.pitch = 22.0
        self.zoom = 1.0
        self.pan = np.zeros(2)
        self.distance = 3.0

    def rotation(self) -> np.ndarray:
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        return rx @ ry


class MeshViewport(QWidget):
    facePicked = Signal(int, bool)          # face id, additive
    vertexPicked = Signal(int, bool)
    selectionCleared = Signal()
    cameraChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 320)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.camera = Camera()

        self.mesh: Mesh | None = None
        self.selected_faces: set[int] = set()
        self.selected_vertices: set[int] = set()
        self.defect_faces: set[int] = set()
        self.defect_vertices: set[int] = set()
        self.added_faces: set[int] = set()
        self.highlight_loops: list[list[int]] = []
        self.show_edges = True
        self.show_vertices = False
        self.point_size = 4
        self._rot_center = np.zeros(3)
        self._scale = 1.0

        self._drag_button = None
        self._last_pos = QPoint()
        self._box_mode = False
        self._box_start: QPoint | None = None
        self._box_current: QPoint | None = None

        self._projected: np.ndarray | None = None
        self._face_depths: np.ndarray | None = None
        self._pickable = True

    # -- data -----------------------------------------------------------
    def set_mesh(self, mesh: Mesh | None, *, recenter: bool = True) -> None:
        self.mesh = mesh
        self.selected_faces.clear()
        self.selected_vertices.clear()
        self.defect_faces.clear()
        self.defect_vertices.clear()
        self.added_faces.clear()
        self.highlight_loops.clear()
        self._projected = None
        if mesh is not None and recenter:
            self.frame_model()
        self.update()

    def frame_model(self) -> None:
        if self.mesh is None or self.mesh.vertex_count == 0:
            return
        b = self.mesh.bounds()
        self._rot_center = (b[0] + b[1]) / 2.0
        diag = max(self.mesh.diagonal(), 1e-9)
        self._scale = 1.0 / diag
        self.camera.pan[:] = 0
        self.update()

    def refresh_mesh(self, mesh: Mesh) -> None:
        """Replace geometry but keep selection where still valid."""
        self.mesh = mesh
        self.selected_faces = {f for f in self.selected_faces if f < mesh.face_count}
        self.selected_vertices = {v for v in self.selected_vertices if v < mesh.vertex_count}
        self.added_faces = {f for f in self.added_faces if f < mesh.face_count}
        self._projected = None
        self.update()

    # -- projection -----------------------------------------------------
    def _project_points(self) -> np.ndarray | None:
        if self.mesh is None or self.mesh.vertex_count == 0:
            return None
        v = (self.mesh.vertices - self._rot_center) * self._scale
        r = self.camera.rotation()
        p = v @ r.T
        # Camera sits on +Z looking back; larger p_z means closer to viewer.
        depth = p[:, 2] + self.camera.distance
        depth = np.where(depth < 1e-4, 1e-4, depth)
        w, h = max(self.width(), 1), max(self.height(), 1)
        f = min(w, h) * 3.2 * self.camera.zoom
        x = p[:, 0] * f / depth + w / 2 + self.camera.pan[0]
        y = -p[:, 1] * f / depth + h / 2 + self.camera.pan[1]
        return np.stack([x, y], axis=1)

    def _zdepth(self) -> np.ndarray:
        v = (self.mesh.vertices - self._rot_center) * self._scale
        r = self.camera.rotation()
        return (v @ r.T)[:, 2]

    def _face_depth(self, projected: np.ndarray, faces: np.ndarray) -> np.ndarray:
        # larger value = closer to the camera
        return self._zdepth()[faces].mean(axis=1)

    # -- painting -------------------------------------------------------
    def paintEvent(self, event) -> None:
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing, True)
        w, h = self.width(), self.height()
        grad = QColor(BG_TOP)
        qp.fillRect(0, 0, w, h, grad)
        qp.fillRect(0, 0, w, h // 2, BG_TOP)

        if self.mesh is None or self.mesh.face_count == 0:
            self._draw_placeholder(qp)
            qp.end()
            return

        pts = self._project_points()
        self._projected = pts
        faces = self.mesh.faces
        depth = self._face_depth(pts, faces)
        order = np.argsort(depth)  # ascending depth = far first

        normals = self.mesh.face_normals()
        rot = self.camera.rotation()
        view_n = rot @ np.array([0.0, 0.0, 1.0])
        light = -view_n + np.array([0.3, 0.4, 0.2])
        light = light / (np.linalg.norm(light) or 1.0)

        shading = np.clip(normals @ light, 0.0, 1.0)
        colors = FACE_BASE[None, :] * (0.35 + 0.65 * shading[:, None])

        face_polys = []
        for fid in order:
            tri = faces[fid]
            poly = QPolygonF([QPointF(float(pts[v, 0]), float(pts[v, 1])) for v in tri])
            base = colors[fid]
            if fid in self.added_faces:
                base = base * 0.35 + FACE_ADDED * 0.65
            if fid in self.defect_faces:
                base = base * 0.55 + FACE_DEFECT * 0.45
            if fid in self.selected_faces:
                base = base * 0.25 + FACE_SELECTED * 0.75
            qp.setBrush(QColor.fromRgbF(float(np.clip(base[0], 0, 1)),
                                        float(np.clip(base[1], 0, 1)),
                                        float(np.clip(base[2], 0, 1))))
            if self.show_edges:
                qp.setPen(QPen(EDGE_COLOR, 1.0))
            else:
                qp.setPen(Qt.NoPen)
            qp.drawPolygon(poly)
            face_polys.append(fid)

        if self.highlight_loops:
            qp.setPen(QPen(LOOP_COLOR, 2.5))
            qp.setBrush(Qt.NoBrush)
            for loop in self.highlight_loops:
                poly = QPolygonF([QPointF(float(pts[v, 0]), float(pts[v, 1])) for v in loop])
                qp.drawPolygon(poly)

        if self.show_vertices:
            qp.setPen(Qt.NoPen)
            for vid, p in enumerate(pts):
                if vid in self.selected_vertices:
                    qp.setBrush(VERTEX_SELECTED)
                    r = self.point_size + 2
                elif vid in self.defect_vertices:
                    qp.setBrush(QColor(230, 60, 70))
                    r = self.point_size
                else:
                    qp.setBrush(VERTEX_COLOR)
                    r = self.point_size - 1
                qp.drawEllipse(QPointF(float(p[0]), float(p[1])), r, r)

        self._draw_overlay(qp)
        if self._box_start is not None and self._box_current is not None:
            rect = QRect(self._box_start, self._box_current).normalized()
            qp.setPen(QPen(QColor(255, 200, 80), 1, Qt.DashLine))
            qp.setBrush(QColor(255, 200, 80, 30))
            qp.drawRect(rect)
        qp.end()

    def _draw_overlay(self, qp: QPainter) -> None:
        if self.mesh is None:
            return
        qp.setFont(QFont("Sans", 9))
        qp.setPen(QColor(210, 215, 225))
        lines = [
            f"面 {self.mesh.face_count}  顶点 {self.mesh.vertex_count}  "
            f"组件 {self._component_count()}",
            f"选中面 {len(self.selected_faces)}  选中顶点 {len(self.selected_vertices)}",
        ]
        y = 16
        for ln in lines:
            qp.drawText(10, y, ln)
            y += 15

    def _component_count(self) -> int:
        if self.mesh is None:
            return 0
        from ..core.topology import face_components
        try:
            return face_components(self.mesh.faces, self.mesh.vertex_count)[1]
        except Exception:
            return 0

    def _draw_placeholder(self, qp: QPainter) -> None:
        qp.setPen(QColor(120, 128, 140))
        qp.setFont(QFont("Sans", 12))
        qp.drawText(self.rect(), Qt.AlignCenter, "导入网格以开始\n支持 STL / OBJ / PLY")

    # -- interaction ----------------------------------------------------
    def mousePressEvent(self, event) -> None:
        self._last_pos = event.position().toPoint()
        if event.button() == Qt.LeftButton:
            if self._box_mode:
                self._box_start = self._last_pos
                self._box_current = self._last_pos
                self.update()
                return
            self._drag_button = Qt.LeftButton
        elif event.button() == Qt.RightButton:
            self._drag_button = Qt.RightButton
        elif event.button() == Qt.MiddleButton:
            self._drag_button = Qt.MiddleButton

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._box_start is not None:
            self._box_current = pos
            self.update()
            return
        if self._drag_button is None:
            return
        dx = pos.x() - self._last_pos.x()
        dy = pos.y() - self._last_pos.y()
        if self._drag_button == Qt.LeftButton:
            self.camera.yaw += dx * 0.5
            self.camera.pitch = float(np.clip(self.camera.pitch + dy * 0.5, -89, 89))
        elif self._drag_button in (Qt.RightButton, Qt.MiddleButton):
            self.camera.pan[0] += dx
            self.camera.pan[1] += dy
        self._last_pos = pos
        self._projected = None
        self.update()
        self.cameraChanged.emit()

    def mouseReleaseEvent(self, event) -> None:
        if self._box_start is not None and event.button() == Qt.LeftButton:
            self._apply_box_selection(event.position().toPoint())
            self._box_start = None
            self._box_current = None
            self.update()
            return
        if self._drag_button == Qt.LeftButton:
            moved = (event.position().toPoint() - self._last_pos).manhattanLength()
            if moved < 3:
                self._pick_at(event.position().toPoint(),
                              additive=bool(event.modifiers() & Qt.ShiftModifier))
        self._drag_button = None

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        factor = math.exp(delta * 0.0012)
        self.camera.zoom = float(np.clip(self.camera.zoom * factor, 0.05, 40.0))
        self._projected = None
        self.update()
        self.cameraChanged.emit()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_B:
            self._box_mode = not self._box_mode
        elif event.key() == Qt.Key_Escape:
            self.selectionCleared.emit()
        elif event.key() == Qt.Key_F:
            self.frame_model()
        else:
            super().keyPressEvent(event)

    # -- picking --------------------------------------------------------
    def _pick_at(self, pos: QPoint, additive: bool) -> None:
        if self.mesh is None or self._projected is None:
            return
        if self.show_vertices:
            v = self._nearest_vertex(pos, max_dist=10.0)
            if v is not None:
                self.vertexPicked.emit(v, additive)
                return
        f = self._face_at_point(pos)
        if f is not None:
            self.facePicked.emit(f, additive)
        else:
            self.selectionCleared.emit()

    def _nearest_vertex(self, pos: QPoint, max_dist: float) -> int | None:
        pts = self._projected if self._projected is not None else self._project_points()
        if pts is None:
            return None
        d = np.hypot(pts[:, 0] - pos.x(), pts[:, 1] - pos.y())
        idx = int(np.argmin(d))
        return idx if d[idx] <= max_dist else None

    def _face_at_point(self, pos: QPoint) -> int | None:
        """Front-most triangle whose screen polygon contains pos."""
        pts = self._projected
        if pts is None:
            pts = self._project_points()
        if pts is None or self.mesh is None:
            return None
        faces = self.mesh.faces
        tri = pts[faces]
        x, y = float(pos.x()), float(pos.y())
        a = tri[:, 0]
        b = tri[:, 1]
        c = tri[:, 2]

        d1 = (a[:, 0] - x) * (b[:, 1] - y) - (b[:, 0] - x) * (a[:, 1] - y)
        d2 = (b[:, 0] - x) * (c[:, 1] - y) - (c[:, 0] - x) * (b[:, 1] - y)
        d3 = (c[:, 0] - x) * (a[:, 1] - y) - (a[:, 0] - x) * (c[:, 1] - y)
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        inside = ~(has_neg & has_pos)
        if not np.any(inside):
            return None
        depth = self._face_depth(pts, faces)
        candidates = np.nonzero(inside)[0]
        return int(candidates[np.argmax(depth[candidates])])

    def _apply_box_selection(self, end: QPoint) -> None:
        if self._box_start is None or self._projected is None or self.mesh is None:
            return
        rect = QRect(self._box_start, end).normalized()
        pts = self._projected
        if self.show_vertices:
            inside = [
                i for i, p in enumerate(pts)
                if rect.contains(int(p[0]), int(p[1]))
            ]
            self.selected_vertices = set(inside)
            self.update()
            return
        tri = pts[self.mesh.faces]
        hit = []
        for fid, poly in enumerate(tri):
            cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
            if rect.contains(int(cx), int(cy)):
                hit.append(fid)
        self.selected_faces = set(hit)
        self.update()

    # -- external selection --------------------------------------------
    def set_face_selection(self, faces) -> None:
        self.selected_faces = set(int(f) for f in faces)
        self.update()

    def set_vertex_selection(self, verts) -> None:
        self.selected_vertices = set(int(v) for v in verts)
        self.update()

    def set_defect_highlight(self, faces=(), vertices=()) -> None:
        self.defect_faces = set(int(f) for f in faces)
        self.defect_vertices = set(int(v) for v in vertices)
        self.update()

    def set_added_faces(self, faces) -> None:
        self.added_faces = set(int(f) for f in faces)
        self.update()

    def set_highlight_loops(self, loops) -> None:
        self.highlight_loops = [[int(v) for v in l] for l in loops]
        self.update()

    def grab_png(self, path: str) -> bool:
        img: QImage = self.grab().toImage()
        return bool(img.save(path))
