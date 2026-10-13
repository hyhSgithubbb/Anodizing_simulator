"""3D 보기 — 마우스로 회전·이동·확대.

    왼쪽 드래그        : 회전 (궤도)
    오른쪽/가운데 드래그 : 이동 (Ctrl + 왼쪽 드래그도 이동)
    휠               : 확대/축소

PyOpenGL 이 없거나 OpenGL 을 쓸 수 없는 환경에서는 안내 문구로 대체된다.
"""

from __future__ import annotations

import math

import numpy as np
from pyqtgraph import Vector
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QVector4D
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ...core import units as u
from ...core.geometry import Geometry
from .. import style
from .colormap import ColorLegend, colors

try:  # PyOpenGL 미설치 시에도 나머지 GUI 는 동작해야 한다
    import pyqtgraph.opengl as gl
    _GL_ERROR = None
except Exception as exc:  # pragma: no cover - 환경 의존
    gl = None
    _GL_ERROR = exc

# 시점 프리셋: (이름, 고도각, 방위각). 방위각 −90° 는 −y 쪽에서 +y 를 보는 측면도 방향
VIEW_PRESETS = (("등각", 22.0, -62.0), ("측면 (2D와 같은 방향)", 0.0, -90.0), ("끝면", 0.0, 0.0), ("위", 90.0, -90.0))


def _rgba(color, alpha=1.0):
    c = QColor(color)
    return (c.redF(), c.greenF(), c.blueF(), alpha)


def box_mesh(x0, x1, y0, y1, z0, z1):
    """바깥쪽을 향하는 법선을 갖는 직육면체 메시."""
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return gl.MeshData(vertexes=v, faces=f)


def box_edges(x0, x1, y0, y1, z0, z1, top=True):
    """직육면체 모서리 선분 (mode='lines' 용 점 쌍). top=False 면 윗면 테두리 제외 (열린 용기)."""
    c = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=float)
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7)]
    if top:
        pairs += [(4, 5), (5, 6), (6, 7), (7, 4)]
    return np.array([c[i] for pair in pairs for i in pair])


def anode_surface_quads(mesh, box, eps: float = 0.0):
    """양극 표면 요소를 3D 사각형으로 (SI 단위, box 는 PlateElectrode.box 결과).

    돌려주는 값: (꼭짓점 (Q, 4, 3), 각 사각형이 속한 요소 번호 (Q,)).
    −x/+x 면 요소는 같은 높이 구간의 y 끝면 절반(두께 방향 반쪽)도 함께 칠한다 (요소 면적 정의와 같음).
    eps 만큼 바깥으로 띄워 판 메시와 겹쳐 깜빡이지 않게 한다.
    """
    x0, x1, y0, y1 = box.x0, box.x1, box.y0, box.y1
    z0, z1 = box.z0, box.z1
    xm = 0.5 * (x0 + x1)
    quads, owner = [], []
    for i in range(mesh.size):
        f, c, half = int(mesh.face[i]), (mesh.x[i], mesh.z[i]), 0.5 * mesh.length[i]
        if f in (0, 2):
            za, zb = c[1] - half, c[1] + half
            xf = x0 - eps if f == 0 else x1 + eps
            xa, xb = (x0, xm) if f == 0 else (xm, x1)
            quads.append([[xf, y0, za], [xf, y1, za], [xf, y1, zb], [xf, y0, zb]])
            quads.append([[xa, y0 - eps, za], [xb, y0 - eps, za], [xb, y0 - eps, zb], [xa, y0 - eps, zb]])
            quads.append([[xa, y1 + eps, za], [xb, y1 + eps, za], [xb, y1 + eps, zb], [xa, y1 + eps, zb]])
            owner += [i, i, i]
        else:
            xa, xb = c[0] - half, c[0] + half
            zf = z0 - eps if f == 1 else z1 + eps
            quads.append([[xa, y0, zf], [xb, y0, zf], [xb, y1, zf], [xa, y1, zf]])
            owner.append(i)
    return np.array(quads, dtype=float).reshape(-1, 4, 3), np.array(owner, dtype=int)


if gl is not None:
    class OrbitView(gl.GLViewWidget):
        """GLViewWidget + 오른쪽 드래그 이동 + 화면에 겹쳐 그리는 3D 위치 라벨.

        라벨은 OpenGL 텍스트 대신 일반 QLabel 을 3D 점의 화면 투영 위치에 놓는다
        (드라이버·원격 데스크톱 환경에 관계없이 항상 보이도록).
        """

        def __init__(self, parent=None):
            super().__init__(parent)
            self._labels: list[tuple[QLabel, np.ndarray]] = []

        def set_labels(self, labels) -> None:
            """labels: [(텍스트, (x, y, z), 색)]"""
            for label, _ in self._labels:
                label.hide()          # deleteLater 는 이벤트 루프로 돌아가야 처리되므로 먼저 숨긴다
                label.deleteLater()
            self._labels = []
            for text, pos, color in labels:
                label = QLabel(text, self)
                label.setStyleSheet(f"color: {color}; background: rgba(244, 244, 241, 170); "
                                    "border-radius: 3px; padding: 0px 3px;")
                label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
                label.adjustSize()
                self._labels.append((label, np.asarray(pos, dtype=float)))
            self._place_labels()

        def paintGL(self, *args, **kwargs):
            super().paintGL(*args, **kwargs)
            self._place_labels()

        def _place_labels(self) -> None:
            if not self._labels:
                return
            viewport = (0, 0, self.width(), self.height())
            transform = self.projectionMatrix(viewport, viewport) * self.viewMatrix()
            for label, p in self._labels:
                v = transform.map(QVector4D(float(p[0]), float(p[1]), float(p[2]), 1.0))
                if v.w() <= 0:
                    label.hide()
                    continue
                x = (v.x() / v.w() + 1.0) * 0.5 * self.width()
                y = (1.0 - v.y() / v.w()) * 0.5 * self.height()
                label.move(int(x - label.width() / 2), int(y - label.height() / 2))
                label.show()

        def mouseMoveEvent(self, ev):
            if ev.buttons() == Qt.MouseButton.RightButton:
                pos = ev.position()
                if not hasattr(self, "mousePos"):
                    self.mousePos = pos
                diff = pos - self.mousePos
                self.mousePos = pos
                self.pan(diff.x(), diff.y(), 0, relative="view-upright")
                return
            super().mouseMoveEvent(ev)


class Geometry3DView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._metal = "Al"
        self._geometry: Geometry | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if gl is None:
            message = QLabel(f"3D 보기를 쓰려면 PyOpenGL 이 필요합니다 (pip install PyOpenGL).\n오류: {_GL_ERROR}")
            message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(message)
            self.view = None
            return

        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 0)
        for name, elevation, azimuth in VIEW_PRESETS:
            button = QPushButton(name)
            button.clicked.connect(lambda _=False, e=elevation, a=azimuth: self.set_view(e, a))
            bar.addWidget(button)
        bar.addStretch(1)
        help_label = QLabel("왼쪽 드래그: 회전 · 오른쪽 드래그: 이동 · 휠: 확대/축소")
        help_label.setStyleSheet(f"color: {style.INK_SECONDARY};")
        bar.addWidget(help_label)
        layout.addLayout(bar)

        self.view = OrbitView()
        self.view.setBackgroundColor(QColor("#f4f4f1"))
        layout.addWidget(self.view, 1)
        self.legend = ColorLegend(parent=self.view)
        self._surface = None
        self._surface_owner = None
        self._surface_size = 0

    def set_metal(self, metal: str) -> None:
        self._metal = metal

    def show_geometry(self, g: Geometry) -> None:
        if self.view is None:
            return
        first = self._geometry is None
        self._geometry = g
        for item in list(self.view.items):
            self.view.removeItem(item)
        self._surface = None          # 형상이 바뀌면 이전 계산의 두께 색은 지운다
        self.legend.hide()
        self._build_scene(g)
        if first:
            self.set_view(*VIEW_PRESETS[0][1:])

    def set_profile(self, mesh, values, vmin: float, vmax: float, label: str, rgb=None, legend_colors=None) -> None:
        """양극 표면을 요소별 값(두께)으로 색칠한다. 같은 요소 구성이면 색만 바꾼다 (실시간 갱신).

        rgb (N, 3, 0–1) 를 주면 색상표 대신 그 색(예: Ti 간섭색)으로 칠한다.
        """
        if self.view is None or self._geometry is None or mesh.size < 2:
            return
        if self._surface is None or self._surface_size != mesh.size:
            self._build_surface(mesh)
        if rgb is None:
            element_colors = colors(values, vmin, vmax)
        else:
            element_colors = np.column_stack([np.asarray(rgb, dtype=float), np.ones(mesh.size)])
        face_colors = np.repeat(element_colors[self._surface_owner], 2, axis=0)  # 사각형 = 삼각형 2개
        self._surface_data.setFaceColors(face_colors)
        self._surface.meshDataChanged()
        self.legend.set_range(vmin, vmax, label, colors=legend_colors)
        self._place_legend()

    def clear_profile(self) -> None:
        """두께 색칠을 지운다 (새 계산을 시작할 때)."""
        if self.view is None:
            return
        if self._surface is not None:
            self.view.removeItem(self._surface)
            self._surface = None
        self.legend.hide()

    def _build_surface(self, mesh):
        box = self._geometry.anode.box(self._geometry.tank)
        quads, owner = anode_surface_quads(mesh, box, eps=0.3 * u.MM)
        vertexes = quads.reshape(-1, 3) / u.CM
        base = 4 * np.arange(quads.shape[0])[:, np.newaxis]
        faces = np.vstack([base + [0, 1, 2], base + [0, 2, 3]])
        faces = faces.reshape(2, -1, 3).transpose(1, 0, 2).reshape(-1, 3)   # 사각형마다 삼각형 2개 연속
        self._surface_data = gl.MeshData(vertexes=vertexes, faces=faces)
        self._surface = gl.GLMeshItem(meshdata=self._surface_data, smooth=False, shader=None, glOptions="opaque")
        self._surface.setDepthValue(-1)   # 반투명 전해질보다 먼저 그려야 가려지지 않는다
        self.view.addItem(self._surface)
        self._surface_owner = owner
        self._surface_size = mesh.size

    def _place_legend(self) -> None:
        self.legend.move(12, self.view.height() - self.legend.height() - 12)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "view", None) is not None:
            self._place_legend()

    def set_view(self, elevation: float, azimuth: float) -> None:
        if self.view is None or self._geometry is None:
            return
        t = self._geometry.tank
        w, d, h = t.width / u.CM, t.depth / u.CM, t.height / u.CM
        self.view.setCameraPosition(pos=Vector(w / 2, d / 2, h / 2 + 2.0), distance=self._fit_distance(w, d, h),
                                    elevation=elevation, azimuth=azimuth)

    def _fit_distance(self, w, d, h) -> float:
        """화면 가로·세로 시야각 모두에 장면(전해조 + 버스바 여유)이 들어가는 카메라 거리."""
        radius = 0.5 * math.sqrt(w * w + d * d + (h + 10.0) ** 2)
        fov_h = math.radians(self.view.opts["fov"])
        aspect = max(self.view.height(), 1) / max(self.view.width(), 1)
        fov_v = 2.0 * math.atan(math.tan(0.5 * fov_h) * aspect)
        return 1.1 * radius / math.sin(0.5 * min(fov_h, fov_v))

    # ── 장면 구성 (단위: cm) ────────────────────────────────────────────────
    def _build_scene(self, g: Geometry) -> None:
        t = g.tank
        w, d, h, level = t.width / u.CM, t.depth / u.CM, t.height / u.CM, t.liquid_level / u.CM

        grid = gl.GLGridItem(color=(0, 0, 0, 40), glOptions="translucent")
        grid.setSize(w * 1.6, d * 1.6)
        grid.setSpacing(5, 5)
        grid.translate(w / 2, d / 2, -0.05)
        self.view.addItem(grid)

        # 전극 (불투명) 먼저, 전해질(반투명)은 마지막에
        labels = []
        bus_z = h + 4.0
        anode_color = style.METAL_COLORS.get(self._metal, "#b9bec6")
        self._plate(g.anode.box(t), anode_color, style.ANODE_EDGE, bus_z)
        labels.append(("양극 (+)", (g.anode.x_center / u.CM, d / 2, bus_z + 2.0), style.INK))
        for i, c in enumerate(g.cathodes, start=1):
            self._plate(c.box(t), style.CATHODE_COLOR, style.INK, bus_z)
            labels.append((f"음극 {i} (−)", (c.x_center / u.CM, d / 2, bus_z + 2.0), style.INK))

        self._lines(box_edges(0, w, 0, d, 0, h, top=False), style.TANK_LINE, 2.0)
        labels += self._axes()

        liquid = gl.GLMeshItem(meshdata=box_mesh(0, w, 0, d, 0, min(level, h)), color=_rgba("#5b9be6", 0.18),
                               smooth=False, shader=None, glOptions="translucent")
        self.view.addItem(liquid)
        surface = min(level, h)
        corners = np.array([[0, 0, surface], [w, 0, surface], [w, d, surface], [0, d, surface]], dtype=float)
        self._lines(corners[[0, 1, 1, 2, 2, 3, 3, 0]], style.LIQUID_LINE, 1.5)   # 액면 테두리
        self.view.set_labels(labels)

    def _plate(self, b, fill, edge, bus_z):
        x0, x1, y0, y1 = b.x0 / u.CM, b.x1 / u.CM, b.y0 / u.CM, b.y1 / u.CM
        z0, z1 = b.z0 / u.CM, b.z1 / u.CM
        mesh = gl.GLMeshItem(meshdata=box_mesh(x0, x1, y0, y1, z0, z1), color=_rgba(fill), smooth=False,
                             shader="shaded", drawEdges=True, edgeColor=_rgba(edge), glOptions="opaque")
        self.view.addItem(mesh)
        xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if z1 < bus_z:   # 행거
            self._lines(np.array([[xc, yc, z1], [xc, yc, bus_z]]), edge, 1.5)

    def _lines(self, points, color, width):
        item = gl.GLLinePlotItem(pos=points, color=_rgba(color), width=width, mode="lines",
                                 antialias=True, glOptions="translucent")
        self.view.addItem(item)

    def _axes(self):
        """원점 근처의 좌표축 표시 (x 빨강, y 초록, z 파랑). 라벨 목록을 돌려준다."""
        o = np.array([-4.0, -4.0, 0.0])
        length = 6.0
        labels = []
        for axis, color, name in ((0, "#d03b3b", "x"), (1, "#0ca30c", "y"), (2, "#2a78d6", "z")):
            end = o.copy()
            end[axis] += length
            self._lines(np.array([o, end]), color, 2.5)
            labels.append((name, tuple(end + 1.2 * (end - o) / length), color))
        return labels
