"""측면도 (x–z 단면) — 기본 2D 보기."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtWidgets import QGraphicsRectItem

from ...core import units as u
from ...core.geometry import Geometry
from .. import style
from .colormap import THICKNESS_CMAP, ColorLegend

_STRIP_GAP = 0.05     # cm — 양극 표면과 색 띠 사이 간격
_STRIP_WIDTH = 0.35   # cm — 색 띠 두께


def _pen(color, width=1.5, style_=Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(QColor(color))
    pen.setWidthF(width)
    pen.setCosmetic(True)
    pen.setStyle(style_)
    return pen


class SideView(pg.PlotWidget):
    """전해조를 옆에서 본 단면. 좌표는 cm."""

    _bounds = None   # 부모 생성자 안에서 resizeEvent 가 먼저 불릴 수 있다

    def __init__(self, parent=None):
        super().__init__(parent, background=style.SURFACE)
        self.setAspectLocked(True)
        self.setMenuEnabled(False)
        self.hideButtons()
        for name, label in (("bottom", "x [cm]"), ("left", "z [cm]")):
            axis = self.getAxis(name)
            axis.setLabel(label)
            axis.setPen(pg.mkPen(style.AXIS))
            axis.setTextPen(pg.mkPen(style.INK_SECONDARY))
        self.showGrid(x=True, y=True, alpha=0.12)
        self.setTitle("측면도 (x–z 단면, 화면 안쪽이 깊이 y)", color=style.INK_SECONDARY, size="10pt")
        self._items = []
        self._metal = "Al"
        self._strips: list[tuple[pg.ImageItem, np.ndarray]] = []   # 면마다 색 띠 (이미지, 요소 번호)
        self._strip_mesh = None
        self._lut = THICKNESS_CMAP.getLookupTable(nPts=256)
        self.legend = ColorLegend(parent=self)

    def set_metal(self, metal: str) -> None:
        self._metal = metal

    def show_geometry(self, g: Geometry) -> None:
        for item in self._items:
            self.removeItem(item)
        self._items.clear()
        self.clear_profile()   # 형상이 바뀌면 이전 계산의 두께 색은 의미가 없다
        t = g.tank
        w, h, level = t.width / u.CM, t.height / u.CM, t.liquid_level / u.CM

        self._rect(0, 0, w, min(level, h), brush=QColor(*style.LIQUID_FILL), pen=Qt.PenStyle.NoPen)
        self._line([0, w], [level, level], style.LIQUID_LINE, 1.2, Qt.PenStyle.DashLine)
        self._line([0, 0, w, w], [h, 0, 0, h], style.TANK_LINE, 2.5)       # 위가 열린 용기
        self._text("전해질", 1.0, 1.0, style.LIQUID_LINE, anchor=(0, 1))

        bus_z = h + 3.0
        a = g.anode
        self._electrode(a, style.METAL_COLORS.get(self._metal, "#b9bec6"), style.ANODE_EDGE, bus_z)
        self._text("양극 (+)", a.x_center / u.CM, bus_z + 0.5, style.INK, anchor=(0.5, 1))
        for i, c in enumerate(g.cathodes, start=1):
            self._electrode(c, style.CATHODE_COLOR, style.INK, bus_z)
            # 양극 라벨과 겹치지 않도록 음극 라벨은 전극 아래쪽에
            self._text(f"음극 {i} (−)", c.x_center / u.CM, c.z_bottom / u.CM - 0.3, style.INK, anchor=(0.5, 0))
            self._gap(a, c, level)

        pad = 4.0
        self._bounds = (-pad, w + pad, -pad, bus_z + 6.0)
        self._fit()

    def set_profile(self, mesh, values, vmin: float, vmax: float, label: str) -> None:
        """양극 둘레 바깥에 요소별 값(두께)을 색 띠로 표시한다 (요소 2개 이상일 때).

        면마다 이미지 한 장(1 × 요소 수)이라 실시간 갱신에도 가볍다.
        """
        if mesh.size < 2:
            self.clear_profile()
            return
        if self._strip_mesh is not mesh:
            self._build_strips(mesh)
        values = np.asarray(values, dtype=float)
        for item, index in self._strips:
            item.setImage(values[index], levels=(vmin, vmax), lut=self._lut, autoLevels=False)
        self.legend.set_range(vmin, vmax, label)
        self._place_legend()

    def clear_profile(self) -> None:
        for item, _ in self._strips:
            self.removeItem(item)
        self._strips.clear()
        self._strip_mesh = None
        self.legend.hide()

    def _build_strips(self, mesh) -> None:
        self.clear_profile()
        self._strip_mesh = mesh
        x0, x1 = mesh.x.min() / u.CM, mesh.x.max() / u.CM
        for face in mesh.faces:
            index = np.flatnonzero(mesh.face == face)
            span = mesh.length[index].sum() / u.CM
            if face in (0, 2):          # 세로 띠: 행 0 = 가장 낮은 z
                z = mesh.z[index] / u.CM
                index = index[np.argsort(z)]
                x = x0 - _STRIP_GAP - _STRIP_WIDTH if face == 0 else x1 + _STRIP_GAP
                rect = QRectF(x, z.min() - 0.5 * span / index.size, _STRIP_WIDTH, span)
                shape = (index.size, 1)
            else:                       # 가로 띠: 열 0 = 가장 작은 x
                x = mesh.x[index] / u.CM
                index = index[np.argsort(x)]
                zf = mesh.z[index[0]] / u.CM
                z = zf - _STRIP_GAP - _STRIP_WIDTH if face == 1 else zf + _STRIP_GAP
                rect = QRectF(x.min() - 0.5 * span / index.size, z, span, _STRIP_WIDTH)
                shape = (1, index.size)
            item = pg.ImageItem(np.zeros(shape), axisOrder="row-major")
            item.setZValue(10)
            item.setRect(rect)
            self.addItem(item)
            self._strips.append((item, index.reshape(shape)))

    def _place_legend(self) -> None:
        self.legend.move(self.width() - self.legend.width() - 12, 34)

    def _fit(self) -> None:
        """비율 고정 상태에서 전해조 전체가 보이도록 범위를 맞춘다 (숨겨진 상태에서 그려졌을 때 대비)."""
        if self._bounds is not None:
            x0, x1, y0, y1 = self._bounds
            self.getPlotItem().vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()
        if self._bounds is not None:
            self._place_legend()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit()

    # ── 그리기 도우미 ───────────────────────────────────────────────────────
    def _rect(self, x, y, w, h, brush=None, pen=None, pen_color=None):
        item = QGraphicsRectItem(QRectF(x, y, w, h))
        item.setBrush(QBrush(brush) if brush is not None else QBrush(Qt.BrushStyle.NoBrush))
        if pen == Qt.PenStyle.NoPen:
            item.setPen(QPen(Qt.PenStyle.NoPen))
        else:
            item.setPen(_pen(pen_color or style.INK, 1.2))
        self.addItem(item)
        self._items.append(item)
        return item

    def _line(self, xs, ys, color, width=1.5, style_=Qt.PenStyle.SolidLine):
        item = pg.PlotDataItem(xs, ys, pen=_pen(color, width, style_))
        self.addItem(item)
        self._items.append(item)

    def _text(self, text, x, y, color, anchor=(0.5, 0.5)):
        item = pg.TextItem(text, color=color, anchor=anchor)
        item.setPos(x, y)
        self.addItem(item)
        self._items.append(item)

    def _electrode(self, plate, fill, edge, bus_z):
        x0, z0 = plate.x_min / u.CM, plate.z_bottom / u.CM
        self._rect(x0, z0, plate.thickness / u.CM, plate.height / u.CM, brush=QColor(fill), pen_color=edge)
        xc = plate.x_center / u.CM
        self._line([xc, xc], [plate.z_top / u.CM, bus_z], edge, 1.0)     # 행거·버스바 연결

    def _gap(self, anode, cathode, level):
        """양극–음극 마주보는 면 사이 거리 표시."""
        if cathode.x_center < anode.x_center:
            x0, x1 = cathode.x_max, anode.x_min
        else:
            x0, x1 = anode.x_max, cathode.x_min
        if x1 <= x0:
            return
        z_low = max(anode.z_bottom, cathode.z_bottom)
        z_high = min(anode.z_top, cathode.z_top, level)
        if z_high <= z_low:
            return
        z = 0.5 * (z_low + z_high) / u.CM
        self._line([x0 / u.CM, x1 / u.CM], [z, z], style.INK_MUTED, 1.0, Qt.PenStyle.DotLine)
        self._text(f"{(x1 - x0) / u.CM:.1f} cm", 0.5 * (x0 + x1) / u.CM, z, style.INK_SECONDARY, anchor=(0.5, 1.1))
