"""측면도 (x–z 단면) — 기본 2D 보기.

- 전극(양극·음극)을 마우스로 끌어 옮길 수 있다. 놓으면 1 mm 단위로 맞춰 electrode_moved 를 보낸다.
  끄는 동안에는 배치 검증(Geometry.validate)을 그대로 돌려, 놓을 수 없는 자리면 테두리가 빨갛게 바뀐다.
- 계산 결과가 있으면 양극 둘레에 두께 색 띠(set_profile), L2 결과면 전해질 전위장(set_potential)을 겹쳐 그린다.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QPainter, QPen
from PyQt6.QtWidgets import QCheckBox, QGraphicsRectItem, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...core import units as u
from ...core.geometry import Geometry, PlateElectrode
from .. import style
from .colormap import POTENTIAL_CMAP, THICKNESS_CMAP, ColorLegend

_STRIP_GAP = 0.05     # cm — 양극 표면과 색 띠 사이 간격
_STRIP_WIDTH = 0.35   # cm — 색 띠 두께
_SNAP = 1 * u.MM      # 끌어 놓은 위치를 맞추는 단위 (입력 칸 소수점 한 자리 cm 와 같음)
POTENTIAL_LABEL = "전해질 전위 [V] (음극 = 0)"


def _pen(color, width=1.5, style_=Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(QColor(color))
    pen.setWidthF(width)
    pen.setCosmetic(True)
    pen.setStyle(style_)
    return pen


class PlateHandle(pg.ROI):
    """끌어서 옮길 수 있는 전극 판. 채움 색을 칠하고, 마우스를 올리면 테두리를 강조한다."""

    def __init__(self, kind: str, index: int, plate: PlateElectrode, fill, edge, bounds: QRectF):
        super().__init__((plate.x_min / u.CM, plate.z_bottom / u.CM), (plate.thickness / u.CM, plate.height / u.CM),
                         movable=True, rotatable=False, resizable=False, removable=False,
                         pen=_pen(edge, 1.2), hoverPen=_pen(style.SERIES[0], 2.5), maxBounds=bounds)
        self.kind = kind            # "anode" | "cathode"
        self.index = index          # 음극이면 Geometry.cathodes 안의 순번
        self.plate = plate
        self.edge = edge
        self._brush = QBrush(QColor(fill))
        self.setZValue(5)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setToolTip("끌어서 옮기기 (1 mm 단위)")

    def moved_plate(self) -> PlateElectrode:
        """지금 위치의 판 (1 mm 단위로 맞춤)."""
        pos = self.pos()
        x_center = (pos.x() + 0.5 * self.plate.thickness / u.CM) * u.CM
        z_bottom = pos.y() * u.CM
        return replace(self.plate, x_center=round(x_center / _SNAP) * _SNAP, z_bottom=round(z_bottom / _SNAP) * _SNAP)

    def set_valid(self, valid: bool) -> None:
        self.setPen(_pen(self.edge if valid else style.STATUS_CRITICAL, 1.2 if valid else 2.5))

    def paint(self, p, opt, widget):
        r = QRectF(0, 0, self.state["size"][0], self.state["size"][1]).normalized()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(self.currentPen)
        p.setBrush(self._brush)
        p.drawRect(r)


class SideView(pg.PlotWidget):
    """전해조를 옆에서 본 단면. 좌표는 cm."""

    electrode_moved = pyqtSignal(str, int, float, float)   # (종류, 음극 순번, x 중심 [m], 바닥 z [m])

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
        self._gap_items = []
        self._handles: list[PlateHandle] = []
        self._followers: dict[PlateHandle, list] = {}   # 판과 함께 움직이는 행거·라벨
        self._editable = True
        self.geometry: Geometry | None = None
        self._metal = "Al"
        self._strips: list[tuple[pg.ImageItem, np.ndarray]] = []   # 면마다 색 띠 (이미지, 요소 번호)
        self._strip_mesh = None
        self._lut = THICKNESS_CMAP.getLookupTable(nPts=256)
        self.legend = ColorLegend(parent=self)
        self._potential: pg.ImageItem | None = None
        self._potential_lut = POTENTIAL_CMAP.getLookupTable(nPts=256)
        self.potential_legend = ColorLegend(POTENTIAL_LABEL, parent=self, cmap=POTENTIAL_CMAP)
        self._drag_label = pg.TextItem("", color=style.INK, anchor=(0, 1), fill=pg.mkBrush(252, 252, 251, 230))
        self._drag_label.setZValue(20)

    def set_metal(self, metal: str) -> None:
        self._metal = metal

    def set_editable(self, editable: bool) -> None:
        """계산 중에는 전극을 끌 수 없게 한다."""
        self._editable = editable
        for handle in self._handles:
            handle.translatable = editable
            handle.setCursor(Qt.CursorShape.SizeAllCursor if editable else Qt.CursorShape.ArrowCursor)

    def show_geometry(self, g: Geometry) -> None:
        for item in self._items + self._gap_items + self._handles:
            self.removeItem(item)
        if self._drag_label.scene() is not None:
            self.removeItem(self._drag_label)
        self._items.clear()
        self._gap_items.clear()
        self._handles.clear()
        self._followers.clear()
        self.clear_profile()    # 형상이 바뀌면 이전 계산의 두께 색·전위장은 의미가 없다
        self.clear_potential()
        self.geometry = g
        t = g.tank
        w, h, level = t.width / u.CM, t.height / u.CM, t.liquid_level / u.CM

        self._rect(0, 0, w, min(level, h), brush=QColor(*style.LIQUID_FILL), pen=Qt.PenStyle.NoPen)
        self._line([0, w], [level, level], style.LIQUID_LINE, 1.2, Qt.PenStyle.DashLine)
        self._line([0, 0, w, w], [h, 0, 0, h], style.TANK_LINE, 2.5)       # 위가 열린 용기
        self._text("전해질", 1.0, 1.0, style.LIQUID_LINE, anchor=(0, 1))

        bus_z = h + 3.0
        bounds = QRectF(0.0, 0.0, w, 1e4)   # 전해조 가로 안, 바닥 위 (위쪽은 열어 둔다)
        a = g.anode
        anode = self._electrode("anode", 0, a, style.METAL_COLORS.get(self._metal, "#b9bec6"), style.ANODE_EDGE,
                                bus_z, bounds)
        self._followers[anode].append(self._text("양극 (+)", a.x_center / u.CM, bus_z + 0.5, style.INK,
                                                 anchor=(0.5, 1)))
        for i, c in enumerate(g.cathodes):
            handle = self._electrode("cathode", i, c, style.CATHODE_COLOR, style.INK, bus_z, bounds)
            # 양극 라벨과 겹치지 않도록 음극 라벨은 전극 아래쪽에
            self._followers[handle].append(self._text(f"음극 {i + 1} (−)", c.x_center / u.CM, c.z_bottom / u.CM - 0.3,
                                                      style.INK, anchor=(0.5, 0)))
            self._gap(a, c, level)
        self.set_editable(self._editable)

        pad = 4.0
        self._bounds = (-pad, w + pad, -pad, bus_z + 6.0)
        self._fit()

    # ── 두께 색 띠 ──────────────────────────────────────────────────────────
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
        self._place_legends()

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

    # ── 전해질 전위장 (L2) ──────────────────────────────────────────────────
    def set_potential(self, image, extent, vmax: float) -> None:
        """전해질 전위 이미지 [V] (행 = z 아래→위). extent = (x0, x1, z0, z1) [m]. 전극 안(nan)은 투명."""
        if self._potential is None:
            self._potential = pg.ImageItem(axisOrder="row-major")
            self._potential.setZValue(1)       # 전해질 채움 위, 전극 아래
            self._potential.setOpacity(0.92)
            self.addItem(self._potential)
        x0, x1, z0, z1 = (v / u.CM for v in extent)
        self._potential.setImage(np.asarray(image, dtype=float), levels=(0.0, vmax), lut=self._potential_lut,
                                 autoLevels=False)
        self._potential.setRect(QRectF(x0, z0, x1 - x0, z1 - z0))
        self.potential_legend.set_range(0.0, vmax)
        self._place_legends()

    def clear_potential(self) -> None:
        if self._potential is not None:
            self.removeItem(self._potential)
            self._potential = None
        self.potential_legend.hide()

    # ── 끌어서 옮기기 ───────────────────────────────────────────────────────
    def _candidate(self, handle: PlateHandle) -> Geometry:
        g = self.geometry
        plate = handle.moved_plate()
        if handle.kind == "anode":
            return replace(g, anode=plate)
        cathodes = list(g.cathodes)
        cathodes[handle.index] = plate
        return replace(g, cathodes=tuple(cathodes))

    def _on_drag(self, handle: PlateHandle) -> None:
        if self.geometry is None:
            return
        # 옮기는 동안에는 이전 배치 기준의 표시(극간 거리, 두께 색, 전위장)를 감춘다
        for item in self._gap_items:
            item.setVisible(False)
        self.clear_profile()
        self.clear_potential()
        plate = handle.moved_plate()
        dx = (plate.x_center - handle.plate.x_center) / u.CM
        dz = (plate.z_bottom - handle.plate.z_bottom) / u.CM
        for item in self._followers.get(handle, []):
            item.setPos(item.home.x() + dx, item.home.y() + dz)
        errors = self._candidate(handle).validate()
        handle.set_valid(not errors)
        text = f"x {plate.x_center / u.CM:.1f} cm · 바닥 z {plate.z_bottom / u.CM:.1f} cm"
        if errors:
            text += f"\n{errors[0]}"
        self._drag_label.setText(text)
        self._drag_label.setColor(QColor(style.STATUS_CRITICAL if errors else style.INK))
        self._drag_label.setPos(plate.x_max / u.CM + 0.8, plate.z_top / u.CM)
        if self._drag_label.scene() is None:
            self.addItem(self._drag_label)

    def _on_drop(self, handle: PlateHandle) -> None:
        plate = handle.moved_plate()
        # 끌기 처리(ROI 의 마우스 이벤트) 안에서 장면을 다시 그리지 않도록 이벤트 루프로 미룬다
        QTimer.singleShot(0, lambda: self.electrode_moved.emit(handle.kind, handle.index, plate.x_center,
                                                               plate.z_bottom))

    def _place_legends(self) -> None:
        self.legend.move(self.width() - self.legend.width() - 12, 34)
        self.potential_legend.move(70, 34)

    def _fit(self) -> None:
        """비율 고정 상태에서 전해조 전체가 보이도록 범위를 맞춘다 (숨겨진 상태에서 그려졌을 때 대비)."""
        if self._bounds is not None:
            x0, x1, y0, y1 = self._bounds
            self.getPlotItem().vb.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()
        if self._bounds is not None:
            self._place_legends()

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

    def _line(self, xs, ys, color, width=1.5, style_=Qt.PenStyle.SolidLine, group=None):
        item = pg.PlotDataItem(xs, ys, pen=_pen(color, width, style_))
        item.setZValue(6)        # 전위장 이미지(1) 위
        self.addItem(item)
        (self._items if group is None else group).append(item)
        return item

    def _text(self, text, x, y, color, anchor=(0.5, 0.5), group=None):
        item = pg.TextItem(text, color=color, anchor=anchor)
        item.setPos(x, y)
        item.setZValue(6)
        item.home = item.pos()   # 끄는 동안 판과 함께 옮길 때의 기준 위치
        self.addItem(item)
        (self._items if group is None else group).append(item)
        return item

    def _electrode(self, kind, index, plate, fill, edge, bus_z, bounds) -> PlateHandle:
        handle = PlateHandle(kind, index, plate, fill, edge, bounds)
        self.addItem(handle)
        self._handles.append(handle)
        xc = plate.x_center / u.CM
        hanger = self._line([0.0, 0.0], [plate.z_top / u.CM, bus_z], edge, 1.0)   # 행거·버스바 연결
        hanger.setPos(xc, 0.0)
        hanger.home = hanger.pos()
        self._followers[handle] = [hanger]
        handle.sigRegionChanged.connect(self._on_drag)
        handle.sigRegionChangeFinished.connect(self._on_drop)
        return handle

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
        self._line([x0 / u.CM, x1 / u.CM], [z, z], style.INK_MUTED, 1.0, Qt.PenStyle.DotLine, group=self._gap_items)
        self._text(f"{(x1 - x0) / u.CM:.1f} cm", 0.5 * (x0 + x1) / u.CM, z, style.INK_SECONDARY, anchor=(0.5, 1.1),
                   group=self._gap_items)


class SideViewPanel(QWidget):
    """측면도 + 위쪽 도구 줄 (전위장 표시 선택, 끌어 옮기기 안내)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.view = SideView()
        self.show_potential = QCheckBox("전해질 전위 표시")
        self.show_potential.setChecked(True)
        self.show_potential.setEnabled(False)
        self.show_potential.setToolTip("해석 단계 L2 로 계산하면 전해질 속 전위 분포를 겹쳐 그립니다.")
        hint = QLabel("전극을 마우스로 끌어 옮길 수 있습니다 (1 mm 단위)")
        hint.setStyleSheet(f"color: {style.INK_SECONDARY};")
        bar = QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 0)
        bar.addWidget(self.show_potential)
        bar.addStretch(1)
        bar.addWidget(hint)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(self.view, 1)
