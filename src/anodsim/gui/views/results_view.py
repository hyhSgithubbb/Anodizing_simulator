"""L0 결과 그래프: 전압, 전류밀도, 산화막 두께, 배리어층 두께 (시간축 연동 + 십자선 판독)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...core import units as u
from ...core.limits import FLAG_MESSAGES, WARNING_FLAGS
from .. import style

CONTROL_LABELS = {"CV": "정전압", "CC": "정전류", "OFF": "출력 없음"}


def _series_pen(index: int) -> pg.QtGui.QPen:
    pen = pg.mkPen(style.SERIES[index], width=2)
    pen.setCosmetic(True)
    return pen


class _Plot(pg.PlotWidget):
    def __init__(self, title: str, unit_label: str, show_time_axis: bool):
        super().__init__(background=style.SURFACE)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setTitle(title, color=style.INK, size="10pt")
        self.showGrid(x=True, y=True, alpha=0.15)
        for name in ("bottom", "left"):
            axis = self.getAxis(name)
            axis.setPen(pg.mkPen(style.AXIS))
            axis.setTextPen(pg.mkPen(style.INK_SECONDARY))
            axis.enableAutoSIPrefix(False)   # 단위는 축 제목에 고정 (×0.001 같은 자동 배율 표시 방지)
        self.getAxis("left").setLabel(unit_label, color=style.INK_SECONDARY)
        self.getAxis("left").setWidth(56)
        if show_time_axis:
            self.getAxis("bottom").setLabel("시간 [min]", color=style.INK_SECONDARY)
        else:   # 시간축은 맨 아래 그래프에만 (모든 그래프가 같은 시간축을 공유)
            self.getAxis("bottom").setStyle(showValues=False)
        self.cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(style.INK_MUTED, width=1))
        self.cursor.setVisible(False)
        self.addItem(self.cursor, ignoreBounds=True)
        self.legend = None
        self.curves: list[pg.PlotDataItem] = []

    def set_series(self, t, series: list[tuple[str, np.ndarray]]) -> None:
        for curve in self.curves:
            self.removeItem(curve)
        self.curves.clear()
        if self.legend is not None:
            self.legend.scene().removeItem(self.legend)
            self.getPlotItem().legend = None
            self.legend = None
        if len(series) > 1:   # 시리즈가 2개 이상일 때만 범례
            self.legend = self.addLegend(offset=(-8, -8), labelTextColor=style.INK_SECONDARY,
                                         brush=pg.mkBrush(252, 252, 251, 220), pen=pg.mkPen(style.GRID))
        for i, (name, values) in enumerate(series):
            self.curves.append(self.plot(t, values, pen=_series_pen(i), name=name if len(series) > 1 else None))
        self.enableAutoRange()


class ResultsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._t_min = None

        self.readout = QLabel("‘계산 실행’을 누르면 시간에 따른 전압·전류·두께가 여기에 표시됩니다.")
        self.readout.setTextFormat(Qt.TextFormat.RichText)
        self.readout.setWordWrap(True)
        self.readout.setStyleSheet(f"color: {style.INK_SECONDARY}; padding: 2px 4px;")
        self.notices = QLabel()
        self.notices.setWordWrap(True)
        self.notices.setVisible(False)

        self.voltage = _Plot("전압", "전압 [V]", show_time_axis=False)
        self.current = _Plot("전류밀도 j (양극 평균)", "j [A/dm²]", show_time_axis=False)
        self.thickness = _Plot("산화막 두께", "두께 [μm]", show_time_axis=False)
        self.barrier = _Plot("배리어층 두께 δ_b", "δ_b [nm]", show_time_axis=True)
        self.plots = (self.voltage, self.current, self.thickness, self.barrier)
        for p in self.plots[1:]:
            p.setXLink(self.voltage)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.addWidget(self.readout)
        for p in self.plots:
            layout.addWidget(p, 1)
        layout.addWidget(self.notices)

        for p in self.plots:
            p.scene().sigMouseMoved.connect(lambda pos, plot=p: self._on_mouse(plot, pos))

    def clear(self) -> None:
        self._result = None
        for p in self.plots:
            p.set_series([], [])
            p.cursor.setVisible(False)

    def show_result(self, result, regime: str, stopped: bool = False) -> None:
        self._result = result
        t = result.time / u.MINUTE
        self._t_min = t
        self.voltage.set_series(t, [("인가 전압 U", result.voltage), ("막 전압 U_f", result.film_voltage)])
        self.current.set_series(t, [("j", result.current_density / u.A_PER_DM2)])
        self.thickness.setTitle("산화막 총 두께 h (배리어층 + 기공층)" if regime == "porous" else "산화막 두께 h",
                                color=style.INK, size="10pt")
        self.thickness.set_series(t, [("h", result.total / u.UM)])
        self.barrier.set_series(t, [("δ_b", result.barrier / u.NM)])
        self._show_readout(result.time.size - 1, final=True)
        self._show_notices(result, stopped)

    # ── 판독 ────────────────────────────────────────────────────────────────
    def _on_mouse(self, plot, pos):
        if self._result is None or not plot.sceneBoundingRect().contains(pos):
            return
        x = plot.getPlotItem().vb.mapSceneToView(pos).x()
        k = int(np.clip(np.searchsorted(self._t_min, x), 0, self._t_min.size - 1))
        if k > 0 and abs(self._t_min[k - 1] - x) < abs(self._t_min[k] - x):
            k -= 1
        for p in self.plots:
            p.cursor.setPos(self._t_min[k])
            p.cursor.setVisible(True)
        self._show_readout(k)

    def _show_readout(self, k: int, final: bool = False) -> None:
        r = self._result
        prefix = "<b>최종</b>" if final else f"<b>t = {r.time[k] / u.MINUTE:.2f} min</b>"
        parts = [
            prefix,
            f"U = {r.voltage[k]:.2f} V",
            f"U_f = {r.film_voltage[k]:.2f} V",
            f"I = {r.current[k]:.3g} A",
            f"j = {r.current_density[k] / u.A_PER_DM2:.3g} A/dm²",
            f"h = {r.total[k] / u.UM:.3g} μm",
            f"δ_b = {r.barrier[k] / u.NM:.1f} nm",
            CONTROL_LABELS.get(r.control[k], r.control[k]),
        ]
        # 값과 단위가 줄바꿈으로 갈라지지 않도록 항목 안의 공백은 줄바꿈 금지 공백으로
        self.readout.setText(" │ ".join(p.replace(" ", "&nbsp;") for p in parts))

    def _show_notices(self, result, stopped: bool) -> None:
        lines = []
        if stopped:
            lines.append("■ 사용자가 계산을 중지했습니다 — 중지 시점까지의 결과입니다.")
        for flag in WARNING_FLAGS:
            if flag in result.flags:
                lines.append(f"▲ 경고: {FLAG_MESSAGES[flag]}")
        for w in result.warnings:
            lines.append(f"△ 주의: {w}")
        self.notices.setVisible(bool(lines))
        self.notices.setText("\n".join(lines))
        critical = any(f in result.flags for f in WARNING_FLAGS)
        border = style.STATUS_CRITICAL if critical else style.STATUS_WARNING
        self.notices.setStyleSheet(f"background: #fffaf0; border: 1px solid {border}; border-radius: 6px; "
                                   f"padding: 6px; color: {style.INK};")
