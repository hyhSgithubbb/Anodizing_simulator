"""결과 그래프.

- '시간 이력' 탭: 전압, 전류밀도, 산화막 두께(평균 + 위치별 최소–최대 범위), [욕 온도 — 열수지를 계산했을 때],
  배리어층 두께 (시간축 연동 + 십자선 판독). Ti 치밀막은 판독 줄에 예상 간섭색을 함께 보인다.
- '위치별 분포' 탭 (L1): 둘레 위치 s 에 따른 두께 h(s)·전류밀도 j(s), 히트맵 h(s, t)

계산 중에는 워커가 보내는 스냅샷으로 실시간 갱신하고(add_live), 끝나면 전체 결과로 다시 그린다(show_result).
"""

from __future__ import annotations

import re

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QTabWidget, QVBoxLayout, QWidget

from ...core import optics
from ...core import units as u
from ...core.field import FACE_NAMES
from ...core.limits import FLAG_MESSAGES, WARNING_FLAGS
from .. import style
from .colormap import DEVIATION_CMAP, THICKNESS_CMAP

CONTROL_LABELS = {"CV": "정전압", "CC": "정전류", "OFF": "출력 없음"}
_TAG_OR_SPACE = re.compile(r"(<[^>]*>)| ")
HEATMAP_COLUMNS = 400   # 히트맵 시간 방향 해상도
HEATMAP_ROWS = 240      # 히트맵 위치 방향 해상도


def _series_pen(index: int, width: float = 2.0) -> pg.QtGui.QPen:
    pen = pg.mkPen(style.SERIES[index], width=width)
    pen.setCosmetic(True)
    return pen


def _style_axes(plot: pg.PlotWidget, left: str, bottom: str | None) -> None:
    for name in ("bottom", "left"):
        axis = plot.getAxis(name)
        axis.setPen(pg.mkPen(style.AXIS))
        axis.setTextPen(pg.mkPen(style.INK_SECONDARY))
        axis.enableAutoSIPrefix(False)   # 단위는 축 제목에 고정 (×0.001 같은 자동 배율 표시 방지)
    plot.getAxis("left").setLabel(left, color=style.INK_SECONDARY)
    plot.getAxis("left").setWidth(56)
    if bottom is None:
        plot.getAxis("bottom").setStyle(showValues=False)
    else:
        plot.getAxis("bottom").setLabel(bottom, color=style.INK_SECONDARY)


def _cursor() -> pg.InfiniteLine:
    line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(style.INK_MUTED, width=1))
    line.setVisible(False)
    return line


class _Plot(pg.PlotWidget):
    """시계열 그래프 하나 (축 하나에 단위 하나)."""

    def __init__(self, title: str, unit_label: str, show_time_axis: bool):
        super().__init__(background=style.SURFACE)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setTitle(title, color=style.INK, size="10pt")
        self.showGrid(x=True, y=True, alpha=0.15)
        # 시간축은 맨 아래 그래프에만 (모든 그래프가 같은 시간축을 공유)
        _style_axes(self, unit_label, "시간 [min]" if show_time_axis else None)
        self.cursor = _cursor()
        self.addItem(self.cursor, ignoreBounds=True)
        self.legend = None
        self.curves: list[pg.PlotDataItem] = []
        self.band: pg.FillBetweenItem | None = None
        self._band_edges: tuple[pg.PlotDataItem, pg.PlotDataItem] | None = None

    def set_series(self, t, series: list[tuple[str, np.ndarray]], band=None) -> None:
        """series: [(이름, 값)], band: (이름, 하한, 상한) — 범위를 옅은 띠로 (첫 시리즈 색)."""
        for curve in self.curves:
            self.removeItem(curve)
        self.curves.clear()
        self._remove_band()
        if self.legend is not None:
            self.legend.scene().removeItem(self.legend)
            self.getPlotItem().legend = None
            self.legend = None
        entries = len(series) + (band is not None)
        if entries > 1:   # 항목이 2개 이상일 때만 범례
            self.legend = self.addLegend(offset=(-8, -8), labelTextColor=style.INK_SECONDARY,
                                         brush=pg.mkBrush(252, 252, 251, 220), pen=pg.mkPen(style.GRID))
        if band is not None:
            name, lo, hi = band
            edges = (pg.PlotDataItem(t, lo, pen=None), pg.PlotDataItem(t, hi, pen=None))
            color = pg.mkColor(style.SERIES[0])
            color.setAlpha(60)
            self.band = pg.FillBetweenItem(*edges, brush=pg.mkBrush(color))
            self.addItem(self.band)
            self._band_edges = edges
            if self.legend is not None:   # 범례 견본은 채움 영역을 가진 빈 곡선으로
                sample = pg.PlotDataItem([], [], pen=pg.mkPen(color), fillLevel=0, fillBrush=pg.mkBrush(color))
                self.legend.addItem(sample, name)
        for i, (name, values) in enumerate(series):
            self.curves.append(self.plot(t, values, pen=_series_pen(i), name=name if entries > 1 else None))
        self.enableAutoRange()

    def update_series(self, t, values: list[np.ndarray], band=None) -> None:
        """곡선 구성을 유지한 채 데이터만 바꾼다 (실시간 갱신)."""
        for curve, v in zip(self.curves, values, strict=True):
            curve.setData(t, v)
        if band is not None and self._band_edges is not None:
            self._band_edges[0].setData(t, band[0])
            self._band_edges[1].setData(t, band[1])

    def _remove_band(self) -> None:
        if self.band is not None:
            self.removeItem(self.band)
        self.band = None
        self._band_edges = None


class _ProfilePlot(pg.PlotWidget):
    """둘레 위치 s 에 따른 값. 위쪽 축에 면 이름, 면 경계는 점선."""

    def __init__(self, title: str, unit_label: str, show_s_axis: bool):
        super().__init__(background=style.SURFACE)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.setTitle(title, color=style.INK, size="10pt")
        self.showGrid(x=False, y=True, alpha=0.15)
        _style_axes(self, unit_label, "둘레 위치 s [cm]" if show_s_axis else None)
        self.curve = self.plot([], [], pen=_series_pen(0))
        self._marks: list = []

    def set_mesh(self, mesh) -> None:
        for item in self._marks:
            self.removeItem(item)
        self._marks.clear()
        edges = mesh.edges / u.CM
        ticks = []
        for k, face in enumerate(mesh.faces):
            ticks.append((0.5 * (edges[k] + edges[k + 1]), FACE_NAMES[face]))
        for x in edges[1:-1]:
            line = pg.InfiniteLine(pos=x, angle=90, movable=False,
                                   pen=pg.mkPen(style.AXIS, width=1, style=Qt.PenStyle.DashLine))
            self.addItem(line, ignoreBounds=True)
            self._marks.append(line)
        self.showAxis("top")
        top = self.getAxis("top")
        top.setTicks([ticks, []])
        top.setPen(pg.mkPen(style.AXIS))
        top.setTextPen(pg.mkPen(style.INK_SECONDARY))
        self.setXRange(edges[0], edges[-1], padding=0.01)

    def set_values(self, s_cm, values) -> None:
        self.curve.setData(s_cm, values)
        self.enableAutoRange(axis="y")


class _Heatmap(pg.PlotWidget):
    """h(s, t) 히트맵: 가로 시간, 세로 둘레 위치."""

    def __init__(self):
        super().__init__(background=style.SURFACE)
        self.setMenuEnabled(False)
        self.hideButtons()
        self.showGrid(x=False, y=False)
        _style_axes(self, "둘레 위치 s [cm]", "시간 [min]")
        self.image = pg.ImageItem(axisOrder="row-major")
        self.addItem(self.image)
        self.colorbar = pg.ColorBarItem(colorMap=THICKNESS_CMAP, interactive=False, width=12)
        self.colorbar.setImageItem(self.image, insert_in=self.getPlotItem())
        self.colorbar.getAxis("right").setTextPen(pg.mkPen(style.INK_SECONDARY))
        self.cursor = _cursor()
        self.addItem(self.cursor, ignoreBounds=True)
        self._face_lines: list = []

    def set_mesh(self, mesh) -> None:
        for item in self._face_lines:
            self.removeItem(item)
        self._face_lines.clear()
        for y in mesh.edges[1:-1] / u.CM:
            line = pg.InfiniteLine(pos=y, angle=0, movable=False,
                                   pen=pg.mkPen(255, 255, 255, 140, width=1, style=Qt.PenStyle.DashLine))
            self.addItem(line, ignoreBounds=True)
            self._face_lines.append(line)

    def set_image(self, image: np.ndarray, t_end_min: float, perimeter_cm: float, deviation: bool, label: str):
        if deviation:
            limit = max(float(np.nanmax(np.abs(image))), 1e-9)
            levels, cmap = (-limit, limit), DEVIATION_CMAP
        else:
            lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
            levels, cmap = (lo, hi if hi > lo else lo + 1e-9), THICKNESS_CMAP
        self.colorbar.setColorMap(cmap)
        self.colorbar.setLevels(levels)
        self.colorbar.axis.setLabel(label, color=style.INK_SECONDARY)
        self.image.setImage(image, autoLevels=False, levels=levels)
        self.image.setRect(0.0, 0.0, max(t_end_min, 1e-9), perimeter_cm)
        self.setXRange(0.0, max(t_end_min, 1e-9), padding=0)
        self.setYRange(0.0, perimeter_cm, padding=0)

    def clear_image(self) -> None:
        self.image.clear()


def _resample(times, profiles, mesh, t_end):
    """비균일 시각·요소 길이의 이력을 균일 격자 (s 행 × t 열) 로 바꾼다 (시간은 선형 보간, 위치는 요소 값)."""
    times = np.asarray(times, dtype=float)
    profiles = np.asarray(profiles, dtype=float)
    columns = int(min(HEATMAP_COLUMNS, max(4 * len(times), 2)))
    t_grid = np.linspace(0.0, t_end, columns)
    if len(times) < 2:
        grid = np.repeat(profiles[:1], columns, axis=0)
    else:
        hi = np.clip(np.searchsorted(times, t_grid), 1, len(times) - 1)
        lo = hi - 1
        span = times[hi] - times[lo]
        w = np.clip(np.divide(t_grid - times[lo], span, out=np.zeros_like(t_grid), where=span > 0), 0.0, 1.0)
        grid = profiles[lo] * (1.0 - w)[:, np.newaxis] + profiles[hi] * w[:, np.newaxis]
    bounds = np.concatenate([[0.0], np.cumsum(mesh.length)])
    s_grid = (np.arange(HEATMAP_ROWS) + 0.5) / HEATMAP_ROWS * bounds[-1]
    si = np.clip(np.searchsorted(bounds, s_grid, side="right") - 1, 0, mesh.size - 1)
    return grid[:, si].T


def _colour_swatch(thickness: float) -> str:
    """판독 줄용 간섭색 견본 (색 사각형 + 이름). 공백은 _set_readout 이 줄바꿈 금지 공백으로 바꾼다."""
    rgb = optics.interference_color(thickness)[0]
    return (f"<span style='color:{optics.to_hex(rgb)}; font-size:14pt'>■</span>"
            f" 간섭색 {optics.color_name(rgb)}")


class ResultsView(QWidget):
    time_selected = pyqtSignal(float)   # 판독 시각 [s] (마우스를 올린 위치, 최종이면 끝 시각)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._live = None
        self._mesh = None
        self._regime = "porous"
        self._t_min = None
        self._selected = None   # 위치별 분포를 보여줄 이력 인덱스 (None = 최신)
        self._thermal = False
        self._interference = False   # Ti 치밀막: 판독 줄에 예상 간섭색
        self._step_count = 1         # 전원 프로그램 단계 수 (반복 포함), 1 이면 표시하지 않음

        self.readout = QLabel("‘계산 실행’을 누르면 시간에 따른 전압·전류·두께가 여기에 표시됩니다.")
        self.readout.setTextFormat(Qt.TextFormat.RichText)
        self.readout.setWordWrap(True)
        self.readout.setStyleSheet(f"color: {style.INK_SECONDARY}; padding: 2px 4px;")
        self.notices = QLabel()
        self.notices.setWordWrap(True)
        self.notices.setVisible(False)

        # ── 시간 이력 ───────────────────────────────────────────────────────
        self.voltage = _Plot("전압", "전압 [V]", show_time_axis=False)
        self.current = _Plot("전류밀도 j (양극 평균)", "j [A/dm²]", show_time_axis=False)
        self.thickness = _Plot("산화막 두께", "두께 [μm]", show_time_axis=False)
        self.temperature = _Plot("욕 온도 (열수지)", "온도 [°C]", show_time_axis=False)
        self.temperature.setVisible(False)
        self.barrier = _Plot("배리어층 두께 δ_b", "δ_b [nm]", show_time_axis=True)
        self.plots = (self.voltage, self.current, self.thickness, self.temperature, self.barrier)
        for p in self.plots[1:]:
            p.setXLink(self.voltage)
        history = QWidget()
        h_layout = QVBoxLayout(history)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(2)
        for p in self.plots:
            h_layout.addWidget(p, 1)

        # ── 위치별 분포 ─────────────────────────────────────────────────────
        self.profile_h = _ProfilePlot("산화막 두께 h(s)", "두께 [μm]", show_s_axis=False)
        self.profile_j = _ProfilePlot("전류밀도 j(s)", "j [A/dm²]", show_s_axis=True)
        self.profile_j.setXLink(self.profile_h)
        self.heatmap = _Heatmap()
        self.heat_mode = QComboBox()
        self.heat_mode.addItem("두께 h(s, t)", False)
        self.heat_mode.addItem("평균 대비 편차 (h − 평균)/평균", True)
        heat_bar = QHBoxLayout()
        heat_title = QLabel("히트맵")
        heat_title.setStyleSheet(f"color: {style.INK};")
        heat_bar.addWidget(heat_title)
        heat_bar.addWidget(self.heat_mode)
        heat_bar.addStretch(1)
        self.profile_hint = QLabel()
        self.profile_hint.setStyleSheet(f"color: {style.INK_SECONDARY};")
        self.profile_hint.setWordWrap(True)
        distribution = QWidget()
        d_layout = QVBoxLayout(distribution)
        d_layout.setContentsMargins(0, 0, 0, 0)
        d_layout.setSpacing(2)
        d_layout.addWidget(self.profile_hint)
        d_layout.addWidget(self.profile_h, 3)
        d_layout.addWidget(self.profile_j, 3)
        d_layout.addLayout(heat_bar)
        d_layout.addWidget(self.heatmap, 4)

        self.tabs = QTabWidget()
        self.tabs.addTab(history, "시간 이력")
        self.tabs.addTab(distribution, "위치별 분포")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        layout.addWidget(self.readout)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.notices)

        for p in self.plots:
            p.scene().sigMouseMoved.connect(lambda pos, plot=p: self._on_mouse(plot, pos))
        self.heatmap.scene().sigMouseMoved.connect(lambda pos: self._on_mouse(self.heatmap, pos))
        self.heat_mode.currentIndexChanged.connect(lambda _: self._draw_heatmap())
        self.tabs.currentChanged.connect(self._on_tab)
        self._set_distribution_enabled(False)

    # ── 공개 API ────────────────────────────────────────────────────────────
    def clear(self) -> None:
        self._result = None
        self._live = None
        self._selected = None
        for p in self.plots:
            p.set_series([], [])
            p.cursor.setVisible(False)
        self.profile_h.set_values([], [])
        self.profile_j.set_values([], [])
        self.heatmap.clear_image()
        self.heatmap.cursor.setVisible(False)
        self.notices.setVisible(False)

    def start_live(self, mesh, regime: str, duration: float, thermal: bool = False, interference: bool = False,
                   step_count: int = 1) -> None:
        """계산 시작: 실시간 갱신용 버퍼와 그래프 구성을 준비한다."""
        self.clear()
        self._regime = regime
        self._set_options(thermal, interference, step_count)
        self._set_mesh(mesh)
        self._live = {k: [] for k in ("time", "voltage", "film_voltage", "current", "current_density", "total",
                                      "total_min", "total_max", "barrier", "temperature", "control", "step",
                                      "profile_total", "profile_current_density")}
        self._setup_series(np.zeros(0), {k: np.zeros(0) for k in ("voltage", "film_voltage", "current_density",
                                                                   "total", "total_min", "total_max", "barrier",
                                                                   "temperature")})
        # 계산 중에는 시간축을 공정 시간 전체로 고정한다 (축을 매번 다시 그리지 않아 가볍고, 진행 정도도 보인다)
        for p in self.plots:
            p.enableAutoRange(axis="x", enable=False)
        self.voltage.setXRange(0.0, duration / u.MINUTE, padding=0.02)
        self.readout.setText("계산 중…")

    def add_live(self, snap) -> None:
        live = self._live
        if live is None:
            return
        w = self._mesh.area / self._mesh.area.sum()
        live["time"].append(snap.time)
        live["voltage"].append(snap.voltage)
        live["film_voltage"].append(float(w @ snap.film_voltage))
        live["current"].append(snap.current)
        live["current_density"].append(float(w @ snap.current_density))
        live["total"].append(float(w @ snap.total))
        live["total_min"].append(float(snap.total.min()))
        live["total_max"].append(float(snap.total.max()))
        live["barrier"].append(float(w @ snap.barrier))
        live["temperature"].append(snap.temperature)
        live["control"].append(snap.control)
        live["step"].append(snap.step)
        live["profile_total"].append(snap.total)
        live["profile_current_density"].append(snap.current_density)

        t = np.asarray(live["time"]) / u.MINUTE
        a = {k: np.asarray(live[k]) for k in ("voltage", "film_voltage", "current_density", "total",
                                              "total_min", "total_max", "barrier", "temperature")}
        self.voltage.update_series(t, [a["voltage"], a["film_voltage"]])
        self.current.update_series(t, [a["current_density"] / u.A_PER_DM2])
        if self._distributed:
            self.thickness.update_series(t, [a["total"] / u.UM], band=(a["total_min"] / u.UM, a["total_max"] / u.UM))
        else:
            self.thickness.update_series(t, [a["total"] / u.UM])
        self.barrier.update_series(t, [a["barrier"] / u.NM])
        if self._thermal:
            self.temperature.update_series(t, [u.kelvin_to_celsius(a["temperature"])])
        if self._distributed and self.tabs.currentIndex() == 1:   # 보이지 않는 탭은 탭을 열 때 그린다
            self._show_profile(snap.time, snap.total, snap.current_density)
            self._draw_heatmap()
        self._show_live_readout(snap)

    def show_result(self, result, regime: str, stopped: bool = False, interference: bool = False,
                    step_count: int = 1) -> None:
        self._result = result
        self._live = None
        self._regime = regime
        self._selected = None
        self._set_options(result.thermal, interference, step_count)
        self._set_mesh(result.mesh)
        t = result.time / u.MINUTE
        self._t_min = t
        self._setup_series(t, {"voltage": result.voltage, "film_voltage": result.film_voltage,
                               "current_density": result.current_density, "total": result.total,
                               "total_min": result.total_min, "total_max": result.total_max,
                               "barrier": result.barrier, "temperature": result.temperature})
        if self._distributed:
            f = result.final
            self._show_profile(f.time, f.total, f.current_density)
            self._draw_heatmap()
        self._show_readout(result.time.size - 1, final=True)
        self._show_notices(result, stopped)

    def _on_tab(self, index: int) -> None:
        if index != 1 or self._live is None or not self._live["time"]:
            return
        live = self._live
        self._show_profile(live["time"][-1], live["profile_total"][-1], live["profile_current_density"][-1])
        self._draw_heatmap()

    # ── 내부: 구성 ──────────────────────────────────────────────────────────
    def _set_options(self, thermal: bool, interference: bool, step_count: int) -> None:
        self._thermal = thermal
        self._interference = interference
        self._step_count = step_count
        self.temperature.setVisible(thermal)

    @property
    def _distributed(self) -> bool:
        return self._mesh is not None and self._mesh.size > 1

    def _set_mesh(self, mesh) -> None:
        self._mesh = mesh
        self._set_distribution_enabled(mesh.size > 1)
        if mesh.size > 1:
            self.profile_h.set_mesh(mesh)
            self.profile_j.set_mesh(mesh)
            self.heatmap.set_mesh(mesh)

    def _set_distribution_enabled(self, enabled: bool) -> None:
        self.tabs.setTabEnabled(1, enabled)
        if not enabled:
            self.tabs.setCurrentIndex(0)
            self.tabs.setTabToolTip(1, "해석 단계 L1 에서 계산하면 위치별 두께 분포가 표시됩니다.")
        else:
            self.tabs.setTabToolTip(1, "")

    def _setup_series(self, t, a) -> None:
        porous = self._regime == "porous"
        self.voltage.set_series(t, [("인가 전압 U", a["voltage"]), ("막 전압 U_f (평균)", a["film_voltage"])])
        self.current.set_series(t, [("j", a["current_density"] / u.A_PER_DM2)])
        title = "산화막 총 두께 h (배리어층 + 기공층)" if porous else "산화막 두께 h"
        if self._distributed:
            title += " — 평균과 위치별 범위"
            self.thickness.set_series(t, [("평균", a["total"] / u.UM)],
                                      band=("최소–최대", a["total_min"] / u.UM, a["total_max"] / u.UM))
        else:
            self.thickness.set_series(t, [("h", a["total"] / u.UM)])
        self.thickness.setTitle(title, color=style.INK, size="10pt")
        self.barrier.set_series(t, [("δ_b", a["barrier"] / u.NM)])
        if self._thermal:
            self.temperature.set_series(t, [("T", u.kelvin_to_celsius(a["temperature"]))])
        else:
            self.temperature.set_series([], [])

    # ── 내부: 위치별 분포 ───────────────────────────────────────────────────
    def _history(self):
        """(시각 [s], 요소별 두께, 요소별 전류밀도) — 실시간이면 지금까지 받은 스냅샷."""
        if self._live is not None:
            live = self._live
            if not live["time"]:
                return None
            return np.asarray(live["time"]), live["profile_total"], live["profile_current_density"]
        if self._result is not None:
            r = self._result
            return r.profile_time, r.profile_total, r.profile_current_density
        return None

    def _show_profile(self, time_s, total, current_density) -> None:
        mesh = self._mesh
        s = mesh.s / u.CM
        self.profile_h.set_values(s, total / u.UM)
        self.profile_j.set_values(s, current_density / u.A_PER_DM2)
        mean = float(mesh.area @ total / mesh.area.sum())
        spread = (total.max() - total.min()) / mean * 100.0 if mean > 0 else 0.0
        self.profile_hint.setText(f"t = {time_s / u.MINUTE:.2f} min · 평균 {mean / u.UM:.4g} μm · "
                                  f"최소 {total.min() / u.UM:.4g} · 최대 {total.max() / u.UM:.4g} μm · "
                                  f"편차 (최대−최소)/평균 {spread:.2f} % — 시간 이력·히트맵 위에 마우스를 올리면 그 시각의 분포")
        self.heatmap.cursor.setPos(time_s / u.MINUTE)
        self.heatmap.cursor.setVisible(True)

    def _draw_heatmap(self) -> None:
        history = self._history()
        if history is None or not self._distributed:
            return
        times, totals, _ = history
        t_end = float(times[-1]) if self._live is not None else float(self._result.final.time)
        image = _resample(times, totals, self._mesh, t_end)
        deviation = bool(self.heat_mode.currentData())
        if deviation:
            col_mean = self._column_means(image)
            with np.errstate(invalid="ignore", divide="ignore"):
                image = np.where(col_mean > 0, (image / col_mean - 1.0) * 100.0, 0.0)
            label = "편차 [%]"
        else:
            image = image / u.UM
            label = "두께 [μm]"
        self.heatmap.set_image(image, t_end / u.MINUTE, self._mesh.perimeter / u.CM, deviation, label)

    def _column_means(self, image: np.ndarray) -> np.ndarray:
        """히트맵 열(시각)마다 면적 가중 평균 — 행은 둘레 균일 격자이므로 요소 깊이(면적/길이)로 가중."""
        mesh = self._mesh
        bounds = np.concatenate([[0.0], np.cumsum(mesh.length)])
        s_grid = (np.arange(HEATMAP_ROWS) + 0.5) / HEATMAP_ROWS * bounds[-1]
        si = np.clip(np.searchsorted(bounds, s_grid, side="right") - 1, 0, mesh.size - 1)
        depth = (mesh.area / mesh.length)[si]
        return (depth @ image / depth.sum())[np.newaxis, :]

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
        if self._distributed:
            r = self._result
            m = int(np.clip(np.searchsorted(r.profile_time, r.time[k]), 0, r.profile_time.size - 1))
            if m > 0 and abs(r.profile_time[m - 1] - r.time[k]) < abs(r.profile_time[m] - r.time[k]):
                m -= 1
            self._show_profile(r.profile_time[m], r.profile_total[m], r.profile_current_density[m])
            self.time_selected.emit(float(r.profile_time[m]))

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
        ]
        if self._distributed:
            parts.append(f"({r.total_min[k] / u.UM:.3g}–{r.total_max[k] / u.UM:.3g})")
        parts.append(f"δ_b = {r.barrier[k] / u.NM:.1f} nm")
        if self._thermal:
            parts.append(f"욕 {u.kelvin_to_celsius(r.temperature[k]):.1f} °C")
        if self._step_count > 1:
            parts.append(f"단계 {int(r.step[k]) + 1}/{self._step_count}")
        parts.append(CONTROL_LABELS.get(r.control[k], r.control[k]))
        if self._interference:
            parts.append(_colour_swatch(r.total[k]))
        if final and self._distributed:
            parts.append(f"편차 {r.uniformity * 100:.2f} %")
        if final and np.isfinite(r.wagner):
            parts.append(f"Wa = {r.wagner:.3g}")
        self._set_readout(parts)

    def _show_live_readout(self, snap) -> None:
        live = self._live
        parts = [f"<b>계산 중 t = {snap.time / u.MINUTE:.2f} min</b>", f"U = {snap.voltage:.2f} V",
                 f"I = {snap.current:.3g} A", f"j = {live['current_density'][-1] / u.A_PER_DM2:.3g} A/dm²",
                 f"h = {live['total'][-1] / u.UM:.3g} μm"]
        if self._distributed:
            parts.append(f"({live['total_min'][-1] / u.UM:.3g}–{live['total_max'][-1] / u.UM:.3g})")
        if self._thermal:
            parts.append(f"욕 {u.kelvin_to_celsius(snap.temperature):.1f} °C")
        if self._step_count > 1:
            parts.append(f"단계 {snap.step + 1}/{self._step_count}")
        parts.append(CONTROL_LABELS.get(snap.control, snap.control))
        if self._interference:
            parts.append(_colour_swatch(live["total"][-1]))
        self._set_readout(parts)

    def _set_readout(self, parts) -> None:
        # 값과 단위가 줄바꿈으로 갈라지지 않도록 항목 안의 공백은 줄바꿈 금지 공백으로 (HTML 태그 안은 그대로)
        self.readout.setText(" │ ".join(_TAG_OR_SPACE.sub(lambda m: m.group(1) or "&nbsp;", p) for p in parts))

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
