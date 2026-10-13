"""메인 창: 왼쪽 입력 도크, 가운데 형상 보기(측면도/3D) + 결과 그래프, 파일 메뉴(프로젝트·내보내기)."""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt, QThread, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (QComboBox, QDockWidget, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                             QProgressBar, QPushButton, QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from .. import __version__
from ..core import limits, optics
from ..core import units as u
from ..core.presets import PresetDB
from ..core.simulator import ConfigError, Simulator
from ..core.state import ProcessConfig
from ..io import (PROJECT_SUFFIX, ProjectError, config_to_dict, load_project, save_project, write_profile_csv,
                  write_timeseries_csv)
from . import style
from .panels import CathodePanel, ElectrolytePanel, GeometryPanel, ProcessPanel
from .views import Geometry3DView, ResultsView, SideViewPanel
from .views.colormap import value_range
from .views.view3d import VIEW_PRESETS
from .worker import SimulationWorker

# 배속 = 공정 시간 / 실제 시간
SPEEDS = (("최대 속도", math.inf), ("×3600 (1 h → 1 s)", 3600.0), ("×600 (1 h → 6 s)", 600.0),
          ("×60 (1 h → 1 min)", 60.0), ("×10", 10.0), ("×1 (실시간)", 1.0))
THICKNESS_LABEL = "산화막 두께 [μm]"
INTERFERENCE_LABEL = "Ti 간섭색 — 막 두께 [nm]"
PROJECT_FILTER = f"아노다이징 프로젝트 (*{PROJECT_SUFFIX});;JSON (*.json);;모든 파일 (*)"
LIVE_INTERVAL_MS = 33        # 배속 재생 중 화면 갱신 (≈30 Hz)
LIVE_INTERVAL_FAST_MS = 100  # 최대 속도: 그리기가 계산과 CPU 를 나눠 쓰므로 10 Hz


def _scroll(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    return area


class MainWindow(QMainWindow):
    def __init__(self, presets: PresetDB | None = None):
        super().__init__()
        self.presets = presets or PresetDB.default()
        self.resize(1500, 940)
        self._thread: QThread | None = None
        self._worker: SimulationWorker | None = None
        self._simulator: Simulator | None = None
        self._overlay = None   # 마지막으로 칠한 (요소별 두께, 요소별 전류밀도) — 다시 칠하기용
        self._result = None    # 마지막 계산 결과 (내보내기용)
        self._project_path: Path | None = None
        self._saved_state = None    # 마지막으로 저장·불러온 설정 (제목 줄의 '수정됨' 표시)
        # 화면에 칸이 없는 설정 값 — 프로젝트 파일에서 읽은 값을 다시 저장할 때 지킨다
        self._hidden = {"elements": 120, "offset_voltage": 0.0, "external_resistance": 0.0}
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._pull_live)

        # ── 입력 도크 ────────────────────────────────────────────────────────
        self.process = ProcessPanel(self.presets)
        self.electrolyte = ElectrolytePanel(self.presets)
        self.cathode = CathodePanel(self.presets)
        self.geometry = GeometryPanel()
        self.input_tabs = QTabWidget()
        for panel, title in ((self.process, "공정"), (self.electrolyte, "전해질"),
                             (self.cathode, "음극"), (self.geometry, "배치")):
            self.input_tabs.addTab(_scroll(panel), title)

        self.messages = QLabel()
        self.messages.setWordWrap(True)
        self.messages.setTextFormat(Qt.TextFormat.RichText)
        self.run_button = QPushButton("계산 실행")
        self.run_button.setDefault(True)
        self.run_button.setMinimumHeight(34)
        self.stop_button = QPushButton("중지")
        self.stop_button.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(8)

        self.speed = QComboBox()
        for label, value in SPEEDS:
            self.speed.addItem(label, value)
        self.speed.setToolTip("공정 시간 / 실제 시간. 계산 중에도 바꿀 수 있습니다.")

        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button, 1)
        buttons.addWidget(self.stop_button)
        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("배속"))
        speed_row.addWidget(self.speed, 1)
        dock_body = QWidget()
        dock_layout = QVBoxLayout(dock_body)
        dock_layout.addWidget(self.input_tabs, 1)
        dock_layout.addWidget(self.messages)
        dock_layout.addLayout(speed_row)
        dock_layout.addLayout(buttons)
        dock_layout.addWidget(self.progress)
        dock = QDockWidget("공정 조건")
        dock.setObjectName("inputs")
        dock.setWidget(dock_body)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        dock.setMinimumWidth(380)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        # ── 가운데: 형상 보기 + 결과 ────────────────────────────────────────
        self.side_panel = SideViewPanel()
        self.side_view = self.side_panel.view
        self.view3d = Geometry3DView()
        self.view_tabs = QTabWidget()
        self.view_tabs.addTab(self.side_panel, "측면도 (2D)")
        self.view_tabs.addTab(self.view3d, "3D 보기")
        self.results = ResultsView()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.view_tabs)
        splitter.addWidget(self.results)
        splitter.setStretchFactor(0, 45)
        splitter.setStretchFactor(1, 55)
        self._splitter = splitter
        self._sized = False
        self.setCentralWidget(splitter)

        self.status = QLabel()
        self.statusBar().addWidget(self.status, 1)

        # ── 연결 ────────────────────────────────────────────────────────────
        self.process.metal_changed.connect(self._on_metal)
        self.electrolyte.preset_changed.connect(self._on_preset)
        for panel in (self.process, self.electrolyte, self.cathode):
            panel.changed.connect(self._refresh)
        self.geometry.changed.connect(self._on_geometry)
        self.run_button.clicked.connect(self.run_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.speed.currentIndexChanged.connect(self._on_speed)
        self.results.time_selected.connect(self._on_time_selected)
        self.side_view.electrode_moved.connect(self._on_electrode_moved)
        self.side_panel.show_potential.toggled.connect(lambda _: self._repaint_overlay())
        self.side_panel.colour_mode.currentIndexChanged.connect(lambda _: self._repaint_overlay())

        self._build_menu()
        self._on_metal(self.process.metal_symbol())
        self._default_config = self.build_config()
        self._mark_saved()

    # ── 입력 변화 ───────────────────────────────────────────────────────────
    def _on_metal(self, metal: str) -> None:
        self.side_view.set_metal(metal)
        self.view3d.set_metal(metal)
        self.electrolyte.set_metal(metal)      # → preset_changed → _on_preset
        self._on_geometry()

    def _on_preset(self, preset_id: str) -> None:
        preset = self.presets.electrolyte(preset_id)
        self.cathode.set_electrolyte(preset)
        self.process.set_temperature_c(u.kelvin_to_celsius(preset.temperature_ref))
        self.process.voltage.setValue(preset.typical_voltage)
        self._refresh()

    def _on_geometry(self) -> None:
        g = self.geometry.geometry()
        self.side_view.show_geometry(g)
        self.view3d.show_geometry(g)
        self._refresh()
        self._repaint_overlay()   # 제자리로 돌아온 경우 등 형상이 결과와 같으면 색칠을 되살린다

    def _on_electrode_moved(self, kind: str, index: int, x_center: float, z_bottom: float) -> None:
        """측면도에서 끌어 놓은 위치를 배치 탭 입력에 반영한다 (→ changed → _on_geometry)."""
        if not self.geometry.set_electrode_position(kind, index, x_center, z_bottom):
            self._on_geometry()   # 같은 자리에 놓음: 끄는 동안 감춘 표시를 다시 그린다

    def _refresh(self) -> None:
        """입력 검증, 파생 값(전류밀도 환산·면적·저항·열용량·예상 간섭색) 갱신."""
        g = self.geometry.geometry()
        geometry_ok = not g.validate()
        self.process.set_anode_area(g.anode_area if geometry_ok else 0.0)
        self.cathode.set_areas(g.anode_area, g.cathode_area)
        self.electrolyte.set_temperature(self.process.temperature_k())

        config = self.build_config()
        if config is None:
            return
        errors = config.validate(self.presets)
        warnings = []
        preset = self.electrolyte.current_preset()
        if not errors:
            warnings = limits.range_warnings(config, preset, self.presets.cathode(config.cathode_material))
            kappa = preset.conductivity(config.bath_temperature, config.electrolyte.concentration)
            r = g.lumped_resistance(kappa)
            r_text = f"{r:.3g} Ω" if r >= 1.0 else f"{r * 1e3:.3g} mΩ"
            self.status.setText(f"양극 {g.anode_area / u.CM ** 2:.0f} cm² · 음극 {g.cathode_area / u.CM ** 2:.0f} cm² · "
                                f"전해질 κ {kappa:.3g} S/m · 전해질 저항(L0 등가) {r_text}")
        else:
            self.status.setText("입력을 확인하세요.")
        self._update_derived(config, preset, g)
        self._show_messages(errors, warnings)
        self.run_button.setEnabled(not errors and self._thread is None)
        self._update_title()

    def _update_derived(self, config, preset, g) -> None:
        """열수지 안내(욕 열용량·시정수)와 Ti 치밀막 예상 간섭색."""
        if preset is None:
            return
        volume = g.liquid_volume()
        capacity = preset.density * preset.heat_capacity * volume
        ua = self.process.cooling.value()
        if ua <= 0:
            tau = "단열 (냉각 없음)"
        elif capacity / ua < u.MINUTE:
            tau = f"냉각 시정수 {capacity / ua:.2g} s"
        else:
            tau = f"냉각 시정수 {capacity / ua / u.MINUTE:.0f} min"
        self.process.set_thermal_info(f"욕 {volume * 1e3:.1f} L · 열용량 {capacity / 1e3:.0f} kJ/K · {tau}")
        if config.metal == "Ti" and preset.regime == "compact":
            metal = self.presets.metal(config.metal)
            thickness = metal.native_oxide_thickness + preset.anodizing_ratio * config.power.max_voltage
            rgb = optics.interference_color(thickness)[0]
            self.process.set_colour_preview(
                f"예상 간섭색 (최고 전압에서 막 두께 ≈ {thickness / u.NM:.0f} nm): "
                f"<span style='color:{optics.to_hex(rgb)}; font-size:15pt'>■</span> {optics.color_name(rgb)}")
        else:
            self.process.set_colour_preview(None)

    def _show_messages(self, errors, warnings) -> None:
        lines = [f"<span style='color:{style.STATUS_CRITICAL}'>■ 오류: {html.escape(e)}</span>" for e in errors]
        lines += [f"<span style='color:{style.INK_SECONDARY}'>△ 주의: {html.escape(w)}</span>" for w in warnings]
        self.messages.setText("<br>".join(lines))
        self.messages.setVisible(bool(lines))

    def build_config(self) -> ProcessConfig | None:
        recipe = self.electrolyte.recipe()
        if recipe.preset_id is None:
            return None
        return ProcessConfig(
            metal=self.process.metal_symbol(),
            electrolyte=recipe,
            cathode_material=self.cathode.material_id(),
            power=self.process.power_setting(),
            geometry=self.geometry.geometry(),
            bath_temperature=self.process.temperature_k(),
            duration=self.process.duration_s(),
            fidelity=self.process.fidelity_level(),
            thermal=self.process.thermal_setting(),
            **self._hidden,
        )

    def apply_config(self, config: ProcessConfig) -> None:
        """설정을 입력 패널에 넣는다 (프로젝트 열기·새 프로젝트). 연쇄 신호 순서: 금속 → 전해질 → 음극 → 공정 → 배치."""
        metal_index = self.process.metal.findData(config.metal)
        if metal_index < 0:
            raise ValueError(f"알 수 없는 금속: {config.metal}")
        if metal_index != self.process.metal.currentIndex():
            self.process.metal.setCurrentIndex(metal_index)        # → 전해질 목록·기본값
        self.electrolyte.set_recipe(config.electrolyte.preset_id, config.electrolyte.concentration)
        self.cathode.set_material(config.cathode_material)
        self.process.set_power(config.power, config.duration)    # 프리셋 기본 전압·온도를 덮어쓴다
        self.process.set_temperature_c(u.kelvin_to_celsius(config.bath_temperature))
        self.process.set_fidelity(config.fidelity)
        self.process.set_thermal(config.thermal)
        self._hidden = {"elements": config.elements, "offset_voltage": config.offset_voltage,
                        "external_resistance": config.external_resistance}
        self.geometry.set_geometry(config.geometry)              # → _on_geometry → _refresh

    # ── 실행 ────────────────────────────────────────────────────────────────
    def run_simulation(self) -> None:
        if self._thread is not None:
            return
        config = self.build_config()
        try:
            self._simulator = Simulator(config, self.presets)
        except ConfigError as exc:
            self._show_messages(exc.errors, [])
            return
        self._set_running(True)
        sim = self._simulator
        self._overlay = None          # 이전 실행의 색칠은 새 결과가 올 때까지 남기지 않는다
        self.side_view.clear_profile()
        self.side_view.clear_potential()
        self.view3d.clear_profile()
        self.side_panel.show_potential.setEnabled(sim.field.has_potential)
        ti_compact = self._ti_compact(sim)
        self.side_panel.colour_mode.setEnabled(ti_compact)
        self.side_panel.colour_mode.setCurrentIndex(1 if ti_compact else 0)
        self.results.start_live(sim.mesh, sim.props.regime, config.duration, thermal=sim.thermal is not None,
                                interference=ti_compact, step_count=self._step_count(sim))
        self._thread = QThread(self)
        self._worker = SimulationWorker(sim, self.speed.currentData())
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._set_live_interval()
        self._live_timer.start()
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def stop_simulation(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()

    def _on_speed(self, _index: int) -> None:
        if self._worker is not None:
            self._worker.set_speed(self.speed.currentData())
            self._set_live_interval()

    def _set_live_interval(self) -> None:
        fast = not math.isfinite(self.speed.currentData())
        self._live_timer.setInterval(LIVE_INTERVAL_FAST_MS if fast else LIVE_INTERVAL_MS)

    def _pull_live(self) -> None:
        """타이머마다 워커의 최신 스냅샷을 가져와 그린다 (그리기가 느리면 타이머가 알아서 덜 불린다)."""
        worker = self._worker
        if worker is None:
            return
        self.progress.setValue(int(1000 * worker.progress))
        snap = worker.take_snapshot()
        if snap is not None:
            self.results.add_live(snap)
            self._paint_overlay(snap.total, snap.current_density)

    def _on_time_selected(self, time_s: float) -> None:
        r = self.results._result
        if r is None:
            return
        k = int(min(max(r.profile_time.searchsorted(time_s), 0), r.profile_time.size - 1))
        self._paint_overlay(r.profile_total[k], r.profile_current_density[k])

    @staticmethod
    def _ti_compact(sim) -> bool:
        """Ti 치밀막(착색 아노다이징) — 막 두께가 간섭색을 정한다."""
        return sim.metal.symbol == "Ti" and sim.props.regime == "compact"

    @staticmethod
    def _step_count(sim) -> int:
        program = sim.supply.program
        return program.repeat * len(program.steps)

    def _paint_overlay(self, total, current_density) -> None:
        """측면도·3D 양극 표면을 요소별 두께(또는 Ti 간섭색)로, L2 면 측면도 전해질을 전위로 색칠한다."""
        self._overlay = (total, current_density)
        sim = self._simulator
        self.side_panel.show_potential.setEnabled(sim is not None and sim.field.has_potential)
        # 계산 뒤 형상을 바꿨으면 이전 결과의 요소가 새 형상과 맞지 않으므로 칠하지 않는다
        if sim is None or sim.mesh.size < 2 or sim.config.geometry != self.geometry.geometry():
            return
        mesh = sim.mesh
        if self.side_panel.interference:
            values = total / u.NM
            vmin, vmax = value_range(values)
            rgb = optics.interference_color(total)
            legend = [optics.to_hex(c) for c in optics.interference_color(np.linspace(vmin, vmax, 9) * u.NM)]
            for view in (self.side_view, self.view3d):
                view.set_profile(mesh, values, vmin, vmax, INTERFERENCE_LABEL, rgb=rgb, legend_colors=legend)
        else:
            values = total / u.UM
            vmin, vmax = value_range(values)
            self.side_view.set_profile(mesh, values, vmin, vmax, THICKNESS_LABEL)
            self.view3d.set_profile(mesh, values, vmin, vmax, THICKNESS_LABEL)
        if sim.field.has_potential and self.side_panel.show_potential.isChecked():
            image, extent = sim.field.potential_image(current_density * mesh.area, sim.props.conductivity)
            peak = float(np.nanmax(image)) if np.any(np.isfinite(image)) else 0.0
            self.side_view.set_potential(image, extent, max(peak, 1e-6))
        else:
            self.side_view.clear_potential()

    def _repaint_overlay(self) -> None:
        if self._overlay is not None:
            self._paint_overlay(*self._overlay)

    def _on_finished(self, result, stopped: bool) -> None:
        self._cleanup_thread()
        self.progress.setValue(1000)
        sim = self._simulator
        self._result = result
        self.results.show_result(result, sim.props.regime, stopped, interference=self._ti_compact(sim),
                                 step_count=self._step_count(sim))
        self._update_actions()
        f = result.final
        self._paint_overlay(f.total, f.current_density)
        mean = float(result.total[-1])
        spread = f" (위치별 {f.total.min() / u.UM:.3g}–{f.total.max() / u.UM:.3g} μm)" if result.mesh.size > 1 else ""
        self.statusBar().showMessage(
            f"계산 완료 — {result.steps} 스텝, 평균 두께 {mean / u.UM:.3g} μm{spread}, "
            f"배리어 {result.barrier[-1] / u.NM:.1f} nm", 8000)

    def _on_failed(self, message: str) -> None:
        self._cleanup_thread()
        QMessageBox.critical(self, "계산 오류", message)

    def _cleanup_thread(self) -> None:
        self._live_timer.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
        if self._worker is not None:
            self._worker.deleteLater()
        self._thread = None
        self._worker = None
        self._set_running(False)
        self._refresh()

    def _set_running(self, running: bool) -> None:
        self.input_tabs.setEnabled(not running)
        self.side_view.set_editable(not running)
        for action in (self.action_new, self.action_open):
            action.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if running:
            self.progress.setValue(0)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._sized:   # 처음 표시될 때 (레이아웃이 끝난 뒤) 형상 보기 : 결과 = 45 : 55
            self._sized = True
            QTimer.singleShot(0, self._initial_split)

    def _initial_split(self) -> None:
        total = sum(self._splitter.sizes())
        self._splitter.setSizes([int(0.45 * total), total - int(0.45 * total)])
        self.view3d.set_view(*VIEW_PRESETS[0][1:])

    # ── 파일 메뉴 ───────────────────────────────────────────────────────────
    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("파일(&F)")

        def action(text, slot, shortcut=None):
            a = QAction(text, self)
            if shortcut is not None:
                a.setShortcut(shortcut)
            a.triggered.connect(slot)
            return a

        self.action_new = action("새 프로젝트", self.new_project, QKeySequence.StandardKey.New)
        self.action_open = action("열기…", self.open_project, QKeySequence.StandardKey.Open)
        self.action_save = action("저장", self.save_project, QKeySequence.StandardKey.Save)
        self.action_save_as = action("다른 이름으로 저장…", self.save_project_as, QKeySequence.StandardKey.SaveAs)
        for a in (self.action_new, self.action_open, self.action_save, self.action_save_as):
            menu.addAction(a)
        menu.addSeparator()
        export = menu.addMenu("내보내기")
        self.export_actions = {
            "series": action("시간 이력 CSV…", self.export_timeseries),
            "profile": action("위치별 분포 CSV (최종)…", self.export_profile),
            "results_png": action("결과 그래프 PNG…", lambda: self.export_png("results")),
            "side_png": action("측면도 PNG…", lambda: self.export_png("side")),
            "view3d_png": action("3D 보기 PNG…", lambda: self.export_png("3d")),
        }
        for a in self.export_actions.values():
            export.addAction(a)
        menu.addSeparator()
        menu.addAction(action("종료", self.close, QKeySequence.StandardKey.Quit))
        self._update_actions()

    def _update_actions(self) -> None:
        has_result = self._result is not None
        for key, a in self.export_actions.items():
            a.setEnabled(has_result if key in ("series", "profile", "results_png") else True)

    def _current_state(self):
        config = self.build_config()
        return None if config is None else config_to_dict(config)

    def _mark_saved(self) -> None:
        self._saved_state = self._current_state()
        self._update_title()

    def _update_title(self) -> None:
        name = self._project_path.name if self._project_path is not None else "새 프로젝트"
        dirty = getattr(self, "_saved_state", None) is not None and self._current_state() != self._saved_state
        self.setWindowTitle(f"{'*' if dirty else ''}{name} — 아노다이징 시뮬레이터 {__version__}")

    # 파일 대화상자 (테스트에서 바꿔 끼울 수 있게 메서드로 둔다)
    def _ask_open_path(self) -> str:
        return QFileDialog.getOpenFileName(self, "프로젝트 열기", self._dialog_dir(), PROJECT_FILTER)[0]

    def _ask_save_path(self, title: str, name: str, file_filter: str) -> str:
        return QFileDialog.getSaveFileName(self, title, str(Path(self._dialog_dir()) / name), file_filter)[0]

    def _dialog_dir(self) -> str:
        return str(self._project_path.parent) if self._project_path is not None else str(Path.home())

    def _stem(self) -> str:
        if self._project_path is None:
            return "anodizing"
        return self._project_path.name.removesuffix(PROJECT_SUFFIX).removesuffix(".json")

    def new_project(self) -> None:
        self.apply_config(self._default_config)
        self._project_path = None
        self._mark_saved()

    def open_project(self, path: str | None = None) -> bool:
        path = path or self._ask_open_path()
        if not path:
            return False
        previous = self.build_config()
        try:
            self.apply_config(load_project(path))
        except (OSError, ProjectError, ValueError) as exc:
            if previous is not None:
                self.apply_config(previous)   # 넣다 만 값(금속·전해질 등)을 원래대로 되돌린다
            QMessageBox.warning(self, "프로젝트 열기", f"프로젝트를 열 수 없습니다.\n{exc}")
            return False
        self._project_path = Path(path)
        self._mark_saved()
        self.statusBar().showMessage(f"프로젝트를 열었습니다: {self._project_path.name}", 5000)
        return True

    def save_project(self) -> bool:
        if self._project_path is None:
            return self.save_project_as()
        return self._write_project(self._project_path)

    def save_project_as(self) -> bool:
        path = self._ask_save_path("프로젝트 저장", self._stem() + PROJECT_SUFFIX, PROJECT_FILTER)
        if not path:
            return False
        if not path.endswith((".json",)):
            path += PROJECT_SUFFIX
        return self._write_project(Path(path))

    def _write_project(self, path: Path) -> bool:
        config = self.build_config()
        if config is None:
            return False
        try:
            save_project(path, config)
        except OSError as exc:
            QMessageBox.warning(self, "프로젝트 저장", f"저장하지 못했습니다.\n{exc}")
            return False
        self._project_path = path
        self._mark_saved()
        self.statusBar().showMessage(f"프로젝트를 저장했습니다: {path.name}", 5000)
        return True

    def export_timeseries(self) -> bool:
        return self._export("시간 이력 CSV 저장", "_시간이력.csv", "CSV (*.csv)",
                            lambda p: write_timeseries_csv(p, self._result))

    def export_profile(self) -> bool:
        return self._export("위치별 분포 CSV 저장", "_위치별분포.csv", "CSV (*.csv)",
                            lambda p: write_profile_csv(p, self._result))

    def export_png(self, what: str) -> bool:
        names = {"results": "_결과.png", "side": "_측면도.png", "3d": "_3D.png"}

        def write(path):
            if what == "results":
                image = self.results.grab()
            elif what == "side":
                image = self.side_panel.grab()
            else:
                view = self.view3d.view
                image = view.grabFramebuffer() if view is not None else self.view3d.grab()
            if not image.save(str(path)):
                raise OSError("이미지를 저장하지 못했습니다.")

        return self._export("그림 저장", names[what], "PNG (*.png)", write)

    def _export(self, title: str, suffix: str, file_filter: str, writer) -> bool:
        path = self._ask_save_path(title, self._stem() + suffix, file_filter)
        if not path:
            return False
        try:
            writer(path)
        except OSError as exc:
            QMessageBox.warning(self, title, f"저장하지 못했습니다.\n{exc}")
            return False
        self.statusBar().showMessage(f"저장했습니다: {Path(path).name}", 5000)
        return True

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.request_stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)
