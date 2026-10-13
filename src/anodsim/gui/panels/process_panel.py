"""공정 조건: 금속, 전원 모드(단일 설정 또는 다단계 프로그램), 전압·전류, 램프, 공정 시간, 욕 온도,
열수지(욕 온도 변화), 해석 단계."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QWidget

from ...core import units as u
from ...core.field import FIDELITY_LEVELS
from ...core.power_supply import PowerMode, PowerProgram, PowerSetting, PowerStep
from ...core.thermal import ThermalSetting
from .program_editor import ProgramEditor
from .widgets import hint, spin

PROGRAM = "PROGRAM"   # 전원 모드 콤보의 '프로그램' 항목
MODE_LABELS = (
    (PowerMode.CCCV, "CC→CV (정전류 후 정전압)"),
    (PowerMode.CV, "CV (정전압)"),
    (PowerMode.CC, "CC (정전류)"),
    (PROGRAM, "프로그램 (다단계·펄스)"),
)


class ProcessPanel(QWidget):
    changed = pyqtSignal()
    metal_changed = pyqtSignal(str)

    def __init__(self, presets, parent=None):
        super().__init__(parent)
        self._anode_area = 0.0
        self._coolant_edited = False
        self._last_mode = PowerMode.CCCV   # 프로그램으로 바꾸기 직전의 단일 모드 (첫 단계로 옮겨 적는다)

        self.metal = QComboBox()
        for m in presets.metals.values():
            self.metal.addItem(f"{m.symbol} — {m.name}", m.symbol)

        self.mode = QComboBox()
        for mode, label in MODE_LABELS:
            self.mode.addItem(label, mode)

        self.voltage = spin(15.0, 0.1, 1000.0, decimals=1, step=0.5, suffix="V")
        self.current = spin(5.0, 0.001, 1000.0, decimals=3, step=0.1, suffix="A")
        self.unlimited = QCheckBox("제한 없음")
        self.current_hint = hint()
        self._current_row = QHBoxLayout()
        self._current_row.addWidget(self.current, 1)
        self._current_row.addWidget(self.unlimited)
        self.program = ProgramEditor()

        self.ramp = spin(30.0, 0.0, 3600.0, decimals=0, step=5.0, suffix="s")
        self.duration = spin(60.0, 0.1, 1440.0, decimals=1, step=5.0, suffix="min")
        self.temperature = spin(20.0, -20.0, 100.0, decimals=1, step=0.5, suffix="°C")
        self.fidelity = QComboBox()
        for level, label in FIDELITY_LEVELS.items():
            self.fidelity.addItem(label, level)
        self.fidelity.setCurrentIndex(self.fidelity.findData("L1"))
        self.colour_preview = QLabel()
        self.colour_preview.setTextFormat(Qt.TextFormat.RichText)
        self.colour_preview.setWordWrap(True)

        # 열수지: 체크하면 계산 중 욕 온도가 변하고 물성이 따라 바뀐다
        self.thermal_box = QGroupBox("욕 온도 변화 계산 (열수지)")
        self.thermal_box.setCheckable(True)
        self.thermal_box.setChecked(False)
        self.cooling = spin(10.0, 0.0, 100000.0, decimals=1, step=5.0, suffix="W/K")
        self.coolant = spin(20.0, -30.0, 100.0, decimals=1, step=0.5, suffix="°C")
        self.thermal_info = hint()
        thermal_form = QFormLayout(self.thermal_box)
        thermal_form.addRow("냉각 능력 UA", self.cooling)
        thermal_form.addRow("냉각수 온도", self.coolant)
        thermal_form.addRow(self.thermal_info)
        thermal_form.addRow(hint("전기 입력 U·I 가 모두 욕을 데우고, 냉각기가 UA·(욕 − 냉각수) 만큼 식힙니다. "
                                 "UA = 0 이면 단열입니다. 온도에 따라 전도도·용해 속도가 바뀝니다."))

        form = QFormLayout(self)
        form.addRow("금속", self.metal)
        form.addRow("전원 모드", self.mode)
        form.addRow("전압", self.voltage)
        form.addRow("전류", self._current_row)
        form.addRow("", self.current_hint)
        form.addRow("램프 (소프트 스타트)", self.ramp)
        form.addRow(self.program)
        form.addRow("공정 시간", self.duration)
        form.addRow("욕 온도", self.temperature)
        form.addRow("해석 단계", self.fidelity)
        self.mode_hint = hint()
        form.addRow(self.mode_hint)
        form.addRow(self.colour_preview)
        form.addRow(self.thermal_box)
        self._form = form
        form.setRowVisible(self.colour_preview, False)

        self.metal.currentIndexChanged.connect(lambda _: self.metal_changed.emit(self.metal_symbol()))
        self.metal.currentIndexChanged.connect(lambda _: self.changed.emit())
        self.mode.currentIndexChanged.connect(self._update_mode)
        self.fidelity.currentIndexChanged.connect(lambda _: self.changed.emit())
        self.unlimited.toggled.connect(self._update_mode)
        for box in (self.voltage, self.current, self.ramp, self.duration, self.temperature):
            box.valueChanged.connect(self._on_value)
        self.program.changed.connect(self._on_program)
        self.thermal_box.toggled.connect(lambda _: self.changed.emit())
        self.cooling.valueChanged.connect(lambda _: self.changed.emit())
        self.coolant.valueChanged.connect(lambda _: self.changed.emit())
        self.coolant.editingFinished.connect(self._mark_coolant_edited)
        self.temperature.valueChanged.connect(self._follow_bath_temperature)
        self._update_mode()

    # ── 값 ──────────────────────────────────────────────────────────────────
    def metal_symbol(self) -> str:
        return self.metal.currentData()

    def power_mode(self):
        """PowerMode 또는 PROGRAM."""
        return self.mode.currentData()

    def is_program(self) -> bool:
        return self.power_mode() == PROGRAM

    def power_setting(self) -> PowerSetting | PowerProgram:
        if self.is_program():
            return self.program.program()
        mode = self.power_mode()
        current = math.inf if (mode is PowerMode.CV and self.unlimited.isChecked()) else self.current.value()
        return PowerSetting(mode, self.voltage.value(), current, self.ramp.value())

    def fidelity_level(self) -> str:
        return self.fidelity.currentData()

    def duration_s(self) -> float:
        if self.is_program():
            return self.program.program().duration
        return self.duration.value() * u.MINUTE

    def temperature_k(self) -> float:
        return u.celsius_to_kelvin(self.temperature.value())

    def thermal_setting(self) -> ThermalSetting | None:
        if not self.thermal_box.isChecked():
            return None
        return ThermalSetting(self.cooling.value(), u.celsius_to_kelvin(self.coolant.value()))

    def set_temperature_c(self, value: float) -> None:
        self.temperature.setValue(value)

    def set_anode_area(self, area_m2: float) -> None:
        self._anode_area = area_m2
        self._update_current_hint()

    def set_thermal_info(self, text: str) -> None:
        self.thermal_info.setText(text)

    def set_colour_preview(self, html: str | None) -> None:
        """Ti 치밀막의 예상 간섭색 (None 이면 숨김)."""
        self._form.setRowVisible(self.colour_preview, html is not None)
        self.colour_preview.setText(html or "")

    # ── 불러오기 (프로젝트 파일) ────────────────────────────────────────────
    def set_power(self, power: PowerSetting | PowerProgram, duration: float) -> None:
        if isinstance(power, PowerProgram):
            self.program.set_program(power)
            self.mode.setCurrentIndex(self.mode.findData(PROGRAM))
        else:
            for box, value in ((self.voltage, power.voltage), (self.ramp, power.ramp_time)):
                box.setValue(value)
            unlimited = math.isinf(power.current)
            self.unlimited.setChecked(unlimited)
            if not unlimited:
                self.current.setValue(power.current)
            self.mode.setCurrentIndex(self.mode.findData(power.mode))
        self.duration.setValue(duration / u.MINUTE)

    def set_thermal(self, thermal: ThermalSetting | None) -> None:
        self.thermal_box.setChecked(thermal is not None)
        if thermal is not None:
            self.cooling.setValue(thermal.cooling_ua)
            self.coolant.setValue(u.kelvin_to_celsius(thermal.coolant_temperature))
            self._coolant_edited = True

    def set_fidelity(self, level: str) -> None:
        self.fidelity.setCurrentIndex(self.fidelity.findData(level))

    # ── 내부 ────────────────────────────────────────────────────────────────
    def _on_value(self, *_):
        self._update_current_hint()
        self.changed.emit()

    def _on_program(self):
        self.duration.blockSignals(True)
        self.duration.setValue(self.program.program().duration / u.MINUTE)
        self.duration.blockSignals(False)
        self.changed.emit()

    def _mark_coolant_edited(self):
        self._coolant_edited = True

    def _follow_bath_temperature(self, value: float):
        """냉각수 온도를 직접 고치기 전까지는 욕 온도(설정값)를 따라간다."""
        if not self._coolant_edited:
            self.coolant.blockSignals(True)
            self.coolant.setValue(value)
            self.coolant.blockSignals(False)

    def _update_mode(self, *_):
        program = self.is_program()
        if program and self.program.table.rowCount() == 0:   # 처음 고르면 지금 단일 설정을 첫 단계로
            single = self._single_setting()
            self.program.set_program(PowerProgram((PowerStep(single.mode, single.voltage, single.current,
                                                             self.duration.value() * u.MINUTE,
                                                             single.ramp_time),)))
        for row in (self.voltage, self._current_row, self.current_hint, self.ramp):
            self._form.setRowVisible(row, not program)
        self._form.setRowVisible(self.program, program)
        self.duration.setEnabled(not program)
        if program:
            self._on_program()
            self.mode_hint.setText("단계를 차례로 실행합니다. 공정 시간은 프로그램 길이로 정해집니다.")
            self.changed.emit()
            return
        mode = self.power_mode()
        self._last_mode = mode
        is_cv = mode is PowerMode.CV
        self.unlimited.setVisible(is_cv)
        self.current.setEnabled(not (is_cv and self.unlimited.isChecked()))
        labels = {
            PowerMode.CCCV: ("전압 설정", "전류 설정",
                             "설정 전류로 시작해 전압이 설정값에 닿으면 정전압으로 바뀝니다 (일반 정류기)."),
            PowerMode.CV: ("전압 설정", "전류 제한", "전압을 램프로 올린 뒤 일정하게 유지합니다."),
            PowerMode.CC: ("전압 제한", "전류 설정", "전류를 램프로 올린 뒤 일정하게 유지합니다. 전압은 안전 제한값입니다."),
        }[mode]
        self._form.labelForField(self.voltage).setText(labels[0])
        self._form.labelForField(self._current_row).setText(labels[1])
        self.mode_hint.setText(labels[2])
        self._update_current_hint()
        self.changed.emit()

    def _single_setting(self) -> PowerSetting:
        mode = self._last_mode
        current = math.inf if (mode is PowerMode.CV and self.unlimited.isChecked()) else self.current.value()
        return PowerSetting(mode, self.voltage.value(), current, self.ramp.value())

    def _update_current_hint(self):
        if self.is_program():
            return
        setting = self.power_setting()
        if not math.isfinite(setting.current):
            self.current_hint.setText("전류 제한 없음 — 초기 돌입 전류는 전해질 저항으로만 제한됩니다.")
        elif self._anode_area > 0:
            j = setting.current / self._anode_area / u.A_PER_DM2
            self.current_hint.setText(f"양극 면적 {self._anode_area / u.CM ** 2:.0f} cm² 기준 ≈ {j:.2f} A/dm²")
        else:
            self.current_hint.setText("")
