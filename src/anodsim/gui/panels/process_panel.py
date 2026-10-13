"""공정 조건: 금속, 전원 모드, 전압·전류, 램프, 공정 시간, 욕 온도."""

from __future__ import annotations

import math

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QWidget

from ...core import units as u
from ...core.power_supply import PowerMode, PowerSetting
from .widgets import hint, spin

MODE_LABELS = (
    (PowerMode.CCCV, "CC→CV (정전류 후 정전압)"),
    (PowerMode.CV, "CV (정전압)"),
    (PowerMode.CC, "CC (정전류)"),
)


class ProcessPanel(QWidget):
    changed = pyqtSignal()
    metal_changed = pyqtSignal(str)

    def __init__(self, presets, parent=None):
        super().__init__(parent)
        self._anode_area = 0.0

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

        self.ramp = spin(30.0, 0.0, 3600.0, decimals=0, step=5.0, suffix="s")
        self.duration = spin(60.0, 0.1, 1440.0, decimals=1, step=5.0, suffix="min")
        self.temperature = spin(20.0, -20.0, 100.0, decimals=1, step=0.5, suffix="°C")

        form = QFormLayout(self)
        form.addRow("금속", self.metal)
        form.addRow("전원 모드", self.mode)
        form.addRow("전압", self.voltage)
        form.addRow("전류", self._current_row)
        form.addRow("", self.current_hint)
        form.addRow("램프 (소프트 스타트)", self.ramp)
        form.addRow("공정 시간", self.duration)
        form.addRow("욕 온도", self.temperature)
        self.mode_hint = hint()
        form.addRow(self.mode_hint)
        self._form = form

        self.metal.currentIndexChanged.connect(lambda _: self.metal_changed.emit(self.metal_symbol()))
        self.metal.currentIndexChanged.connect(lambda _: self.changed.emit())
        self.mode.currentIndexChanged.connect(self._update_mode)
        self.unlimited.toggled.connect(self._update_mode)
        for box in (self.voltage, self.current, self.ramp, self.duration, self.temperature):
            box.valueChanged.connect(self._on_value)
        self._update_mode()

    # ── 값 ──────────────────────────────────────────────────────────────────
    def metal_symbol(self) -> str:
        return self.metal.currentData()

    def power_mode(self) -> PowerMode:
        return self.mode.currentData()

    def power_setting(self) -> PowerSetting:
        mode = self.power_mode()
        current = math.inf if (mode is PowerMode.CV and self.unlimited.isChecked()) else self.current.value()
        return PowerSetting(mode, self.voltage.value(), current, self.ramp.value())

    def duration_s(self) -> float:
        return self.duration.value() * u.MINUTE

    def temperature_k(self) -> float:
        return u.celsius_to_kelvin(self.temperature.value())

    def set_temperature_c(self, value: float) -> None:
        self.temperature.setValue(value)

    def set_anode_area(self, area_m2: float) -> None:
        self._anode_area = area_m2
        self._update_current_hint()

    # ── 내부 ────────────────────────────────────────────────────────────────
    def _on_value(self, *_):
        self._update_current_hint()
        self.changed.emit()

    def _update_mode(self, *_):
        mode = self.power_mode()
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

    def _update_current_hint(self):
        setting = self.power_setting()
        if not math.isfinite(setting.current):
            self.current_hint.setText("전류 제한 없음 — 초기 돌입 전류는 전해질 저항으로만 제한됩니다.")
        elif self._anode_area > 0:
            j = setting.current / self._anode_area / u.A_PER_DM2
            self.current_hint.setText(f"양극 면적 {self._anode_area / u.CM ** 2:.0f} cm² 기준 ≈ {j:.2f} A/dm²")
        else:
            self.current_hint.setText("")
