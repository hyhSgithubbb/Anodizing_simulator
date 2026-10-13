"""전해질 레시피: 프리셋 선택, 농도, 그리고 현재 온도에서 계산된 모델 파라미터 표시."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QFormLayout, QGroupBox, QLabel, QVBoxLayout, QWidget

from ...core import units as u
from ...core.state import ElectrolyteRecipe
from .widgets import hint, spin

REGIME_LABELS = {"porous": "다공성 (배리어층 + 기공층)", "compact": "치밀 (배리어형)"}


class ElectrolytePanel(QWidget):
    changed = pyqtSignal()
    preset_changed = pyqtSignal(str)

    def __init__(self, presets, parent=None):
        super().__init__(parent)
        self.presets = presets
        self._temperature = u.celsius_to_kelvin(20.0)

        self.preset = QComboBox()
        self.description = hint()
        self.concentration = spin(15.0, 0.001, 100.0, decimals=3, step=0.5)
        self.concentration_hint = hint()

        form = QFormLayout()
        form.addRow("프리셋", self.preset)
        form.addRow(self.description)
        form.addRow("농도", self.concentration)
        form.addRow("", self.concentration_hint)

        self.props_box = QGroupBox("모델 파라미터 (현재 온도·농도 기준)")
        self.props_form = QFormLayout(self.props_box)
        self._values: dict[str, QLabel] = {}
        for key, label in (("regime", "성장 모드"), ("kappa", "전도도 κ"), ("eta", "전류효율 η"),
                           ("porosity", "기공률 P"), ("ratio", "양극산화비 k_a"), ("jss", "정상 전류밀도"),
                           ("dissolution", "표면 용해 속도"), ("breakdown", "절연파괴 전압"), ("calibrated", "보정 상태")):
            value = QLabel("–")
            self._values[key] = value
            self.props_form.addRow(label, value)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.props_box)
        layout.addStretch(1)

        self.preset.currentIndexChanged.connect(self._on_preset)
        self.concentration.valueChanged.connect(self._on_value)

    # ── 값 ──────────────────────────────────────────────────────────────────
    def set_metal(self, metal: str) -> None:
        self.preset.blockSignals(True)
        self.preset.clear()
        for e in self.presets.electrolytes_for(metal):
            self.preset.addItem(e.name, e.id)
        self.preset.blockSignals(False)
        self._on_preset()

    def set_temperature(self, temperature_k: float) -> None:
        self._temperature = temperature_k
        self._refresh_properties()

    def current_preset(self):
        preset_id = self.preset.currentData()
        return None if preset_id is None else self.presets.electrolyte(preset_id)

    def recipe(self) -> ElectrolyteRecipe:
        return ElectrolyteRecipe(self.preset.currentData(), self.concentration.value())

    # ── 내부 ────────────────────────────────────────────────────────────────
    def _on_preset(self, *_):
        e = self.current_preset()
        if e is None:
            return
        self.description.setText(e.description)
        self.concentration.blockSignals(True)
        self.concentration.setSuffix(f" {e.concentration_unit} {e.species}")
        self.concentration.setValue(e.concentration_ref)
        self.concentration.blockSignals(False)
        r = e.concentration_range
        self.concentration_hint.setText(f"프리셋 검증 범위 {r.low:g}–{r.high:g} {e.concentration_unit}")
        self._refresh_properties()
        self.preset_changed.emit(e.id)
        self.changed.emit()

    def _on_value(self, *_):
        self._refresh_properties()
        self.changed.emit()

    def _refresh_properties(self):
        e = self.current_preset()
        if e is None:
            return
        p = e.properties(self._temperature, self.concentration.value())
        v = self._values
        v["regime"].setText(REGIME_LABELS[e.regime])
        v["kappa"].setText(f"{p.conductivity:.3g} S/m")
        v["eta"].setText(f"{p.current_efficiency:.2f}")
        v["porosity"].setText(f"{p.porosity:.2f}" if e.regime == "porous" else "— (치밀막)")
        v["ratio"].setText(f"{e.anodizing_ratio / u.NM:.2f} nm/V @ {e.anodizing_ratio_j_ref / u.A_PER_DM2:g} A/dm²")
        if e.regime == "porous":
            v["jss"].setText(f"{p.j_ss_ref / u.A_PER_DM2:.2f} A/dm² @ {p.U_ref:g} V (막 전압)")
            v["dissolution"].setText(f"{p.surface_dissolution / u.UM_PER_HOUR:.3g} μm/h")
        else:
            v["jss"].setText("— (치밀막)")
            v["dissolution"].setText("— (치밀막)")
        v["breakdown"].setText(f"{p.breakdown_voltage:g} V")
        v["calibrated"].setText("보정됨" if e.calibrated else "미보정 — 초기 추정값")
