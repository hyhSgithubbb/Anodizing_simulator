"""음극재 선택과 호환성·면적 안내. 음극의 위치·크기는 '배치' 탭에서 정한다."""

from __future__ import annotations

import math

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QFormLayout, QLabel, QVBoxLayout, QWidget

from ...core import units as u
from .widgets import hint


class CathodePanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, presets, parent=None):
        super().__init__(parent)
        self.presets = presets
        self._electrolyte = None
        self._user_picked = False

        self.material = QComboBox()
        for c in presets.cathodes.values():
            self.material.addItem(c.name, c.id)
        self.note = hint()
        self.tafel = hint()
        self.recommended = hint()
        self.compatibility = QLabel()
        self.compatibility.setWordWrap(True)
        self.area = hint()

        form = QFormLayout()
        form.addRow("재질", self.material)
        form.addRow(self.note)
        form.addRow("수소발생 Tafel", self.tafel)
        form.addRow("추천 음극", self.recommended)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.compatibility)
        layout.addWidget(self.area)
        layout.addStretch(1)

        self.material.currentIndexChanged.connect(self._on_material)
        self.material.activated.connect(self._mark_user_choice)
        self._on_material()

    def material_id(self) -> str:
        return self.material.currentData()

    def set_material(self, material_id: str) -> None:
        """프로젝트 불러오기: 저장된 음극재를 고른다 (사용자가 고른 것으로 취급)."""
        index = self.material.findData(material_id)
        if index < 0:
            raise ValueError(f"알 수 없는 음극재: {material_id}")
        self.material.setCurrentIndex(index)
        self._user_picked = True

    def set_electrolyte(self, preset) -> None:
        """전해질이 바뀌면 추천 음극을 보여주고, 사용자가 고른 적이 없거나 맞지 않으면 추천 1순위로 바꾼다."""
        self._electrolyte = preset
        names = [self.presets.cathode(c).name for c in preset.recommended_cathodes]
        self.recommended.setText(", ".join(names))
        current = self.presets.cathode(self.material_id())
        if not self._user_picked or not current.is_compatible(preset.tags):
            index = self.material.findData(preset.recommended_cathodes[0])
            self.material.setCurrentIndex(index)
        self._update_compatibility()

    def set_areas(self, anode_area: float, cathode_area: float) -> None:
        if anode_area <= 0 or cathode_area <= 0:
            self.area.setText("")
            return
        self.area.setText(f"잠긴 음극 면적 합계 {cathode_area / u.CM ** 2:.0f} cm² "
                          f"(양극의 {cathode_area / anode_area:.1f}배)")

    def _mark_user_choice(self, *_):
        self._user_picked = True

    def _on_material(self, *_):
        c = self.presets.cathode(self.material_id())
        self.note.setText(c.note)
        j0 = c.exchange_current_density / u.A_PER_CM2
        self.tafel.setText(f"j₀ ≈ 10^{math.log10(j0):.1f} A/cm², b = {c.tafel_slope * 1000:.0f} mV/dec")
        self._update_compatibility()
        self.changed.emit()

    def _update_compatibility(self):
        if self._electrolyte is None:
            self.compatibility.setText("")
            return
        c = self.presets.cathode(self.material_id())
        if c.is_compatible(self._electrolyte.tags):
            self.compatibility.setText("")
            self.compatibility.setStyleSheet("")
        else:
            self.compatibility.setText(f"주의: '{c.name}'은(는) 이 전해질에 적합하지 않습니다. {c.note}")
            self.compatibility.setStyleSheet("background: #fff4d6; border: 1px solid #fab219; "
                                             "border-radius: 6px; padding: 6px; color: #0b0b0b;")
