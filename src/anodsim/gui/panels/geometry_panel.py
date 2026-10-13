"""전해조·전극 배치 (측면도 좌표: x 가로, z 높이, y 깊이)."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFormLayout, QGroupBox, QPushButton, QVBoxLayout, QWidget

from ...core import units as u
from ...core.geometry import Geometry, PlateElectrode, Tank
from .widgets import hint, spin


class PlateEditor(QGroupBox):
    changed = pyqtSignal()

    def __init__(self, title: str, plate: PlateElectrode, checkable: bool = False, parent=None):
        super().__init__(title, parent)
        self.setCheckable(checkable)
        self.x = spin(0.0, -1000.0, 1000.0, decimals=1, step=0.5, suffix="cm")
        self.z = spin(0.0, -1000.0, 1000.0, decimals=1, step=0.5, suffix="cm")
        self.height_box = spin(1.0, 0.1, 1000.0, decimals=1, step=0.5, suffix="cm")
        self.width_box = spin(1.0, 0.1, 1000.0, decimals=1, step=0.5, suffix="cm")
        self.thickness = spin(1.0, 0.1, 100.0, decimals=1, step=0.5, suffix="mm")
        form = QFormLayout(self)
        form.addRow("위치 x (중심)", self.x)
        form.addRow("바닥 높이 z", self.z)
        form.addRow("높이", self.height_box)
        form.addRow("폭 (깊이 방향 y)", self.width_box)
        form.addRow("두께", self.thickness)
        self.set_plate(plate)
        for box in (self.x, self.z, self.height_box, self.width_box, self.thickness):
            box.valueChanged.connect(lambda *_: self.changed.emit())
        self.toggled.connect(lambda *_: self.changed.emit())

    def plate(self) -> PlateElectrode:
        return PlateElectrode(
            x_center=self.x.value() * u.CM,
            z_bottom=self.z.value() * u.CM,
            height=self.height_box.value() * u.CM,
            thickness=self.thickness.value() * u.MM,
            width=self.width_box.value() * u.CM,
        )

    def set_plate(self, plate: PlateElectrode) -> None:
        boxes = ((self.x, plate.x_center / u.CM), (self.z, plate.z_bottom / u.CM),
                 (self.height_box, plate.height / u.CM), (self.width_box, plate.width / u.CM),
                 (self.thickness, plate.thickness / u.MM))
        for box, value in boxes:
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)


class GeometryPanel(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        default = Geometry.default()

        tank_box = QGroupBox("전해조")
        self.tank_w = spin(0.0, 1.0, 1000.0, decimals=1, step=1.0, suffix="cm")
        self.tank_d = spin(0.0, 1.0, 1000.0, decimals=1, step=1.0, suffix="cm")
        self.tank_h = spin(0.0, 1.0, 1000.0, decimals=1, step=1.0, suffix="cm")
        self.level = spin(0.0, 0.1, 1000.0, decimals=1, step=0.5, suffix="cm")
        form = QFormLayout(tank_box)
        form.addRow("가로 x", self.tank_w)
        form.addRow("깊이 y", self.tank_d)
        form.addRow("높이 z", self.tank_h)
        form.addRow("액면 높이", self.level)

        self.anode = PlateEditor("양극 (+) — 피처리물", default.anode)
        self.cathode1 = PlateEditor("음극 1", default.cathodes[0], checkable=True)
        self.cathode2 = PlateEditor("음극 2", default.cathodes[1], checkable=True)
        self.reset_button = QPushButton("기본 배치로 되돌리기")

        layout = QVBoxLayout(self)
        layout.addWidget(hint("측면도 기준 좌표입니다. x 는 가로, z 는 높이(바닥 0), y 는 화면 안쪽 깊이이며 "
                              "전극은 깊이 방향 가운데에 놓입니다."))
        layout.addWidget(tank_box)
        layout.addWidget(self.anode)
        layout.addWidget(self.cathode1)
        layout.addWidget(self.cathode2)
        layout.addWidget(self.reset_button)
        layout.addStretch(1)

        self._set_tank(default.tank)
        for box in (self.tank_w, self.tank_d, self.tank_h, self.level):
            box.valueChanged.connect(lambda *_: self.changed.emit())
        for editor in (self.anode, self.cathode1, self.cathode2):
            editor.changed.connect(self.changed.emit)
        self.reset_button.clicked.connect(self.reset)

    def geometry(self) -> Geometry:
        tank = Tank(width=self.tank_w.value() * u.CM, depth=self.tank_d.value() * u.CM,
                    height=self.tank_h.value() * u.CM, liquid_level=self.level.value() * u.CM)
        cathodes = tuple(e.plate() for e in (self.cathode1, self.cathode2) if e.isChecked())
        return Geometry(tank, self.anode.plate(), cathodes)

    def reset(self) -> None:
        default = Geometry.default()
        self._set_tank(default.tank)
        self.anode.set_plate(default.anode)
        for editor, plate in zip((self.cathode1, self.cathode2), default.cathodes, strict=True):
            editor.blockSignals(True)
            editor.setChecked(True)
            editor.blockSignals(False)
            editor.set_plate(plate)
        self.changed.emit()

    def _set_tank(self, tank: Tank) -> None:
        for box, value in ((self.tank_w, tank.width), (self.tank_d, tank.depth),
                           (self.tank_h, tank.height), (self.level, tank.liquid_level)):
            box.blockSignals(True)
            box.setValue(value / u.CM)
            box.blockSignals(False)
