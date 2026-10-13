"""전원 프로그램 편집기: 단계 표(모드·전압·전류·시간·램프) + 반복 횟수.

다단 승압은 단계를 차례로, 펄스 아노다이징은 '높음/낮음(또는 0 V)' 두 단계를 만들고 반복 횟수를 늘린다.
"""

from __future__ import annotations

import math

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QHBoxLayout, QHeaderView, QLabel,
                             QPushButton, QSpinBox, QTableWidget, QVBoxLayout, QWidget)

from ...core import units as u
from ...core.power_supply import PowerMode, PowerProgram, PowerStep
from .widgets import hint

COLUMNS = ("모드", "전압 [V]", "전류 [A]", "시간 [min]", "램프 [s]")
MODE_SHORT = ((PowerMode.CV, "CV"), (PowerMode.CC, "CC"), (PowerMode.CCCV, "CC→CV"))
NO_LIMIT = -1.0   # 전류 칸의 '제한 없음' (최솟값에 특수 문구)


def _cell_spin(value, minimum, maximum, decimals, step) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setKeyboardTracking(False)
    box.setFrame(False)
    return box


class ProgramEditor(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setMinimumHeight(150)

        self.add_button = QPushButton("단계 추가")
        self.remove_button = QPushButton("단계 삭제")
        self.up_button = QPushButton("▲")
        self.down_button = QPushButton("▼")
        for b in (self.up_button, self.down_button):
            b.setFixedWidth(32)
        self.repeat = QSpinBox()
        self.repeat.setRange(1, 100000)
        self.repeat.setPrefix("반복 ")
        self.repeat.setSuffix(" 회")
        self.total = QLabel()

        buttons = QHBoxLayout()
        for w in (self.add_button, self.remove_button, self.up_button, self.down_button):
            buttons.addWidget(w)
        buttons.addStretch(1)
        buttons.addWidget(self.repeat)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        layout.addLayout(buttons)
        layout.addWidget(self.total)
        layout.addWidget(hint("전류 칸을 맨 아래로 내리면 '제한 없음'입니다. 램프는 직전 단계 값에서 이 단계 값까지 "
                              "선형으로 바꾸는 시간입니다. 펄스는 높음·낮음(0 V 면 꺼짐) 두 단계를 만들고 반복합니다."))

        self.add_button.clicked.connect(self._add_after_selection)
        self.remove_button.clicked.connect(self._remove_selected)
        self.up_button.clicked.connect(lambda: self._move_selected(-1))
        self.down_button.clicked.connect(lambda: self._move_selected(1))
        self.repeat.valueChanged.connect(self._on_change)

    # ── 값 ──────────────────────────────────────────────────────────────────
    def program(self) -> PowerProgram:
        steps = []
        for row in range(self.table.rowCount()):
            mode = self.table.cellWidget(row, 0).currentData()
            voltage, current, minutes, ramp = (self.table.cellWidget(row, c).value() for c in range(1, 5))
            steps.append(PowerStep(mode, voltage, math.inf if current <= NO_LIMIT else current,
                                   minutes * u.MINUTE, ramp))
        return PowerProgram(tuple(steps), self.repeat.value())

    def set_program(self, program: PowerProgram) -> None:
        self.blockSignals(True)
        self.table.setRowCount(0)
        for step in program.steps:
            self._insert_row(self.table.rowCount(), step)
        self.repeat.setValue(program.repeat)
        self.blockSignals(False)
        self._on_change()

    # ── 내부 ────────────────────────────────────────────────────────────────
    def _insert_row(self, row: int, step: PowerStep) -> None:
        self.table.insertRow(row)
        mode = QComboBox()
        mode.setFrame(False)
        for value, label in MODE_SHORT:
            mode.addItem(label, value)
        mode.setCurrentIndex(mode.findData(step.mode))
        current = NO_LIMIT if math.isinf(step.current) else step.current
        cells = (mode,
                 _cell_spin(step.voltage, 0.0, 1000.0, 1, 0.5),
                 _cell_spin(current, NO_LIMIT, 1000.0, 3, 0.1),
                 _cell_spin(step.duration / u.MINUTE, 0.01, 1440.0, 2, 1.0),
                 _cell_spin(step.ramp_time, 0.0, 3600.0, 0, 5.0))
        cells[2].setSpecialValueText("제한 없음")
        for column, widget in enumerate(cells):
            self.table.setCellWidget(row, column, widget)
        mode.currentIndexChanged.connect(self._on_change)
        for widget in cells[1:]:
            widget.valueChanged.connect(self._on_change)

    def _row_step(self, row: int) -> PowerStep:
        return self.program().steps[row]

    def _add_after_selection(self) -> None:
        rows = self.table.rowCount()
        row = self.table.currentRow()
        row = rows - 1 if row < 0 else row
        template = self._row_step(row) if rows else PowerStep(PowerMode.CV, 15.0, math.inf, 600.0, 30.0)
        self._insert_row(row + 1, template)
        self.table.selectRow(row + 1)
        self._on_change()

    def _remove_selected(self) -> None:
        if self.table.rowCount() <= 1:
            return
        row = self.table.currentRow()
        self.table.removeRow(row if row >= 0 else self.table.rowCount() - 1)
        self._on_change()

    def _move_selected(self, offset: int) -> None:
        row = self.table.currentRow()
        target = row + offset
        if row < 0 or not 0 <= target < self.table.rowCount():
            return
        program = self.program()
        steps = list(program.steps)
        steps[row], steps[target] = steps[target], steps[row]
        self.set_program(PowerProgram(tuple(steps), program.repeat))
        self.table.selectRow(target)

    def _on_change(self, *_):
        program = self.program()
        self.remove_button.setEnabled(self.table.rowCount() > 1)
        cycle = program.cycle / u.MINUTE
        text = f"한 묶음 {cycle:g} min"
        if program.repeat > 1:
            text += f" × {program.repeat} 회"
        self.total.setText(f"{text} = 총 {program.duration / u.MINUTE:g} min")
        if not self.signalsBlocked():
            self.changed.emit()
