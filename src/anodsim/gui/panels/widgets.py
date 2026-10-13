"""입력 위젯 공용 도우미."""

from PyQt6.QtWidgets import QDoubleSpinBox, QLabel


def spin(value: float, minimum: float, maximum: float, *, decimals: int = 1, step: float = 1.0,
         suffix: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    if suffix:
        box.setSuffix(f" {suffix}")
    box.setValue(value)
    box.setKeyboardTracking(False)   # 입력 확정(Enter/포커스 이동) 시에만 valueChanged
    box.setAccelerated(True)
    return box


def hint(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: #52514e;")
    return label
