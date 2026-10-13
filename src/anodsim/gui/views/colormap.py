"""두께 색칠 공용 색상표와 범례 (측면도·3D·히트맵이 같은 색을 쓴다)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .. import style

# 순차(sequential) 색상표: 지각적으로 균일하고 색각 이상에도 단조로 읽힌다
THICKNESS_CMAP = pg.colormap.get("viridis")
# 발산(diverging) 색상표: 평균 대비 편차 (가운데 = 0)
DEVIATION_CMAP = pg.colormap.get("CET-D1")


def value_range(values) -> tuple[float, float]:
    """색 범위. 값이 거의 같으면 (균일 분포) 평균 ±0.5 % 로 넓혀 중간색으로 보이게 한다."""
    v = np.asarray(values, dtype=float)
    lo, hi = float(np.min(v)), float(np.max(v))
    span = max(abs(hi), abs(lo), 1e-30)
    if hi - lo < 1e-6 * span:
        mid = 0.5 * (lo + hi)
        return mid - 0.005 * span, mid + 0.005 * span
    return lo, hi


def colors(values, vmin: float, vmax: float) -> np.ndarray:
    """(N, 4) RGBA (0–1)."""
    t = np.clip((np.asarray(values, dtype=float) - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.asarray(THICKNESS_CMAP.map(t, mode="float"), dtype=float)


def _gradient_css() -> str:
    stops = []
    for t in np.linspace(0.0, 1.0, 9):
        r, g, b, _ = THICKNESS_CMAP.map(float(t), mode="byte")
        stops.append(f"stop:{t:.3f} rgb({r},{g},{b})")
    return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, {', '.join(stops)})"


class ColorLegend(QFrame):
    """'산화막 두께 [μm]' 가로 색 막대 + 최소·최대 값. 다른 위젯 위에 겹쳐 놓는다."""

    def __init__(self, title: str = "산화막 두께 [μm]", parent: QWidget | None = None):
        super().__init__(parent)
        self.setStyleSheet(f"ColorLegend {{ background: rgba(252, 252, 251, 225); border: 1px solid {style.GRID}; "
                           "border-radius: 4px; }")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.title = QLabel(title)
        self.bar = QFrame()
        self.bar.setFixedHeight(10)
        self.bar.setStyleSheet(f"background: {_gradient_css()}; border: none; border-radius: 2px;")
        self.low = QLabel()
        self.high = QLabel()
        for label in (self.title, self.low, self.high):
            label.setStyleSheet(f"color: {style.INK_SECONDARY}; background: transparent; border: none;")
        ends = QHBoxLayout()
        ends.setContentsMargins(0, 0, 0, 0)
        ends.addWidget(self.low)
        ends.addStretch(1)
        ends.addWidget(self.high)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)
        layout.addWidget(self.title)
        layout.addWidget(self.bar)
        layout.addLayout(ends)
        self.setFixedWidth(190)
        self.hide()

    def set_range(self, vmin: float, vmax: float, label: str | None = None) -> None:
        digits = 4 if vmax - vmin < 0.01 * max(abs(vmax), 1e-30) else 3
        self.low.setText(f"{vmin:.{digits}g}")
        self.high.setText(f"{vmax:.{digits}g}")
        if label is not None:
            self.title.setText(label)
        self.adjustSize()
        self.show()
