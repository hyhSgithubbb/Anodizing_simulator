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
# 전해질 전위: 두께(viridis)와 같은 화면에 함께 나오므로 다른 색조의 단색 순차 램프 (주황, 밝음 → 어두움).
# OKLCH 에서 색조를 고정하고 명도만 0.96 → 0.40 으로 단조 감소시킨 7단계.
POTENTIAL_CMAP = pg.ColorMap(np.linspace(0.0, 1.0, 7),
                             ["#ffeee8", "#fec4af", "#ff956f", "#ee6b37", "#cd4d10", "#a43a05", "#7b2902"])


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


def _gradient_css(cmap=None, colors=None) -> str:
    """가로 그라디언트 CSS. colors (hex 목록)를 주면 색상표 대신 그 색들을 고르게 놓는다."""
    if colors is None:
        colors = []
        for t in np.linspace(0.0, 1.0, 9):
            r, g, b = cmap.map(float(t), mode="byte")[:3]
            colors.append(f"#{r:02x}{g:02x}{b:02x}")
    positions = np.linspace(0.0, 1.0, len(colors))
    stops = [f"stop:{t:.3f} {c}" for t, c in zip(positions, colors, strict=True)]
    return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, {', '.join(stops)})"


class ColorLegend(QFrame):
    """'산화막 두께 [μm]' 같은 제목 + 가로 색 막대 + 최소·최대 값. 다른 위젯 위에 겹쳐 놓는다."""

    def __init__(self, title: str = "산화막 두께 [μm]", parent: QWidget | None = None, cmap=THICKNESS_CMAP):
        super().__init__(parent)
        self.setStyleSheet(f"ColorLegend {{ background: rgba(252, 252, 251, 225); border: 1px solid {style.GRID}; "
                           "border-radius: 4px; }")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.title = QLabel(title)
        self.bar = QFrame()
        self.bar.setFixedHeight(10)
        self._cmap = cmap
        self._custom = False
        self._set_bar(_gradient_css(cmap))
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

    def _set_bar(self, css: str) -> None:
        self.bar.setStyleSheet(f"background: {css}; border: none; border-radius: 2px;")

    def set_range(self, vmin: float, vmax: float, label: str | None = None, colors=None) -> None:
        """colors: 색상표 대신 쓸 색 목록 (예: 두께별 간섭색). None 이면 원래 색상표."""
        if colors is not None:
            self._set_bar(_gradient_css(colors=list(colors)))
            self._custom = True
        elif self._custom:
            self._set_bar(_gradient_css(self._cmap))
            self._custom = False
        digits = 4 if vmax - vmin < 0.01 * max(abs(vmax), 1e-30) else 3
        self.low.setText(f"{vmin:.{digits}g}")
        self.high.setText(f"{vmax:.{digits}g}")
        if label is not None:
            self.title.setText(label)
        self.adjustSize()
        self.show()
