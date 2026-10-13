"""음극재: 수소발생 반응(HER) 과전압.

|η_c| = (b/ln10)·asinh(I / (2·j0·A_c))

전류가 큰 영역에서 Tafel 식 b·log10(j_c/j0) 와 같고, 전류 0 에서 0 으로 매끄럽게 이어진다.
음극 표면의 전류 분포는 무시하고 총전류/총면적으로 균일하게 묶는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import LN10


@dataclass(frozen=True)
class CathodeMaterial:
    id: str
    name: str
    exchange_current_density: float   # j0 [A/m²]
    tafel_slope: float                # b [V/decade]
    incompatible_tags: tuple[str, ...] = ()
    note: str = ""

    def overpotential(self, current: float, area: float) -> float:
        """음극 과전압 크기 [V] (항상 ≥ 0)."""
        if current <= 0.0:
            return 0.0
        return self.tafel_slope / LN10 * math.asinh(current / (2.0 * self.exchange_current_density * area))

    def d_overpotential(self, current: float, area: float) -> float:
        """d|η_c|/dI [Ω]."""
        c = 2.0 * self.exchange_current_density * area
        return self.tafel_slope / LN10 / math.sqrt(c * c + max(current, 0.0) ** 2)

    def is_compatible(self, electrolyte_tags) -> bool:
        return not set(self.incompatible_tags) & set(electrolyte_tags)
