"""배리어층 고전계 이온전도 법칙.

    j = 2A·sinh(B·U_f/δ)   ⇔   U_f = (δ/B)·asinh(j / 2A)

j ≫ A 영역에서는 고전적인 j = A·exp(B·E) 와 같고, j → 0 에서 U_f → 0 으로 매끄럽게 이어져
수치 해석(Newton)이 안정적이다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_MAX_EXPONENT = 700.0  # exp/sinh overflow 방지


@dataclass(frozen=True)
class HighFieldLaw:
    A: float   # A/m²
    B: float   # m/V

    @classmethod
    def from_anodizing_ratio(cls, B: float, ratio: float, j_ref: float) -> HighFieldLaw:
        """측정하기 쉬운 양극산화비 k_a [m/V] @ j_ref 로부터 A 를 유도한다: A = j_ref·exp(−B/k_a)."""
        if B <= 0 or ratio <= 0 or j_ref <= 0:
            raise ValueError("B, anodizing ratio, j_ref must be positive")
        return cls(A=j_ref * math.exp(-B / ratio), B=B)

    def film_voltage(self, j, delta):
        """막 전압 U_f [V]."""
        return np.asarray(delta) / self.B * np.arcsinh(np.asarray(j) / (2.0 * self.A))

    def d_film_voltage_d_lnj(self, j, delta):
        """∂U_f/∂(ln j) — 회로 해석기의 야코비안 대각 성분."""
        j = np.asarray(j)
        return np.asarray(delta) / self.B * j / np.sqrt(j * j + 4.0 * self.A * self.A)

    def current_density(self, film_voltage, delta):
        """막 전압과 두께로부터 이온 전류밀도 j [A/m²]."""
        arg = np.clip(self.B * np.asarray(film_voltage) / np.asarray(delta), -_MAX_EXPONENT, _MAX_EXPONENT)
        return 2.0 * self.A * np.sinh(arg)

    def anodizing_ratio(self, j):
        """전류밀도 j 에서의 양극산화비 δ/U_f [m/V]."""
        return self.B / np.arcsinh(np.asarray(j) / (2.0 * self.A))
