"""성장 모델 공통부: 패러데이 법칙 + 고전계 막 전압 + 전하 분배.

전하 분배 규칙 (모든 모드 공통):
    산화물 형성 속도 F = η·k_F·j  (치밀 산화물 부피 / 면적 / 시간)
    배리어 성장 f_b ≥ 0 이면 그만큼이 배리어로, 나머지 F − f_b 가 기공층으로 간다.
    배리어가 얇아지는 중(f_b < 0, 화학 용해)이면 F 전부가 기공층으로 간다.
이 규칙 덕분에 시간 적분기는 어떤 j 를 쓰든 전하 보존(η·k_F·ΔQ = 생성 치밀 부피)을 정확히 지킨다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

MIN_BARRIER_THICKNESS = 0.5e-9  # m — 수치적 하한


@dataclass(frozen=True)
class GrowthRates:
    """표면 요소별 두께 변화율 [m/s]."""

    barrier: np.ndarray      # dδ_b/dt
    porous: np.ndarray       # dh_p/dt
    metal: np.ndarray        # 금속 후퇴 속도


class GrowthModel(ABC):
    regime: ClassVar[str]

    def __init__(self, metal, props):
        self.metal = metal
        self.props = props
        self.law = props.high_field
        self.k_F = metal.oxide.growth_coefficient      # m³/C
        self.k_M = metal.recession_coefficient         # m³/C
        self.eta = props.current_efficiency
        self.min_barrier = MIN_BARRIER_THICKNESS

    def film_voltage(self, j, barrier):
        return self.law.film_voltage(j, barrier)

    def d_film_voltage_d_lnj(self, j, barrier):
        return self.law.d_film_voltage_d_lnj(j, barrier)

    def formation_rate(self, j):
        """치밀 산화물 생성 속도 η·k_F·j [m/s] (패러데이 법칙)."""
        return self.eta * self.k_F * np.asarray(j, float)

    def metal_rate(self, j):
        return self.eta * self.k_M * np.asarray(j, float)

    @staticmethod
    def pore_formation(formation, barrier_rate):
        """형성 속도 중 기공층으로 가는 몫 (치밀 부피 기준)."""
        return np.maximum(formation - np.maximum(barrier_rate, 0.0), 0.0)

    def rates(self, j, film_voltage, barrier, porous) -> GrowthRates:
        f_b = self.barrier_rate(j, film_voltage, barrier)
        pores = self.pore_formation(self.formation_rate(j), f_b)
        return GrowthRates(barrier=f_b, porous=self.porous_rate(pores, porous), metal=self.metal_rate(j))

    @abstractmethod
    def barrier_rate(self, j, film_voltage, barrier):
        """배리어(치밀)층 두께 변화율 f_b [m/s]."""

    @abstractmethod
    def barrier_rate_partials(self, j, film_voltage, barrier):
        """f_b 의 편미분 (∂f_b/∂j, ∂f_b/∂U_f) — 요소별 배열.

        시간 적분기가 회로 민감도와 묶어 야코비안 ∂f_b/∂δ 를 만든다 (고전계 법칙 때문에 stiff).
        """

    @abstractmethod
    def porous_rate(self, pore_formation, porous):
        """기공층으로 가는 형성 속도(치밀 부피 기준)와 현재 두께로부터 dh_p/dt [m/s]."""
