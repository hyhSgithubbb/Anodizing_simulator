"""전해질 프리셋과 (온도, 농도)에서의 모델 파라미터 계산.

레시피 → 물성을 직접 주는 범용 이론은 없으므로 프리셋(JSON) + 보간 + 아레니우스 보정을 쓴다.
프리셋의 수치는 초기 추정값이며 실측 데이터로 보정하는 것을 전제로 한다 (``calibrated`` 플래그).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .constants import GAS_CONSTANT
from .growth.high_field import HighFieldLaw

Regime = Literal["compact", "porous"]


def arrhenius(activation_energy: float, temperature: float, reference_temperature: float) -> float:
    """exp[−E/R·(1/T − 1/T_ref)] — T_ref 에서 1."""
    return math.exp(-activation_energy / GAS_CONSTANT * (1.0 / temperature - 1.0 / reference_temperature))


@dataclass(frozen=True)
class Range:
    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


@dataclass(frozen=True)
class SteadyStateParams:
    """다공성 정상상태 분극곡선 j_ss = j_ref·exp((U_f − U_ref)/U_s)·Arrhenius."""

    j_ref: float               # A/m²
    U_ref: float               # V
    U_s: float                 # V
    activation_energy: float   # J/mol


@dataclass(frozen=True)
class DissolutionParams:
    surface_rate: float        # v_cd [m/s] — 표면 화학 용해
    pore_wall_rate: float      # k_pw [1/s] — 기공벽 용해 (한계 두께 생성)
    barrier_thinning: float    # v_thin [m/s] — δ_b > δ_ss 일 때 배리어 박화
    concentration_order: float
    activation_energy: float   # J/mol


@dataclass(frozen=True)
class ElectrolytePreset:
    id: str
    name: str
    metal: str
    regime: Regime
    description: str
    tags: tuple[str, ...]
    recommended_cathodes: tuple[str, ...]
    species: str                       # 농도 기준 성분 (예: H2SO4)
    concentration_unit: str            # "wt%", "M" …
    concentration_ref: float
    concentration_range: Range
    temperature_ref: float             # K
    temperature_range: Range           # K
    voltage_range: Range               # V
    typical_voltage: float             # V — GUI 기본값
    conductivity_concentrations: tuple[float, ...]
    conductivity_values: tuple[float, ...]   # S/m @ temperature_ref
    conductivity_temp_coeff: float     # 1/K
    current_efficiency: float
    porosity: float
    field_coefficient: float           # B [m/V]
    anodizing_ratio: float             # k_a [m/V] @ anodizing_ratio_j_ref
    anodizing_ratio_j_ref: float       # A/m²
    steady_state: SteadyStateParams | None
    dissolution: DissolutionParams | None
    breakdown_voltage: float           # V
    burning_current_density: float     # A/m²
    calibrated: bool = False

    def conductivity(self, temperature: float, concentration: float | None = None) -> float:
        c = self.concentration_ref if concentration is None else concentration
        kappa_ref = float(np.interp(c, self.conductivity_concentrations, self.conductivity_values))
        factor = 1.0 + self.conductivity_temp_coeff * (temperature - self.temperature_ref)
        return max(kappa_ref * factor, 1e-6)

    def properties(self, temperature: float, concentration: float | None = None) -> ElectrolyteProps:
        """욕 온도 T [K], 농도에서 성장 모델이 쓰는 파라미터 묶음을 계산한다."""
        c = self.concentration_ref if concentration is None else concentration
        if c <= 0:
            raise ValueError("concentration must be positive")
        law = HighFieldLaw.from_anodizing_ratio(self.field_coefficient, self.anodizing_ratio,
                                                self.anodizing_ratio_j_ref)
        j_ss_ref = U_ref = U_s = 0.0
        if self.steady_state is not None:
            ss = self.steady_state
            j_ss_ref = ss.j_ref * arrhenius(ss.activation_energy, temperature, self.temperature_ref)
            U_ref, U_s = ss.U_ref, ss.U_s
        v_cd = k_pw = v_thin = 0.0
        if self.dissolution is not None:
            d = self.dissolution
            scale = arrhenius(d.activation_energy, temperature, self.temperature_ref)
            scale *= (c / self.concentration_ref) ** d.concentration_order
            v_cd, k_pw, v_thin = d.surface_rate * scale, d.pore_wall_rate * scale, d.barrier_thinning * scale
        return ElectrolyteProps(
            preset_id=self.id,
            regime=self.regime,
            temperature=temperature,
            concentration=c,
            conductivity=self.conductivity(temperature, c),
            current_efficiency=self.current_efficiency,
            porosity=self.porosity,
            high_field=law,
            j_ss_ref=j_ss_ref,
            U_ref=U_ref,
            U_s=U_s,
            surface_dissolution=v_cd,
            pore_wall_dissolution=k_pw,
            barrier_thinning=v_thin,
            breakdown_voltage=self.breakdown_voltage,
            burning_current_density=self.burning_current_density,
        )


@dataclass(frozen=True)
class ElectrolyteProps:
    """특정 온도·농도에서 평가된 전해질 파라미터 (모두 SI)."""

    preset_id: str
    regime: Regime
    temperature: float
    concentration: float
    conductivity: float
    current_efficiency: float
    porosity: float
    high_field: HighFieldLaw
    j_ss_ref: float
    U_ref: float
    U_s: float
    surface_dissolution: float
    pore_wall_dissolution: float
    barrier_thinning: float
    breakdown_voltage: float
    burning_current_density: float

    def with_overrides(self, **changes) -> ElectrolyteProps:
        """보정·테스트용 — 일부 파라미터만 바꾼 사본."""
        return replace(self, **changes)
