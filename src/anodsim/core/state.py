"""공정 설정(불변)과 시뮬레이션 상태·스냅샷."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .geometry import Geometry
from .power_supply import PowerSetting

SUPPORTED_FIDELITY = ("L0", "L1", "L2")
MAX_ELEMENTS = 400


@dataclass(frozen=True)
class ElectrolyteRecipe:
    preset_id: str
    concentration: float | None = None   # 프리셋 단위 (wt%, M …). None → 프리셋 기준 농도


@dataclass(frozen=True)
class ProcessConfig:
    metal: str                       # "Al" | "Ti"
    electrolyte: ElectrolyteRecipe
    cathode_material: str            # 음극재 프리셋 id
    power: PowerSetting
    geometry: Geometry
    bath_temperature: float          # K
    duration: float                  # s
    fidelity: str = "L0"             # "L0" (0D) | "L1" (위치별, 경로 저항 근사) | "L2" (위치별, FDM 전위장)
    elements: int = 120              # L1·L2 표면 요소 수 (목표값)
    offset_voltage: float = 0.0      # V — 가역 셀 전압 + 접점 전압 (묶음)
    external_resistance: float = 0.0  # Ω — 버스바·케이블

    def validate(self, presets) -> list[str]:
        errors: list[str] = []
        if not self.duration > 0:
            errors.append("공정 시간은 0보다 커야 합니다.")
        if not 200.0 < self.bath_temperature < 400.0 or math.isnan(self.bath_temperature):
            errors.append("욕 온도가 물리적으로 가능한 범위를 벗어났습니다.")
        if self.fidelity not in SUPPORTED_FIDELITY:
            errors.append(f"지원하지 않는 해석 단계: {self.fidelity} (현재 {', '.join(SUPPORTED_FIDELITY)} 만 지원)")
        if self.fidelity != "L0" and not 2 <= self.elements <= MAX_ELEMENTS:
            errors.append(f"표면 요소 수는 2–{MAX_ELEMENTS} 사이여야 합니다.")
        if self.external_resistance < 0:
            errors.append("외부 저항은 0 이상이어야 합니다.")
        errors += self.power.validate()
        errors += self.geometry.validate()
        if self.metal not in presets.metals:
            errors.append(f"알 수 없는 금속: {self.metal}")
        preset = presets.electrolytes.get(self.electrolyte.preset_id)
        if preset is None:
            errors.append(f"알 수 없는 전해질 프리셋: {self.electrolyte.preset_id}")
        elif preset.metal != self.metal:
            errors.append(f"전해질 '{preset.name}'은(는) {preset.metal}용입니다 (선택한 금속: {self.metal}).")
        if self.electrolyte.concentration is not None and not self.electrolyte.concentration > 0:
            errors.append("전해질 농도는 0보다 커야 합니다.")
        if self.cathode_material not in presets.cathodes:
            errors.append(f"알 수 없는 음극재: {self.cathode_material}")
        return errors


@dataclass
class SimState:
    """표면 요소별 상태 (길이 N 배열, L0 에서는 요소 1개)."""

    time: float
    barrier: np.ndarray       # δ_b [m]
    porous: np.ndarray        # h_p [m]
    metal_loss: np.ndarray    # 금속 후퇴 [m]
    charge: np.ndarray        # 누적 전하 [C/m²]

    @property
    def total(self) -> np.ndarray:
        return self.barrier + self.porous


@dataclass(frozen=True)
class Snapshot:
    """GUI·결과 기록용 불변 사본."""

    time: float
    voltage: float
    current: float
    control: str
    cathode_overpotential: float
    current_density: np.ndarray
    film_voltage: np.ndarray
    barrier: np.ndarray
    porous: np.ndarray
    total: np.ndarray
    metal_loss: np.ndarray
    charge: np.ndarray
    flags: frozenset[str] = field(default_factory=frozenset)
