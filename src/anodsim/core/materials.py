"""금속과 산화물 물성. 순수 금속(Al 1xxx 계열, CP Ti)만 다룬다."""

from __future__ import annotations

from dataclasses import dataclass

from .constants import FARADAY


@dataclass(frozen=True)
class Oxide:
    formula: str
    molar_mass: float          # kg/mol
    density: float             # kg/m³ (양극산화 비정질막)
    electrons: int             # 화학식단위 1개 생성에 필요한 전자 수 (Al2O3: 6, TiO2: 4)
    metal_atoms: int           # 화학식단위당 금속 원자 수 (Al2O3: 2, TiO2: 1)
    refractive_index: float

    @property
    def growth_coefficient(self) -> float:
        """패러데이 성장 계수 k_F = M / (z F ρ) [m³/C] — 전하 1 C당 생기는 치밀 산화물 부피."""
        return self.molar_mass / (self.electrons * FARADAY * self.density)

    @property
    def molar_volume(self) -> float:
        return self.molar_mass / self.density


@dataclass(frozen=True)
class Metal:
    symbol: str
    name: str
    molar_mass: float          # kg/mol
    density: float             # kg/m³
    valence: int               # 금속 원자 1개가 산화될 때 내놓는 전자 수
    native_oxide_thickness: float  # m
    oxide: Oxide

    @property
    def recession_coefficient(self) -> float:
        """금속 소모 계수 k_M = M / (z F ρ) [m³/C] — 산화막 형성 전하 1 C당 소모되는 금속 부피."""
        return self.molar_mass / (self.valence * FARADAY * self.density)

    @property
    def expansion_ratio(self) -> float:
        """산화물 부피 / 소모된 금속 부피 (Pilling–Bedworth 비)."""
        metal_volume = self.oxide.metal_atoms * self.molar_mass / self.density
        return self.oxide.molar_volume / metal_volume
