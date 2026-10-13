"""욕 열수지 (lumped) — 설계서 §1.9.

    C · dT/dt = P − UA · (T − T_cool),        P = (U_app − R_ext·I) · I,        C = ρ · c_p · V_욕

- 욕에 들어간 전기 입력 P 를 모두 열로 본다 (외부 저항 R_ext = 버스바·케이블의 줄열은 욕 밖이라 뺀다).
  산화물 생성 반응의 엔탈피(Al 은 발열, 수 % 수준)는 무시한다.
- 냉각기·자연 방열을 UA [W/K] 하나로 묶는다 (UA = 0 이면 단열).
- 욕 부피는 액면 아래 전해조 부피에서 잠긴 전극 부피를 뺀 값이다.
- 욕은 잘 섞인다고 본다 (온도 분포 없음). 기공 바닥의 국부 발열·burning 은 다루지 않는다.

온도가 바뀌면 시뮬레이터가 전해질 물성(κ, j_ss, 용해 속도 — 모두 온도 의존)을 다시 계산한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Geometry


@dataclass(frozen=True)
class ThermalSetting:
    cooling_ua: float             # W/K — 냉각 능력 (0 = 단열)
    coolant_temperature: float    # K — 냉각수(냉각기 설정) 온도

    def validate(self) -> list[str]:
        errors = []
        if not self.cooling_ua >= 0 or math.isinf(self.cooling_ua):
            errors.append("냉각 능력(UA)은 0 이상이어야 합니다.")
        if not 200.0 < self.coolant_temperature < 400.0:
            errors.append("냉각수 온도가 물리적으로 가능한 범위를 벗어났습니다.")
        return errors


@dataclass(frozen=True)
class BathThermal:
    heat_capacity: float          # C [J/K]
    setting: ThermalSetting

    @classmethod
    def from_config(cls, geometry: Geometry, preset, setting: ThermalSetting) -> BathThermal:
        volume = geometry.liquid_volume()
        return cls(heat_capacity=preset.density * preset.heat_capacity * volume, setting=setting)

    def rate(self, temperature: float, power: float) -> float:
        """dT/dt [K/s]."""
        s = self.setting
        return (power - s.cooling_ua * (temperature - s.coolant_temperature)) / self.heat_capacity

    def advance(self, temperature: float, power: float, dt: float) -> float:
        """dt 뒤 욕 온도. 스텝 동안 전력을 일정하게 보고 선형 ODE 를 정확히 푼다.

        냉각 시정수 C/UA 가 스텝보다 짧아도 진동하지 않고 (명시적 오일러는 dt > 2C/UA 에서 불안정),
        단열(UA = 0)이면 C·ΔT = P·Δt 그대로다.
        """
        s = self.setting
        if s.cooling_ua <= 0.0:
            return temperature + dt * power / self.heat_capacity
        equilibrium = s.coolant_temperature + power / s.cooling_ua
        return equilibrium + (temperature - equilibrium) * math.exp(-dt / self.time_constant)

    @property
    def time_constant(self) -> float:
        """냉각 시정수 C/UA [s] (단열이면 inf)."""
        ua = self.setting.cooling_ua
        return self.heat_capacity / ua if ua > 0 else math.inf
