"""정류기(전원) 모델.

실제 정류기는 전압 제한과 전류 제한을 동시에 가지며, 둘 중 먼저 걸리는 쪽으로 동작한다.
세 모드는 어떤 값을 목표로 보고 어떤 값을 제한으로 보는지, 소프트 스타트(램프)를 어디에 거는지만 다르다.

    CV   : 전압 = 설정값 (램프 적용), 전류 = 제한값
    CC   : 전류 = 설정값 (램프 적용), 전압 = 컴플라이언스 제한값
    CCCV : 전류 설정값으로 시작해 전압이 설정값에 닿으면 정전압 (전류에 램프 적용)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class PowerMode(StrEnum):
    CV = "CV"
    CC = "CC"
    CCCV = "CCCV"


@dataclass(frozen=True)
class PowerSetting:
    mode: PowerMode
    voltage: float            # V
    current: float            # A (CV 에서 제한 없음 = math.inf)
    ramp_time: float = 0.0    # s

    def validate(self) -> list[str]:
        errors = []
        if not self.voltage > 0:
            errors.append("전압은 0보다 커야 합니다.")
        if not self.current > 0:
            errors.append("전류는 0보다 커야 합니다.")
        if self.ramp_time < 0 or math.isnan(self.ramp_time):
            errors.append("램프 시간은 0 이상이어야 합니다.")
        return errors


@dataclass(frozen=True)
class Setpoint:
    voltage_limit: float      # V
    current_limit: float      # A


class PowerSupply:
    def __init__(self, setting: PowerSetting):
        self.setting = setting

    def ramp_fraction(self, t: float) -> float:
        ramp = self.setting.ramp_time
        if ramp <= 0.0:
            return 1.0
        return min(1.0, max(0.0, t / ramp))

    def setpoint(self, t: float) -> Setpoint:
        s = self.setting
        f = self.ramp_fraction(t)
        if s.mode is PowerMode.CV:
            return Setpoint(voltage_limit=s.voltage * f, current_limit=s.current)
        return Setpoint(voltage_limit=s.voltage, current_limit=s.current * f)
