"""정류기(전원) 모델.

실제 정류기는 전압 제한과 전류 제한을 동시에 가지며, 둘 중 먼저 걸리는 쪽으로 동작한다.
세 모드는 어떤 값을 목표로 보고 어떤 값을 제한으로 보는지, 소프트 스타트(램프)를 어디에 거는지만 다르다.

    CV   : 전압 = 설정값 (램프 적용), 전류 = 제한값
    CC   : 전류 = 설정값 (램프 적용), 전압 = 컴플라이언스 제한값
    CCCV : 전류 설정값으로 시작해 전압이 설정값에 닿으면 정전압 (전류에 램프 적용)

프로그램 전원(PowerProgram)은 이런 단계(PowerStep)를 차례로 실행하고, 묶음 전체를 반복할 수 있다
(다단 승압, 펄스 아노다이징). 단계가 바뀔 때는 그 단계의 램프 시간 동안 직전 단계의 제한값에서
새 제한값으로 선형으로 옮겨 간다 (램프 0 이면 계단). 맨 처음 단계는 0 에서 시작한다 (소프트 스타트).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from enum import StrEnum


class PowerMode(StrEnum):
    CV = "CV"
    CC = "CC"
    CCCV = "CCCV"


MODE_NAMES = {PowerMode.CV: "정전압", PowerMode.CC: "정전류", PowerMode.CCCV: "정전류→정전압"}


@dataclass(frozen=True)
class PowerSetting:
    """단일 단계 전원 설정 (공정 시간 내내 같은 모드·설정값)."""

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

    def as_program(self, duration: float) -> PowerProgram:
        return PowerProgram((PowerStep(self.mode, self.voltage, self.current, duration, self.ramp_time),))

    @property
    def max_voltage(self) -> float:
        return self.voltage


@dataclass(frozen=True)
class PowerStep:
    """프로그램의 한 단계. 의미는 PowerSetting 과 같고 지속 시간이 붙는다."""

    mode: PowerMode
    voltage: float            # V — CV 설정값 / CC·CCCV 전압 제한
    current: float            # A — CC·CCCV 설정값 / CV 전류 제한 (math.inf = 제한 없음)
    duration: float           # s
    ramp_time: float = 0.0    # s — 단계 시작에서 직전 제한값 → 이 단계 제한값으로 선형 변화

    def limits(self) -> tuple[float, float]:
        """(전압 제한, 전류 제한)."""
        return self.voltage, self.current

    def start_limits(self) -> tuple[float, float]:
        """맨 처음 단계의 출발점: 설정값(CV 는 전압, CC·CCCV 는 전류)을 0 에서 올린다."""
        if self.mode is PowerMode.CV:
            return 0.0, self.current
        return self.voltage, 0.0


@dataclass(frozen=True)
class PowerProgram:
    steps: tuple[PowerStep, ...]
    repeat: int = 1           # 단계 묶음 전체 반복 횟수

    @property
    def cycle(self) -> float:
        return sum(s.duration for s in self.steps)

    @property
    def duration(self) -> float:
        return self.repeat * self.cycle

    @property
    def max_voltage(self) -> float:
        return max(s.voltage for s in self.steps)

    def validate(self) -> list[str]:
        errors = []
        if not self.steps:
            return ["전원 프로그램에 단계가 하나 이상 있어야 합니다."]
        if self.repeat < 1:
            errors.append("반복 횟수는 1 이상이어야 합니다.")
        for k, s in enumerate(self.steps, start=1):
            if not s.duration > 0:
                errors.append(f"{k}단계: 시간은 0보다 커야 합니다.")
            if s.voltage < 0 or math.isnan(s.voltage):
                errors.append(f"{k}단계: 전압은 0 이상이어야 합니다.")
            if s.current < 0 or math.isnan(s.current):
                errors.append(f"{k}단계: 전류는 0 이상이어야 합니다.")
            if s.ramp_time < 0 or math.isnan(s.ramp_time):
                errors.append(f"{k}단계: 램프 시간은 0 이상이어야 합니다.")
            elif s.ramp_time > s.duration:
                errors.append(f"{k}단계: 램프 시간이 단계 시간보다 깁니다.")
        if all(s.voltage <= 0 or s.current <= 0 for s in self.steps):
            errors.append("모든 단계의 출력이 0 입니다.")
        return errors


@dataclass(frozen=True)
class Setpoint:
    voltage_limit: float      # V
    current_limit: float      # A


def _lerp(a: float, b: float, f: float) -> float:
    """선형 보간. 한쪽이 무한대(제한 없음)면 램프 없이 새 값으로."""
    if math.isinf(a) or math.isinf(b):
        return b
    return a + (b - a) * f


class PowerSupply:
    def __init__(self, setting: PowerSetting | PowerProgram, duration: float | None = None):
        """setting 이 단일 PowerSetting 이면 duration (공정 시간) 동안 그 설정을 유지하는 1단계 프로그램이 된다."""
        self.setting = setting
        if isinstance(setting, PowerProgram):
            self.program = setting
        else:
            self.program = setting.as_program(math.inf if duration is None else duration)
        # 단계 시작 시각 (반복 포함, 맨 끝은 프로그램 종료)
        self._starts: list[float] = []
        self._order: list[int] = []
        t = 0.0
        for _ in range(self.program.repeat):
            for k, step in enumerate(self.program.steps):
                self._starts.append(t)
                self._order.append(k)
                t += step.duration
        self._end = t

    def ramp_fraction(self, t: float) -> float:
        """현재 단계 램프의 진행률 (0–1)."""
        n, start = self._locate(t)
        ramp = self.program.steps[self._order[n]].ramp_time
        if ramp <= 0.0:
            return 1.0
        return min(1.0, max(0.0, (t - start) / ramp))

    def step_index(self, t: float) -> int:
        """t 에서 실행 중인 단계 번호 (0부터, 반복은 이어서 센다)."""
        return self._locate(t)[0]

    def setpoint(self, t: float) -> Setpoint:
        n, start = self._locate(t)
        step = self.program.steps[self._order[n]]
        target = step.limits()
        if n == 0:
            previous = step.start_limits()
        else:
            previous = self.program.steps[self._order[n - 1]].limits()
        f = self.ramp_fraction(t)
        return Setpoint(voltage_limit=_lerp(previous[0], target[0], f), current_limit=_lerp(previous[1], target[1], f))

    def next_breakpoint(self, t: float) -> float | None:
        """t 이후 설정값이 꺾이거나 바뀌는 가장 가까운 시각 (단계 시작, 램프 끝). 없으면 None.

        시간 적분이 이 시각을 건너뛰지 않도록 스텝을 맞춘다 (계단 변화는 정확히 그 시각에 일어난다).
        """
        n, start = self._locate(t)
        ramp_end = start + self.program.steps[self._order[n]].ramp_time
        candidates = [ramp_end] if ramp_end > t else []
        if n + 1 < len(self._starts):
            candidates.append(self._starts[n + 1])
        return min(candidates) if candidates else None

    def active_ramp(self, t: float) -> float:
        """t 가 램프 구간 안이면 그 램프 길이, 아니면 0 (램프 해상도로 스텝을 제한하는 데 쓴다)."""
        n, start = self._locate(t)
        ramp = self.program.steps[self._order[n]].ramp_time
        return ramp if start <= t < start + ramp else 0.0

    def _locate(self, t: float) -> tuple[int, float]:
        """(단계 번호, 그 단계 시작 시각). 단계 경계 시각은 새 단계에 속한다."""
        n = max(bisect.bisect_right(self._starts, t) - 1, 0)
        return n, self._starts[n]
