"""한계 판정(실행 중 플래그)과 설정 범위 경고."""

from __future__ import annotations

import numpy as np

from . import units as u

BREAKDOWN = "breakdown"
BURNING = "burning"
DISSOLUTION_LIMITED = "dissolution_limited"
CURRENT_LIMITED = "current_limited"

FLAG_MESSAGES = {
    BREAKDOWN: "막 전압이 절연파괴 전압을 넘었습니다. 스파크(PEO) 영역은 모델 적용 범위 밖입니다.",
    BURNING: "전류밀도가 burning 한계를 넘었습니다 (국부 과열·피막 손상 위험).",
    DISSOLUTION_LIMITED: "용해 속도가 성장 속도 이상입니다. 한계 두께에 도달했습니다.",
    CURRENT_LIMITED: "정류기 전류 제한(정전류)으로 동작한 구간이 있습니다.",
}

# 사용자에게 경고로 보여줄 플래그 (CURRENT_LIMITED 는 정보성)
WARNING_FLAGS = (BREAKDOWN, BURNING, DISSOLUTION_LIMITED)


def check(op, props, porous, rates) -> frozenset[str]:
    flags = set()
    if np.any(op.film_voltage > props.breakdown_voltage):
        flags.add(BREAKDOWN)
    if np.any(op.current_density > props.burning_current_density):
        flags.add(BURNING)
    if props.regime == "porous" and op.current > 0 and np.any((porous > 1e-9) & (rates.porous <= 0.0)):
        flags.add(DISSOLUTION_LIMITED)
    if op.control == "CC":
        flags.add(CURRENT_LIMITED)
    return frozenset(flags)


def range_warnings(config, preset, cathode) -> list[str]:
    """프리셋의 검증 범위를 벗어난 입력, 음극재 호환성 경고."""
    warnings = []
    t_c = u.kelvin_to_celsius(config.bath_temperature)
    if not preset.temperature_range.contains(config.bath_temperature):
        lo = u.kelvin_to_celsius(preset.temperature_range.low)
        hi = u.kelvin_to_celsius(preset.temperature_range.high)
        warnings.append(f"욕 온도 {t_c:.1f} °C 가 프리셋 범위({lo:.0f}–{hi:.0f} °C)를 벗어났습니다.")
    conc = preset.concentration_ref if config.electrolyte.concentration is None else config.electrolyte.concentration
    if not preset.concentration_range.contains(conc):
        r = preset.concentration_range
        warnings.append(f"{preset.species} 농도 {conc:g} {preset.concentration_unit} 가 프리셋 범위"
                        f"({r.low:g}–{r.high:g} {preset.concentration_unit})를 벗어났습니다.")
    if not preset.voltage_range.contains(config.power.voltage):
        r = preset.voltage_range
        warnings.append(f"전압 {config.power.voltage:g} V 가 프리셋 범위({r.low:g}–{r.high:g} V)를 벗어났습니다.")
    if not cathode.is_compatible(preset.tags):
        warnings.append(f"음극재 '{cathode.name}'은(는) 이 전해질에 적합하지 않습니다: {cathode.note}")
    if not preset.calibrated:
        warnings.append("전해질 파라미터가 아직 실측 데이터로 보정되지 않았습니다 (초기 추정값).")
    return warnings
