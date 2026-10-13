"""P4: 프로그램 전원(다단계·펄스)과 욕 열수지 — 설계서 §1.8, §1.9."""

import math
from dataclasses import replace

import numpy as np
import pytest

from anodsim.core import limits
from anodsim.core import units as u
from anodsim.core.geometry import Geometry
from anodsim.core.power_supply import PowerMode, PowerProgram, PowerSetting, PowerStep, PowerSupply
from anodsim.core.simulator import ConfigError, Simulator
from anodsim.core.thermal import BathThermal, ThermalSetting

from conftest import ideal_presets, make_config

CV, CC = PowerMode.CV, PowerMode.CC


def _two_step() -> PowerProgram:
    return PowerProgram((PowerStep(CV, 15.0, math.inf, 600.0, 30.0), PowerStep(CV, 20.0, math.inf, 600.0, 60.0)))


# ── 전원 프로그램 ────────────────────────────────────────────────────────────
def test_single_setting_matches_previous_behaviour():
    cv = PowerSupply(PowerSetting(CV, 20.0, 3.0, 10.0), 600.0)
    assert cv.setpoint(5.0).voltage_limit == pytest.approx(10.0)
    assert cv.setpoint(5.0).current_limit == 3.0
    assert cv.setpoint(50.0).voltage_limit == 20.0
    cc = PowerSupply(PowerSetting(CC, 60.0, 2.0, 20.0), 600.0)
    assert cc.setpoint(5.0).current_limit == pytest.approx(0.5)
    assert cc.setpoint(5.0).voltage_limit == 60.0
    assert cc.next_breakpoint(0.0) == 20.0 and cc.next_breakpoint(20.0) is None


def test_program_setpoints_ramps_and_breakpoints():
    supply = PowerSupply(_two_step())
    assert supply.setpoint(15.0).voltage_limit == pytest.approx(7.5)        # 첫 단계: 0 에서 소프트 스타트
    assert supply.setpoint(599.0).voltage_limit == 15.0
    assert supply.setpoint(600.0).voltage_limit == pytest.approx(15.0)      # 경계 시각은 새 단계 (램프 시작)
    assert supply.step_index(600.0) == 1
    assert supply.setpoint(630.0).voltage_limit == pytest.approx(17.5)      # 15 → 20 V, 60 s 램프의 절반
    assert supply.setpoint(700.0).voltage_limit == 20.0
    assert math.isinf(supply.setpoint(700.0).current_limit)
    points, t = [], 0.0
    while (t := supply.next_breakpoint(t)) is not None:
        points.append(t)
    assert points == [30.0, 600.0, 660.0]


def test_program_repeat_and_mode_change():
    program = PowerProgram((PowerStep(CC, 40.0, 2.0, 60.0), PowerStep(CV, 10.0, math.inf, 30.0)), repeat=3)
    assert program.duration == 270.0 and program.max_voltage == 40.0
    supply = PowerSupply(program)
    assert supply.setpoint(10.0).current_limit == 2.0 and supply.setpoint(10.0).voltage_limit == 40.0
    sp = supply.setpoint(70.0)                   # 정전압 단계: 전류 제한 없음 (2 A → ∞ 는 램프 없이)
    assert sp.voltage_limit == 10.0 and math.isinf(sp.current_limit)
    assert supply.step_index(100.0) == 2 and supply.setpoint(100.0).current_limit == 2.0
    assert supply.step_index(269.0) == 5


@pytest.mark.parametrize("program, message", [
    (PowerProgram(()), "하나 이상"),
    (PowerProgram((PowerStep(CV, 10.0, 1.0, 0.0),)), "시간"),
    (PowerProgram((PowerStep(CV, 10.0, 1.0, 10.0, 20.0),)), "램프"),
    (PowerProgram((PowerStep(CV, -1.0, 1.0, 10.0),)), "전압"),
    (PowerProgram((PowerStep(CV, 0.0, 1.0, 10.0),)), "출력이 0"),
    (PowerProgram((PowerStep(CV, 10.0, 1.0, 10.0),), repeat=0), "반복"),
])
def test_program_validation(program, message):
    assert any(message in e for e in program.validate())


def test_program_run_hits_breakpoints_and_follows_steps(presets):
    res = Simulator(replace(make_config(minutes=20.0), power=_two_step()), presets).run()
    assert 600.0 in res.time and 660.0 in res.time and 30.0 in res.time
    k1 = np.searchsorted(res.time, 590.0)
    k2 = np.searchsorted(res.time, 1190.0)
    assert res.voltage[k1] == pytest.approx(15.0) and res.voltage[k2] == pytest.approx(20.0)
    assert res.current_density[k2] > 2.0 * res.current_density[k1]          # 높은 전압 → 큰 전류
    assert set(res.step) == {0, 1} and res.step[k1] == 0 and res.step[k2] == 1


def test_pulse_program_switches_output_off(presets):
    pulse = PowerProgram((PowerStep(CV, 20.0, math.inf, 30.0), PowerStep(CV, 0.0, math.inf, 30.0)), repeat=4)
    res = Simulator(replace(make_config(minutes=4.0), power=pulse), presets).run()
    off = np.array([c == "OFF" for c in res.control])
    assert off.any() and (~off).any()
    assert np.all(res.current[off] == 0.0)
    charge = np.diff(res.charge)
    assert np.all(charge[off[:-1]] == 0.0)          # 꺼진 상태에서 시작한 스텝은 전하가 늘지 않는다
    assert set(np.unique(res.step)) == set(range(8))


def test_charge_conservation_under_program(presets):
    db = ideal_presets(presets, "al_sulfuric_typeII")
    program = PowerProgram((PowerStep(CC, 30.0, 1.5, 120.0, 10.0), PowerStep(CV, 18.0, 3.0, 120.0, 20.0),
                            PowerStep(CV, 0.0, math.inf, 30.0)), repeat=2)
    sim = Simulator(replace(make_config(minutes=program.duration / 60.0), power=program), db)
    final = sim.run().final
    m = sim.model
    dense = (final.barrier - sim.metal.native_oxide_thickness) + (1 - m.props.porosity) * final.porous
    assert dense[0] == pytest.approx(m.eta * m.k_F * final.charge[0], rel=1e-9)


def test_duration_longer_than_program_is_rejected(presets):
    with pytest.raises(ConfigError, match="프로그램 길이"):
        Simulator(replace(make_config(minutes=21.0), power=_two_step()), presets)
    res = Simulator(replace(make_config(minutes=15.0), power=_two_step()), presets).run()   # 짧게 자르기는 허용
    assert res.time[-1] == pytest.approx(900.0)


def test_range_warning_uses_highest_program_voltage(presets):
    program = PowerProgram((PowerStep(CV, 15.0, math.inf, 60.0), PowerStep(CV, 40.0, math.inf, 60.0)))
    sim = Simulator(replace(make_config(minutes=2.0), power=program), presets)
    assert any("40 V" in w for w in sim.warnings)


# ── 열수지 ──────────────────────────────────────────────────────────────────
def _hard(thermal, minutes=30.0):
    return replace(make_config(preset="al_sulfuric_hard", mode="CCCV", voltage=60.0, current=10.0, temp_c=0.0,
                               ramp=30.0, minutes=minutes), thermal=thermal)


def test_liquid_volume_and_heat_capacity(presets):
    g = Geometry.default()
    electrodes = 0.002 * 0.10 * 0.15 + 2 * 0.003 * 0.15 * 0.25      # 잠긴 양극 + 음극 2장 (액면 30 cm)
    assert g.liquid_volume() == pytest.approx(0.40 * 0.25 * 0.30 - electrodes)
    preset = presets.electrolyte("al_sulfuric_typeII")
    bath = BathThermal.from_config(g, preset, ThermalSetting(10.0, 293.15))
    assert bath.heat_capacity == pytest.approx(1100.0 * 3550.0 * g.liquid_volume())
    assert bath.time_constant == pytest.approx(bath.heat_capacity / 10.0)
    assert math.isinf(BathThermal.from_config(g, preset, ThermalSetting(0.0, 293.15)).time_constant)


@pytest.mark.parametrize("external", [0.0, 0.3])
def test_adiabatic_energy_balance(presets, external):
    """단열: C·ΔT = Σ (U − R_ext·I)·I·Δt — 외부 저항(케이블)의 줄열은 욕 밖. 상대오차 1e-9."""
    sim = Simulator(replace(_hard(ThermalSetting(0.0, u.celsius_to_kelvin(0.0))), external_resistance=external),
                    presets)
    res = sim.run()
    v, i = res.voltage[:-1], res.current[:-1]
    energy = np.sum((v - external * i) * i * np.diff(res.time))
    assert sim.thermal.heat_capacity * (res.temperature[-1] - res.temperature[0]) == pytest.approx(energy, rel=1e-9)
    assert res.temperature[-1] > res.temperature[0] + 3.0                      # 수백 W → 수 K 상승
    assert np.all(np.diff(res.temperature) >= 0.0)
    assert res.thermal


def test_cooling_limits_temperature_rise(presets):
    hot = Simulator(_hard(ThermalSetting(0.0, u.celsius_to_kelvin(0.0))), presets).run()
    cooled = Simulator(_hard(ThermalSetting(50.0, u.celsius_to_kelvin(0.0))), presets).run()
    rise_hot = hot.temperature[-1] - hot.temperature[0]
    rise_cooled = cooled.temperature[-1] - cooled.temperature[0]
    assert 0.0 < rise_cooled < 0.8 * rise_hot
    power = cooled.voltage[-1] * cooled.current[-1]
    assert rise_cooled < power / 50.0                                          # 정상상태 상승 P/UA 를 넘지 않는다


def test_steady_state_approaches_power_over_ua(presets):
    """냉각 시정수보다 훨씬 오래 돌리면 T → T_cool + P/UA (정전류 → 전력이 거의 일정)."""
    cfg = replace(_hard(ThermalSetting(400.0, u.celsius_to_kelvin(0.0)), minutes=60.0))
    sim = Simulator(cfg, presets)
    assert sim.thermal.time_constant < 600.0                                  # 시정수 < 10 min
    res = sim.run()
    power = res.voltage[-1] * res.current[-1]
    assert res.temperature[-1] - u.celsius_to_kelvin(0.0) == pytest.approx(power / 400.0, rel=0.02)


def test_strong_cooling_is_stable(presets):
    """냉각 시정수(≈1 s)가 최대 스텝보다 짧아도 욕 온도가 진동하지 않고 T_cool + P/UA 에 머문다."""
    sim = Simulator(_hard(ThermalSetting(1e5, u.celsius_to_kelvin(0.0)), minutes=60.0), presets)
    assert sim.thermal.time_constant < 0.5 * sim.max_step
    res = sim.run()
    rise = res.temperature - u.celsius_to_kelvin(0.0)
    power = res.voltage * res.current
    assert np.all(rise >= 0.0) and np.all(rise <= power.max() / 1e5 * (1 + 1e-9))
    assert rise[-1] == pytest.approx(power[-1] / 1e5, rel=1e-3)


def test_properties_follow_bath_temperature(presets):
    sim = Simulator(_hard(ThermalSetting(0.0, u.celsius_to_kelvin(0.0))), presets)
    res = sim.run()
    assert sim.props.temperature == pytest.approx(res.temperature[-1], abs=0.06)
    preset = presets.electrolyte("al_sulfuric_hard")
    assert sim.props.conductivity == pytest.approx(preset.conductivity(sim.props.temperature), rel=1e-12)
    assert sim.props.conductivity > preset.conductivity(u.celsius_to_kelvin(0.0))
    assert np.allclose(sim.circuit.resistance, sim.field.resistance(sim.props.conductivity))


def test_temperature_range_flag(presets):
    """경질 피막(검증 범위 −10–10 °C)을 냉각 없이 1 h → 10 °C 를 넘으면 경고."""
    res = Simulator(_hard(ThermalSetting(0.0, u.celsius_to_kelvin(0.0)), minutes=60.0), presets).run()
    assert res.temperature[-1] > u.celsius_to_kelvin(10.0)
    assert limits.TEMPERATURE_RANGE in res.flags
    iso = Simulator(_hard(None, minutes=5.0), presets).run()
    assert not iso.thermal and np.all(iso.temperature == u.celsius_to_kelvin(0.0))
    assert limits.TEMPERATURE_RANGE not in iso.flags


def test_thermal_validation(presets):
    with pytest.raises(ConfigError, match="냉각 능력"):
        Simulator(_hard(ThermalSetting(-1.0, 273.15)), presets)


def test_cv_bath_heating_increases_current(presets):
    """정전압에서는 욕이 데워지면 전도도·j_ss 가 커져 전류가 늘어난다 (등온과 비교)."""
    cfg = make_config(voltage=18.0, minutes=60.0)
    iso = Simulator(cfg, presets).run()
    hot = Simulator(replace(cfg, thermal=ThermalSetting(0.0, u.celsius_to_kelvin(20.0))), presets).run()
    assert hot.temperature[-1] > iso.temperature[-1] + 0.5
    assert hot.current[-1] > iso.current[-1]
