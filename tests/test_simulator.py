"""시간 적분 엔진 — 설계서 §1.13 검증 항목."""

import math
from dataclasses import replace

import numpy as np
import pytest

from anodsim.core import limits
from anodsim.core import units as u
from anodsim.core.simulator import ConfigError, Simulator

from conftest import ideal_presets, make_config


def test_720_rule(presets):
    """§1.13-2: η=1, P=0 이면 0.33 μm/(A/dm²·min) — 황산 아노다이징 720 rule(0.328)과 2 % 이내."""
    db = ideal_presets(presets, "al_sulfuric_typeII", current_efficiency=1.0, porosity=0.0)
    probe = Simulator(make_config(), db)
    current = 1.0 * u.A_PER_DM2 * probe.areas.sum()               # 1 A/dm²
    sim = Simulator(make_config(mode="CC", voltage=100.0, current=current, minutes=10.0), db)
    res = sim.run()
    grown = (res.total[-1] - res.total[0]) / u.UM
    rate = grown / (1.0 * 10.0)                                   # μm / (A/dm² · min)
    assert rate == pytest.approx(0.328, rel=0.02)


@pytest.mark.parametrize("preset_id, metal, cathode, voltage", [
    ("al_sulfuric_typeII", "Al", "pb", 15.0),
    ("al_oxalic", "Al", "pt", 40.0),
    ("ti_color_acid", "Ti", "ti", 50.0),
])
def test_charge_conservation(presets, preset_id, metal, cathode, voltage):
    """§1.13-3: η·k_F·∫j dt = 생성된 치밀 산화물 부피 (용해 없음)."""
    db = ideal_presets(presets, preset_id)
    sim = Simulator(make_config(metal=metal, preset=preset_id, cathode=cathode, voltage=voltage,
                                ramp=20.0, minutes=15.0), db)
    res = sim.run()
    m = sim.model
    final = res.final
    dense_volume = (final.barrier - sim.metal.native_oxide_thickness) + (1 - m.props.porosity) * final.porous
    assert dense_volume[0] == pytest.approx(m.eta * m.k_F * final.charge[0], rel=1e-9)
    assert final.metal_loss[0] == pytest.approx(m.eta * m.k_M * final.charge[0], rel=1e-9)


def test_cc_mode_holds_current(presets):
    """§1.13-4: 정전류 구간에서 총전류 = 설정값."""
    sim = Simulator(make_config(mode="CCCV", voltage=25.0, current=2.0, ramp=10.0, minutes=20.0), presets)
    res = sim.run()
    cc = np.array([c == "CC" for c in res.control]) & (res.time > 10.0)
    assert cc.any()
    assert np.allclose(res.current[cc], 2.0, rtol=1e-8)
    assert np.all(res.voltage <= 25.0 + 1e-9)


def test_ti_cv_anodizing_ratio(presets):
    """§1.13-6: Ti 치밀막 정전압 장시간 → δ_b/U_f ≈ k_a (2.5 nm/V) ±10 %."""
    sim = Simulator(make_config(metal="Ti", preset="ti_color_acid", cathode="ti", voltage=50.0,
                                ramp=10.0, minutes=10.0), presets)
    final = sim.run().final
    ratio = final.barrier[0] / final.film_voltage[0]
    assert ratio == pytest.approx(2.5e-9, rel=0.10)
    assert ratio == pytest.approx(sim.props.high_field.anodizing_ratio(final.current_density[0]), rel=1e-6)


def test_type_ii_regression(presets):
    """Al 황산 Type II 15 V, 60 min: 실무 범위 (1–2 A/dm², 약 15–25 μm/h)."""
    res = Simulator(make_config(mode="CCCV", current=10.0, ramp=30.0, minutes=60.0), presets).run()
    j = res.final.current_density[0] / u.A_PER_DM2
    assert 0.8 < j < 2.0
    assert 12.0 < res.final.total[0] / u.UM < 30.0
    assert 1.0e-9 < res.final.barrier[0] / res.final.film_voltage[0] < 1.4e-9


def test_porous_steady_state_is_stable(presets):
    """회귀: 정상상태에서 stiff 배리어 식이 진동하지 않아야 한다."""
    res = Simulator(make_config(ramp=30.0, minutes=30.0), presets).run()
    late = res.time > 600.0
    j = res.current_density[late]
    assert np.std(j) / np.mean(j) < 1e-6


def test_time_step_convergence(presets):
    cfg = make_config(metal="Ti", preset="ti_color_acid", cathode="ti", voltage=50.0, minutes=10.0)
    coarse = Simulator(cfg, presets).run().final
    fine = Simulator(cfg, presets, max_step=0.15, barrier_tolerance=0.001, current_tolerance=0.01).run().final
    assert coarse.total[0] == pytest.approx(fine.total[0], rel=1e-3)
    assert coarse.charge[0] == pytest.approx(fine.charge[0], rel=2e-3)


@pytest.mark.parametrize("preset_id", [
    "al_sulfuric_typeII", "al_sulfuric_hard", "al_oxalic", "al_phosphoric",
    "al_borate_barrier", "ti_color_acid", "ti_nanotube_eg",
])
def test_every_preset_runs(presets, preset_id):
    e = presets.electrolyte(preset_id)
    voltage = 0.5 * (e.voltage_range.low + e.voltage_range.high)
    cfg = make_config(metal=e.metal, preset=preset_id, cathode=e.recommended_cathodes[0], voltage=voltage,
                      ramp=10.0, minutes=5.0, temp_c=u.kelvin_to_celsius(e.temperature_ref))
    res = Simulator(cfg, presets).run()
    for arr in (res.voltage, res.current, res.barrier, res.porous, res.charge):
        assert np.all(np.isfinite(arr))
    assert res.total[-1] > res.total[0]
    assert np.all(np.diff(res.charge) >= 0)
    assert res.time[-1] == pytest.approx(cfg.duration)


def test_breakdown_flag(presets):
    """치밀막: 막 전압이 절연파괴 전압(Ti 착색 프리셋 120 V)을 넘으면 경고."""
    res = Simulator(make_config(metal="Ti", preset="ti_color_acid", cathode="ti", mode="CCCV", voltage=150.0,
                                current=1.0, ramp=10.0, minutes=10.0), presets).run()
    assert limits.BREAKDOWN in res.flags


def test_burning_flag(presets):
    """다공성막: 과전압이면 정상 전류가 급증해 burning 한계를 넘는다 (황산 80 V)."""
    res = Simulator(make_config(voltage=80.0, ramp=5.0, minutes=1.0), presets).run()
    assert limits.BURNING in res.flags


def test_range_and_cathode_warnings(presets):
    sim = Simulator(make_config(metal="Ti", preset="ti_nanotube_eg", cathode="sus316", voltage=60.0,
                                temp_c=60.0, minutes=1.0), presets)
    text = " ".join(sim.warnings)
    assert "욕 온도" in text and "SUS316" in text and "보정" in text


def test_invalid_config_is_rejected(presets):
    with pytest.raises(ConfigError) as err:
        Simulator(make_config(metal="Ti", preset="al_sulfuric_typeII"), presets)
    assert any("Al용" in e for e in err.value.errors)
    with pytest.raises(ConfigError):
        Simulator(replace(make_config(), duration=0.0), presets)


def test_callback_can_stop(presets):
    sim = Simulator(make_config(minutes=10.0), presets)
    res = sim.run(callback=lambda snap: snap.time < 60.0)
    assert 60.0 <= res.time[-1] < 600.0
    assert not sim.done


def test_no_current_without_cathode_overpotential_budget(presets):
    """전압이 너무 낮으면 (램프 시작) 전류는 0 이고 상태는 유한해야 한다."""
    sim = Simulator(make_config(voltage=15.0, ramp=60.0, minutes=0.5), presets)
    first = sim.snapshot()
    assert first.current == 0.0 and math.isfinite(first.voltage)
