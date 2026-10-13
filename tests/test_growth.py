"""성장 모델 (치밀형 / 다공성형)."""

import numpy as np
import pytest

from anodsim.core.growth import CompactGrowth, PorousGrowth, make_growth_model


def model_for(presets, preset_id, metal):
    preset = presets.electrolyte(preset_id)
    return make_growth_model(presets.metal(metal), preset.properties(preset.temperature_ref))


def test_factory(presets):
    assert isinstance(model_for(presets, "ti_color_acid", "Ti"), CompactGrowth)
    assert isinstance(model_for(presets, "al_oxalic", "Al"), PorousGrowth)


def test_compact_rates_follow_faraday(presets):
    m = model_for(presets, "ti_color_acid", "Ti")
    j = np.array([5.0])
    r = m.rates(j, m.film_voltage(j, 50e-9), np.array([50e-9]), np.array([0.0]))
    assert r.barrier[0] == pytest.approx(m.eta * m.k_F * 5.0)
    assert r.porous[0] == 0.0
    assert r.metal[0] == pytest.approx(m.eta * m.k_M * 5.0)


def test_porous_charge_partition(presets):
    m = model_for(presets, "al_sulfuric_typeII", "Al")
    p = m.props
    u_f = np.array([15.0])
    j_ss = m.steady_current_density(u_f)[0]
    assert j_ss == pytest.approx(p.j_ss_ref)                      # U_f = U_ref

    # 초과 전류: 차이만큼 배리어로, 정상 전류만큼 기공층으로
    r = m.rates(np.array([3 * j_ss]), u_f, np.array([10e-9]), np.array([0.0]))
    assert r.barrier[0] == pytest.approx(m.eta * m.k_F * 2 * j_ss)
    assert r.porous[0] == pytest.approx(m.eta * m.k_F * j_ss / (1 - p.porosity))

    # 정상상태: 배리어 변화 0, 전하는 모두 기공층으로 (기공층이 있으면 용해만큼 감소)
    r = m.rates(np.array([j_ss]), u_f, np.array([16e-9]), np.array([1e-6]))
    assert r.barrier[0] == pytest.approx(0.0, abs=1e-25)
    dissolution = p.surface_dissolution + p.pore_wall_dissolution * 1e-6
    assert r.porous[0] == pytest.approx(m.eta * m.k_F * j_ss / (1 - p.porosity) - dissolution)

    # 전류 부족(전압을 낮춤): 배리어가 v_thin 상한으로 얇아진다
    r = m.rates(np.array([1e-3 * j_ss]), u_f, np.array([20e-9]), np.array([0.0]))
    assert r.barrier[0] == pytest.approx(-p.barrier_thinning)


def test_steady_barrier_thickness_ratio(presets):
    m = model_for(presets, "al_sulfuric_typeII", "Al")
    ratio = m.steady_barrier_thickness(np.array([15.0]))[0] / 15.0
    assert 1.0e-9 < ratio < 1.4e-9                                 # 다공성 알루미나 ≈ 1–1.3 nm/V


@pytest.mark.parametrize("preset_id, metal", [("al_sulfuric_typeII", "Al"), ("ti_color_acid", "Ti")])
def test_barrier_partials(presets, preset_id, metal):
    m = model_for(presets, preset_id, metal)
    j, u_f, delta = np.array([150.0]), np.array([14.0]), np.array([16e-9])
    f_j, f_u = m.barrier_rate_partials(j, u_f, delta)
    hj, hu = 1e-4, 1e-6
    fd_j = (m.barrier_rate(j + hj, u_f, delta) - m.barrier_rate(j - hj, u_f, delta)) / (2 * hj)
    fd_u = (m.barrier_rate(j, u_f + hu, delta) - m.barrier_rate(j, u_f - hu, delta)) / (2 * hu)
    assert f_j[0] == pytest.approx(fd_j[0], rel=1e-6)
    assert f_u[0] == pytest.approx(fd_u[0], rel=1e-5, abs=1e-30)
