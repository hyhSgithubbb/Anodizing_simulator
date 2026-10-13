"""L1 표면 이산화·경로 저항과 위치별 두께 — 설계서 §1.7, §1.13."""

from dataclasses import replace

import numpy as np
import pytest

from anodsim.core import units as u
from anodsim.core.field import build_field, build_surface_mesh, path_lengths
from anodsim.core.geometry import Geometry
from anodsim.core.presets import PresetDB
from anodsim.core.simulator import Simulator

from conftest import ideal_presets, make_config


def _l1(fidelity="L1", **kwargs):
    return replace(make_config(**kwargs), fidelity=fidelity)


def _scaled_conductivity(db: PresetDB, preset_id: str, factor: float) -> PresetDB:
    preset = db.electrolyte(preset_id)
    values = tuple(v * factor for v in preset.conductivity_values)
    return PresetDB(db.metals, {**db.electrolytes, preset_id: replace(preset, conductivity_values=values)},
                    db.cathodes)


def _mirror(mesh, values, x_center):
    """x = x_center 에 대해 거울 대칭인 요소 쌍의 값 (같은 순서로 정렬)."""
    key = np.round(np.stack([np.abs(mesh.x - x_center), mesh.z]) / u.UM)
    left = mesh.x < x_center - 1e-12
    right = mesh.x > x_center + 1e-12
    order_l = np.lexsort(key[:, left])
    order_r = np.lexsort(key[:, right])
    return values[left][order_l], values[right][order_r]


# ── 표면 이산화 ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("level, faces", [(0.30, (0, 1, 2, 3)), (0.20, (0, 1, 2))])
def test_mesh_area_matches_wetted_area(level, faces):
    g = Geometry.default()
    g = replace(g, tank=replace(g.tank, liquid_level=level))
    mesh = build_surface_mesh(g, 120)
    assert mesh.faces == faces                     # 액면 위로 나온 판은 윗면이 없다
    assert mesh.area.sum() == pytest.approx(g.anode_area, rel=1e-12)
    assert mesh.edges[-1] == pytest.approx(mesh.length.sum())
    assert np.all(np.diff(mesh.s) > 0)
    assert abs(mesh.size - 120) <= len(faces)


def test_l0_field_matches_lumped_resistance():
    g = Geometry.default()
    field = build_field(g, "L0")
    assert field.mesh.size == 1
    assert field.resistance(55.0)[0, 0] == pytest.approx(g.lumped_resistance(55.0))


def test_path_lengths_direct_and_detour():
    g = Geometry.default()
    mesh = build_surface_mesh(g, 120)
    lengths = path_lengths(g, mesh)
    a, (c_left, c_right) = g.anode, g.cathodes
    gap = a.x_min - c_left.x_max
    # −x 면 가운데: 왼쪽 음극은 정면 (거리 = 극간), 오른쪽 음극은 판을 돌아간다
    i = np.flatnonzero(mesh.face == 0)[mesh.size // 8]
    assert lengths[i, 0] == pytest.approx(gap)
    walk = min(mesh.z[i] - a.z_bottom, a.z_top - mesh.z[i]) + a.thickness
    assert lengths[i, 1] == pytest.approx(walk + (c_right.x_min - a.x_max))
    # 음극이 덮지 않는 높이는 음극 끝까지의 대각선 거리
    short = replace(g, cathodes=(replace(c_left, z_bottom=0.15, height=0.15),))
    lengths = path_lengths(short, build_surface_mesh(short, 120))
    low = np.flatnonzero((mesh.face == 0) & (mesh.z < 0.10))[0]
    assert lengths[low, 0] == pytest.approx(np.hypot(gap, 0.15 - mesh.z[low]))


def test_detour_only_below_when_plate_pierces_surface():
    g = Geometry.default()
    g = replace(g, tank=replace(g.tank, liquid_level=0.20))   # 양극 윗부분이 액면 위로
    mesh = build_surface_mesh(g, 120)
    lengths = path_lengths(g, mesh)
    a = g.anode
    top = np.flatnonzero(mesh.face == 0)[0]                  # 액면 바로 아래 −x 면
    expected = (mesh.z[top] - a.z_bottom) + a.thickness + (g.cathodes[1].x_min - a.x_max)
    assert lengths[top, 1] == pytest.approx(expected)


def test_resistance_scales_inversely_with_conductivity():
    field = build_field(Geometry.default(), "L1")
    r1, r2 = field.resistance(10.0), field.resistance(20.0)
    assert np.allclose(r2, r1 / 2)
    assert np.count_nonzero(r1 - np.diag(np.diag(r1))) == 0     # L1 은 대각 행렬


# ── 위치별 두께 ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("fidelity", ["L1", "L2"])
def test_symmetric_layout_gives_symmetric_distribution(presets, fidelity):
    """§1.13-5 (엔진 수준): 좌우 대칭 배치 → 좌우 대칭 두께."""
    sim = Simulator(_l1(fidelity, metal="Ti", preset="ti_nanotube_eg", cathode="pt", voltage=60.0, ramp=30.0,
                        minutes=10.0), presets)
    res = sim.run()
    left, right = _mirror(res.mesh, res.final.total, Geometry.default().anode.x_center)
    assert left.size == right.size > 10
    assert np.allclose(left, right, rtol=1e-9)        # L2 격자도 거울 대칭으로 만든다
    assert res.uniformity > 0.01        # 저전도 전해질 → 판 끝이 더 두껍다 (분포가 실제로 생김)


@pytest.mark.parametrize("fidelity", ["L1", "L2"])
def test_one_sided_cathode_thins_back_face(presets, fidelity):
    g = Geometry.default()
    one = replace(g, cathodes=g.cathodes[:1])                  # 왼쪽 음극만
    res = Simulator(_l1(fidelity, metal="Ti", preset="ti_nanotube_eg", cathode="pt", voltage=60.0, ramp=30.0,
                        minutes=10.0, geometry=one), presets).run()
    m, h = res.mesh, res.final.total
    assert h[m.face == 0].mean() > 1.1 * h[m.face == 2].mean()   # 음극을 마주보는 면이 두껍다
    back = m.face == 2
    mid = np.argmin(np.abs(m.z[back] - (g.anode.z_bottom + 0.5 * g.anode.height)))
    assert h[back][mid] == pytest.approx(h[back].min(), rel=0.02)  # 등진 면은 가운데(우회 경로 최장)가 가장 얇다


@pytest.mark.parametrize("fidelity", ["L1", "L2"])
def test_wagner_limit_high_conductivity_is_uniform(presets, fidelity):
    """§1.13-8: κ → ∞ 이면 막 저항만 남아 분포가 균일해진다. κ 가 작을수록 편차가 커진다."""
    spreads = []
    for factor in (1e-1, 1.0, 1e6):
        db = _scaled_conductivity(presets, "ti_nanotube_eg", factor)
        res = Simulator(_l1(fidelity, metal="Ti", preset="ti_nanotube_eg", cathode="pt", voltage=60.0, ramp=30.0,
                            minutes=5.0), db).run()
        spreads.append((res.uniformity, res.wagner))
    assert spreads[2][0] < 1e-4
    assert spreads[0][0] > spreads[1][0] > spreads[2][0]
    assert spreads[0][1] < spreads[1][1] < spreads[2][1]        # Wa ∝ κ


@pytest.mark.parametrize("fidelity", ["L1", "L2"])
@pytest.mark.parametrize("preset_id, metal, cathode, voltage", [
    ("al_sulfuric_typeII", "Al", "pb", 15.0),
    ("ti_color_acid", "Ti", "ti", 50.0),
])
def test_charge_conservation_per_element(presets, preset_id, metal, cathode, voltage, fidelity):
    """§1.13-3 을 요소마다: η·k_F·Q_i = 요소 i 의 치밀 산화물 부피."""
    db = ideal_presets(presets, preset_id)
    g = Geometry.default()
    sim = Simulator(_l1(fidelity, metal=metal, preset=preset_id, cathode=cathode, voltage=voltage, ramp=20.0,
                        minutes=5.0, geometry=replace(g, cathodes=g.cathodes[:1])), db)
    final = sim.run().final
    m = sim.model
    dense = (final.barrier - sim.metal.native_oxide_thickness) + (1 - m.props.porosity) * final.porous
    assert np.allclose(dense, m.eta * m.k_F * final.charge, rtol=1e-9)
    assert np.ptp(final.charge) > 0                             # 요소마다 다른 전하


def test_l1_cc_mode_holds_total_current(presets):
    sim = Simulator(_l1(mode="CCCV", voltage=25.0, current=2.0, ramp=10.0, minutes=5.0), presets)
    res = sim.run()
    cc = np.array([c == "CC" for c in res.control]) & (res.time > 10.0)
    assert cc.any()
    assert np.allclose(res.current[cc], 2.0, rtol=1e-8)


@pytest.mark.parametrize("fidelity", ["L1", "L2"])
def test_distributed_mean_close_to_l0(presets, fidelity):
    """막 저항이 지배하는 황산 Type II 에서는 L1·L2 평균이 L0 와 비슷해야 한다."""
    l0 = Simulator(make_config(minutes=20.0), presets).run()
    res = Simulator(_l1(fidelity, minutes=20.0), presets).run()
    assert res.total[-1] == pytest.approx(l0.total[-1], rel=0.03)
    assert res.uniformity < 0.02


def test_profile_history_shape(presets):
    res = Simulator(_l1(minutes=5.0), presets).run()
    k, n = res.profile_total.shape
    assert n == res.mesh.size and k == res.profile_time.size == res.profile_current_density.shape[0]
    assert 50 < k <= res.time.size
    assert res.profile_time[0] == 0.0 and res.profile_time[-1] == pytest.approx(300.0)
    assert np.allclose(res.profile_total[-1], res.final.total)
    assert np.all(np.diff(res.profile_total.mean(axis=1)) >= -1e-15)   # 두께는 줄지 않는다 (용해 < 성장)


def test_invalid_element_count_rejected(presets):
    from anodsim.core.simulator import ConfigError
    with pytest.raises(ConfigError, match="요소 수"):
        Simulator(replace(_l1(), elements=1), presets)
