"""L2 FDM 저항 행렬·전위장 — 설계서 §1.13-7."""

from dataclasses import replace

import numpy as np
import pytest

from anodsim.core.field import build_field
from anodsim.core.field.fdm import CATHODE, ELECTROLYTE, grid_lines, solve_fdm
from anodsim.core.field.mesh import build_surface_mesh
from anodsim.core.geometry import Geometry, PlateElectrode, Tank


def _total_resistance(r: np.ndarray) -> float:
    """양극 전체가 등전위일 때의 저항 1 / (1ᵀ R⁻¹ 1)."""
    ones = np.ones(r.shape[0])
    return 1.0 / float(ones @ np.linalg.solve(r, ones))


def _parallel_plates() -> Geometry:
    """양극 판이 액면을 뚫고 바닥 1 mm 위까지, 양쪽 음극이 바닥부터 액면 위까지 — 거의 1D 평행판."""
    tank = Tank(width=0.20, depth=0.25, height=0.35, liquid_level=0.30)
    anode = PlateElectrode(x_center=0.10, z_bottom=0.001, height=0.35, thickness=0.002, width=0.10)
    cathode = PlateElectrode(x_center=0.05, z_bottom=0.0, height=0.35, thickness=0.003, width=0.10)
    return Geometry(tank, anode, (cathode, replace(cathode, x_center=0.15)))


def test_grid_lines_hit_keys_and_grade():
    lines = grid_lines([0.1, 0.1004, 0.25], [0.1], 0.0, 0.4, 0.5e-3, 5e-3)
    assert lines[0] == 0.0 and lines[-1] == 0.4
    for key in (0.1, 0.1004, 0.25):
        assert np.min(np.abs(lines - key)) < 1e-12
    steps = np.diff(lines)
    assert steps.min() > 0
    assert steps.max() <= 5e-3 * 1.3 + 1e-12
    near = steps[np.searchsorted(lines, 0.1) - 1]
    assert near <= 0.5e-3 * 1.3 + 1e-12            # 전극 표면 옆은 촘촘하다


def test_parallel_plates_match_analytic():
    """평행판: R = L / (κ · 2h · D) 와 1 % 이내 (양면, 단위 κ)."""
    g = _parallel_plates()
    assert g.validate() == []
    field = build_field(g, "L2")
    a, c = g.anode, g.cathodes[0]
    gap = a.x_min - c.x_max
    h = g.tank.liquid_level - a.z_bottom
    expected = gap / (1.0 * 2.0 * h * field.fdm.depth)
    assert _total_resistance(field.unit_resistance) == pytest.approx(expected, rel=0.01)
    # 면 가운데 요소는 1D 해와 같은 전류밀도 (양 끝은 모서리 효과)
    current = np.linalg.solve(field.unit_resistance, np.ones(field.mesh.size))
    j = current / field.mesh.area
    mid = np.argmin(np.abs(field.mesh.z - 0.15) + 10.0 * (field.mesh.face != 0))
    assert j[mid] == pytest.approx(1.0 / gap, rel=0.02)            # κ = 1, 전위 1 V


def test_resistance_matrix_is_symmetric_positive_definite():
    field = build_field(Geometry.default(), "L2")
    r = field.unit_resistance
    assert np.array_equal(r, r.T)
    assert np.linalg.eigvalsh(r).min() > 0
    assert np.count_nonzero(r - np.diag(np.diag(r))) > 0          # L1 과 달리 요소 사이 결합이 있다
    assert np.allclose(field.resistance(20.0), r / 20.0)


def test_grid_convergence():
    """격자를 두 배 촘촘히 해도 전체 저항과 균일 전류밀도 응답 R·a 는 0.5 % 이내로 같다.

    (요소 하나에만 전류를 넣는 대각 성분과 등전위 1차 분포 R⁻¹ 의 모서리 값은 특이점 때문에 격자에 민감하지만,
    시뮬레이션에서는 막 저항이 분포를 완만하게 만들어 최종 두께가 0.1 % 이내로 수렴한다 — 설계서 §1.13.)
    """
    g = Geometry.default()
    mesh = build_surface_mesh(g, 60)
    coarse, fine = solve_fdm(g, mesh, 1.0), solve_fdm(g, mesh, 0.5)
    assert fine.x.size > 1.5 * coarse.x.size
    rc, rf = coarse.unit_resistance, fine.unit_resistance
    assert _total_resistance(rc) == pytest.approx(_total_resistance(rf), rel=0.005)
    assert np.allclose(rc @ mesh.area, rf @ mesh.area, rtol=0.005)


def test_l2_captures_edge_concentration_l1_does_not():
    """음극 1장이 판보다 길게 마주볼 때: 1차 전류 분포에서 L2 는 마주보는 면의 양 끝이 진하고, L1 은 균일."""
    g = Geometry.default()
    one = replace(g, cathodes=g.cathodes[:1])
    ratios = {}
    for level in ("L1", "L2"):
        field = build_field(one, level)
        m = field.mesh
        j = np.linalg.solve(field.unit_resistance, np.ones(m.size)) / m.area
        front = np.flatnonzero(m.face == 0)
        ratios[level] = (j[front[0]] / j[front[front.size // 2]], j[front[-1]] / j[front[front.size // 2]])
    assert ratios["L1"] == pytest.approx((1.0, 1.0), abs=1e-9)
    assert min(ratios["L2"]) > 1.3


def test_potential_field_bounds():
    g = Geometry.default()
    field = build_field(g, "L2")
    fdm = field.fdm
    current = np.full(field.mesh.size, 1e-3)                       # 요소마다 1 mA
    surface = field.resistance(10.0) @ current                     # 양극 표면 평균 전위
    phi = fdm.node_potential(current, 10.0)
    assert np.nanmin(phi) >= -1e-9                                  # 최대 원리: 음극(0)과 양극 사이
    assert np.nanmax(phi) == pytest.approx(surface.max(), rel=0.02)
    cathode_nodes = np.zeros(phi.shape, dtype=bool)
    cathode_nodes[:-1, :-1] |= fdm.cells == CATHODE
    assert np.all(phi[cathode_nodes & np.isfinite(phi)] == 0.0)
    image, extent = field.potential_image(current, 10.0)
    assert extent == pytest.approx((0.0, g.tank.width, 0.0, g.tank.liquid_level))
    assert image.shape[1] == 400 and image.shape[0] == pytest.approx(300, abs=1)
    assert np.isnan(image).any() and np.nanmax(image) <= np.nanmax(phi) + 1e-9
    zc = (np.arange(image.shape[0]) + 0.5) / image.shape[0] * extent[3]
    xc = (np.arange(image.shape[1]) + 0.5) / image.shape[1] * extent[1]
    inside_anode = (np.abs(xc - g.anode.x_center) < 0.4 * g.anode.thickness)
    rows = (zc > g.anode.z_bottom + 0.01) & (zc < g.anode.z_top - 0.01)
    assert np.all(np.isnan(image[np.ix_(rows, inside_anode)]))      # 전극 안은 투명
    assert np.all(np.isfinite(image[:, :5]))                         # 벽 옆 전해질은 값이 있다


def test_l1_l0_fields_have_no_potential():
    for level in ("L0", "L1"):
        field = build_field(Geometry.default(), level)
        assert not field.has_potential
        assert field.potential_image(np.ones(field.mesh.size), 1.0) is None


def test_field_cache_reuses_l2_solution():
    g = Geometry.default()
    assert build_field(g, "L2") is build_field(g, "L2")
    moved = replace(g, anode=replace(g.anode, x_center=0.21))
    assert build_field(moved, "L2") is not build_field(g, "L2")


def test_anode_without_electrolyte_contact_is_rejected():
    g = Geometry.default()
    on_floor = replace(g, anode=replace(g.anode, z_bottom=0.0))     # 바닥면이 바닥에 붙음
    assert any("1 mm" in e for e in on_floor.validate())
    with pytest.raises(ValueError, match="닿지 않는"):
        solve_fdm(on_floor, build_surface_mesh(on_floor, 60))


@pytest.mark.parametrize("change, message", [
    (lambda g: replace(g, anode=replace(g.anode, z_bottom=0.0005)), "1 mm"),
    (lambda g: replace(g, anode=replace(g.anode, x_center=0.0015)), "1 mm"),
    (lambda g: replace(g, anode=replace(g.anode, x_center=0.0817)), "겹칩니다"),   # 음극 1 과 0.5 mm
])
def test_clearance_validation(change, message):
    errors = change(Geometry.default()).validate()
    assert any(message in e for e in errors), errors


def test_cells_classify_electrodes():
    g = Geometry.default()
    fdm = build_field(g, "L2").fdm
    xc = 0.5 * (fdm.x[:-1] + fdm.x[1:])
    zc = 0.5 * (fdm.z[:-1] + fdm.z[1:])
    j = np.searchsorted(zc, 0.15)
    i_gap = np.searchsorted(xc, 0.14)
    assert fdm.cells[j, i_gap] == ELECTROLYTE
    i_cathode = np.argmin(np.abs(xc - g.cathodes[0].x_center))
    assert fdm.cells[j, i_cathode] == CATHODE
    assert fdm.contact == pytest.approx(build_surface_mesh(g, 120).length)   # 모든 요소가 전해질에 닿음
