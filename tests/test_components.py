"""고전계 법칙, 음극 과전압, 형상, 전원 모델."""

import math
from dataclasses import replace

import numpy as np
import pytest

from anodsim.core.cathode import CathodeMaterial
from anodsim.core.constants import LN10
from anodsim.core.geometry import Geometry, PlateElectrode
from anodsim.core.growth.high_field import HighFieldLaw
from anodsim.core.power_supply import PowerMode, PowerSetting, PowerSupply

LAW = HighFieldLaw.from_anodizing_ratio(B=33e-9, ratio=1.3e-9, j_ref=10.0)


# ── 고전계 법칙 ──────────────────────────────────────────────────────────────
def test_high_field_inverse_consistency():
    j = np.logspace(-6, 4, 21)
    delta = 20e-9
    assert np.allclose(LAW.current_density(LAW.film_voltage(j, delta), delta), j, rtol=1e-10)


def test_high_field_anodizing_ratio_at_reference():
    assert LAW.anodizing_ratio(10.0) == pytest.approx(1.3e-9, rel=1e-12)
    assert LAW.anodizing_ratio(1.0) > LAW.anodizing_ratio(10.0)   # 전류가 작을수록 nm/V 가 커진다


def test_high_field_derivative_matches_finite_difference():
    j, delta, h = 37.0, 15e-9, 1e-6
    fd = (LAW.film_voltage(j * math.exp(h), delta) - LAW.film_voltage(j * math.exp(-h), delta)) / (2 * h)
    assert LAW.d_film_voltage_d_lnj(j, delta) == pytest.approx(fd, rel=1e-6)


def test_high_field_zero_current():
    assert LAW.film_voltage(0.0, 10e-9) == 0.0
    assert LAW.current_density(0.0, 10e-9) == 0.0


# ── 음극 ────────────────────────────────────────────────────────────────────
CATHODE = CathodeMaterial("x", "x", exchange_current_density=1e-2, tafel_slope=0.12)


def test_cathode_tafel_asymptote():
    area, current = 0.1, 10.0                      # j_c = 100 A/m² ≫ j0
    tafel = 0.12 * math.log10(current / area / 1e-2)
    assert CATHODE.overpotential(current, area) == pytest.approx(tafel, rel=1e-6)
    assert CATHODE.overpotential(0.0, area) == 0.0


def test_cathode_derivative():
    area, current, h = 0.1, 3.0, 1e-6
    fd = (CATHODE.overpotential(current + h, area) - CATHODE.overpotential(current - h, area)) / (2 * h)
    assert CATHODE.d_overpotential(current, area) == pytest.approx(fd, rel=1e-6)
    assert CATHODE.d_overpotential(1e6, area) == pytest.approx(0.12 / LN10 / 1e6, rel=1e-6)


def test_cathode_compatibility(presets):
    nt = presets.electrolyte("ti_nanotube_eg")
    assert not presets.cathode("sus316").is_compatible(nt.tags)
    assert presets.cathode("pt").is_compatible(nt.tags)


# ── 형상 ────────────────────────────────────────────────────────────────────
def test_default_geometry_is_valid():
    g = Geometry.default()
    assert g.validate() == []
    a = g.anode
    expected = 2 * a.height * a.width + 2 * a.height * a.thickness + a.thickness * a.width
    assert g.anode_area == pytest.approx(expected)


def test_partially_immersed_electrode():
    plate = PlateElectrode(x_center=0.1, z_bottom=0.2, height=0.2, thickness=0.002, width=0.1)
    assert plate.immersed_height(0.3) == pytest.approx(0.1)
    assert plate.immersed_height(0.1) == 0.0
    assert plate.wetted_area(0.1) == 0.0


@pytest.mark.parametrize("change, message", [
    (lambda g: replace(g, anode=replace(g.anode, x_center=0.3995)), "전해조 밖"),
    (lambda g: replace(g, anode=replace(g.anode, z_bottom=0.31)), "잠기지 않았"),
    (lambda g: replace(g, anode=replace(g.anode, x_center=0.081)), "겹칩니다"),
    (lambda g: replace(g, cathodes=()), "음극이 최소"),
    (lambda g: replace(g, tank=replace(g.tank, liquid_level=0.5)), "액면"),
])
def test_geometry_validation(change, message):
    errors = change(Geometry.default()).validate()
    assert any(message in e for e in errors), errors


def test_lumped_resistance_trends():
    g = Geometry.default()
    r = g.lumped_resistance(50.0)
    assert r > 0
    assert g.lumped_resistance(100.0) == pytest.approx(r / 2)          # R ∝ 1/κ
    closer = replace(g, cathodes=tuple(replace(c, x_center=0.2 + 0.5 * (c.x_center - 0.2)) for c in g.cathodes))
    assert closer.lumped_resistance(50.0) < r                           # 가까울수록 작다
    one_side = replace(g, cathodes=g.cathodes[:1])
    assert one_side.lumped_resistance(50.0) > r                         # 한쪽 음극만 있으면 크다


# ── 전원 ────────────────────────────────────────────────────────────────────
def test_power_supply_ramps():
    cv = PowerSupply(PowerSetting(PowerMode.CV, 20.0, 5.0, ramp_time=10.0))
    assert cv.setpoint(5.0).voltage_limit == pytest.approx(10.0)
    assert cv.setpoint(5.0).current_limit == 5.0
    assert cv.setpoint(50.0).voltage_limit == 20.0
    cc = PowerSupply(PowerSetting(PowerMode.CC, 20.0, 5.0, ramp_time=10.0))
    assert cc.setpoint(5.0).current_limit == pytest.approx(2.5)
    assert cc.setpoint(5.0).voltage_limit == 20.0
    no_ramp = PowerSupply(PowerSetting(PowerMode.CCCV, 20.0, 5.0))
    assert no_ramp.setpoint(0.0).current_limit == 5.0


def test_power_setting_validation():
    assert PowerSetting(PowerMode.CV, 10.0, math.inf).validate() == []
    assert len(PowerSetting(PowerMode.CV, 0.0, -1.0, -1.0).validate()) == 3
