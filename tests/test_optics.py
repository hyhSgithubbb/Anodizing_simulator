"""Ti 간섭색 (core/optics.py)."""

import colorsys

import numpy as np
import pytest

from anodsim.core import optics
from anodsim.core import units as u


def test_white_reflector_is_d65_white():
    flat = np.ones(optics.WAVELENGTHS.size)
    xyz = optics.spectrum_to_xyz(flat)
    assert xyz[1] == pytest.approx(1.0)
    assert xyz[0] / xyz.sum() == pytest.approx(0.3127, abs=0.002)    # D65 백색점
    assert xyz[1] / xyz.sum() == pytest.approx(0.3290, abs=0.002)
    assert np.allclose(optics.xyz_to_srgb(xyz), 1.0, atol=2e-3)


def test_bare_titanium_is_neutral_grey():
    rgb = optics.interference_color(0.0)[0]
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    assert s < 0.1 and 0.6 < v < 0.9
    r = optics.reflectance(0.0)[0]
    assert 0.45 < r[np.searchsorted(optics.WAVELENGTHS, 550.0)] < 0.65      # Ti 반사율 ≈ 0.55
    assert optics.color_name(rgb) == "은회색"


def test_first_reflectance_minimum_moves_to_red_with_thickness():
    minima = [optics.WAVELENGTHS[np.argmin(optics.reflectance(d * u.NM)[0])] for d in (30, 40, 50, 60)]
    assert minima == sorted(minima) and minima[-1] > minima[0] + 150.0


@pytest.mark.parametrize("thickness_nm, check", [
    (25, lambda r, g, b: r > b and g > b),                # 금색·청동 (파랑 흡수)
    (50, lambda r, g, b: b > r and b > g),                # 파랑
    (120, lambda r, g, b: r > b and g > b),               # 2차 금색
    (150, lambda r, g, b: r > g and b > g),               # 보라·자홍
])
def test_colour_sequence_matches_ti_anodizing_chart(thickness_nm, check):
    r, g, b = optics.interference_color(thickness_nm * u.NM)[0]
    assert check(r, g, b), (thickness_nm, optics.to_hex((r, g, b)))


def test_vectorised_and_helpers():
    colours = optics.interference_color(np.array([0.0, 50e-9, 120e-9]))
    assert colours.shape == (3, 3) and np.all((colours >= 0) & (colours <= 1))
    assert optics.to_hex((1.0, 0.5, 0.0)) == "#ff8000"
    assert optics.color_name(optics.interference_color(50e-9)[0]) == "파랑"
