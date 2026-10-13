import pytest

from anodsim.core import units as u

UM_PER_C_CM2 = u.A_PER_CM2 / u.UM   # m³/C → μm per (C/cm²)


def test_faraday_growth_coefficients(presets):
    """설계서 §1.13-1: Al2O3 0.550, TiO2 0.545 μm per C/cm² (η=1, 치밀막)."""
    assert presets.metal("Al").oxide.growth_coefficient * UM_PER_C_CM2 == pytest.approx(0.550, rel=2e-3)
    assert presets.metal("Ti").oxide.growth_coefficient * UM_PER_C_CM2 == pytest.approx(0.545, rel=2e-3)


def test_metal_recession_coefficients(presets):
    assert presets.metal("Al").recession_coefficient * UM_PER_C_CM2 == pytest.approx(0.345, rel=2e-3)
    assert presets.metal("Ti").recession_coefficient * UM_PER_C_CM2 == pytest.approx(0.275, rel=2e-3)


def test_expansion_ratio(presets):
    assert presets.metal("Al").expansion_ratio == pytest.approx(1.59, rel=1e-2)
    assert presets.metal("Ti").expansion_ratio == pytest.approx(1.98, rel=1e-2)
