"""셀 회로 해석기 (설계서 §1.13-4: 정전류 제약)."""

import math

import numpy as np
import pytest

from anodsim.core import units as u
from anodsim.core.circuit import CircuitSolver
from anodsim.core.growth import make_growth_model
from anodsim.core.power_supply import Setpoint


@pytest.fixture
def setup(presets):
    metal = presets.metal("Al")
    props = presets.electrolyte("al_sulfuric_typeII").properties(u.celsius_to_kelvin(20.0))
    model = make_growth_model(metal, props)
    return model, presets.cathode("pb")


def solver_n(setup, n, resistance=None, areas=None):
    model, cathode = setup
    areas = np.full(n, 0.03 / n) if areas is None else areas
    if resistance is None:
        rng = np.random.default_rng(1)
        m = rng.uniform(0.0, 0.02, (n, n))
        resistance = 0.05 * np.eye(n) + 0.5 * (m + m.T)     # 대칭·양의 정부호
    return model, CircuitSolver(areas, resistance, cathode, cathode_area=0.15, external_resistance=0.01)


def balance(model, solver, barrier, op):
    """요소별 전압 수지 잔차 [V]."""
    i_el = solver.areas * op.current_density
    total = i_el.sum()
    series = solver.external_resistance * total + solver.cathode.overpotential(total, solver.cathode_area)
    return solver.offset_voltage + model.film_voltage(op.current_density, barrier) + solver.resistance @ i_el + series - op.voltage


def test_cv_voltage_balance(setup):
    model, solver = solver_n(setup, 1)
    barrier = np.array([15e-9])
    op = solver.solve(model, barrier, Setpoint(15.0, math.inf))
    assert op.control == "CV"
    assert np.max(np.abs(balance(model, solver, barrier, op))) < 1e-8
    assert op.current == pytest.approx(float(solver.areas @ op.current_density))


@pytest.mark.parametrize("n", [1, 5])
def test_cc_total_current(setup, n):
    model, solver = solver_n(setup, n)
    barrier = np.linspace(10e-9, 20e-9, n)
    target = 2.5
    op = solver.solve(model, barrier, Setpoint(1000.0, target))
    assert op.control == "CC"
    assert op.current == pytest.approx(target, rel=1e-8)
    assert np.max(np.abs(balance(model, solver, barrier, op))) < 1e-8


def test_cc_cv_switching_is_unique(setup):
    model, solver = solver_n(setup, 3)
    barrier = np.array([12e-9, 15e-9, 18e-9])
    for sp in (Setpoint(15.0, 1.0), Setpoint(15.0, 100.0)):
        a = solver.solve(model, barrier, sp, prefer="CV")
        b = solver.solve(model, barrier, sp, prefer="CC")
        assert a.control == b.control
        assert a.voltage == pytest.approx(b.voltage, rel=1e-9)
        assert np.allclose(a.current_density, b.current_density, rtol=1e-8)
        assert a.voltage <= sp.voltage_limit + 1e-9 and a.current <= sp.current_limit * (1 + 1e-9)


def test_off_state(setup):
    model, solver = solver_n(setup, 1)
    op = solver.solve(model, np.array([3e-9]), Setpoint(15.0, 0.0))
    assert op.control == "OFF" and op.current == 0.0


def test_warm_start_from_zero_current(setup):
    """회귀: 전류 0(OFF) 상태의 해를 warm start 로 써도 수렴해야 한다."""
    model, solver = solver_n(setup, 1)
    barrier = np.array([3e-9])
    off = solver.solve(model, barrier, Setpoint(15.0, 0.0))
    op = solver.solve(model, barrier, Setpoint(5.0, math.inf), guess=off.log_j)
    assert op.current > 0 and np.max(np.abs(balance(model, solver, barrier, op))) < 1e-8


def test_symmetric_layout_gives_symmetric_current(setup):
    n = 4
    r = np.array([[4, 1, .5, .2], [1, 4, 1, .5], [.5, 1, 4, 1], [.2, .5, 1, 4]]) * 0.01
    model, solver = solver_n(setup, n, resistance=r)
    barrier = np.array([14e-9, 16e-9, 16e-9, 14e-9])
    op = solver.solve(model, barrier, Setpoint(15.0, math.inf))
    assert np.allclose(op.current_density, op.current_density[::-1], rtol=1e-9)


@pytest.mark.parametrize("setpoint", [Setpoint(15.0, math.inf), Setpoint(1000.0, 2.0)])
def test_barrier_sensitivity_matches_finite_difference(setup, setpoint):
    model, solver = solver_n(setup, 3)
    barrier = np.array([12e-9, 15e-9, 18e-9])
    op = solver.solve(model, barrier, setpoint)
    dj, d_uf = solver.barrier_sensitivity(model, barrier, op)
    h = 1e-13
    for k in range(3):
        plus, minus = barrier.copy(), barrier.copy()
        plus[k] += h
        minus[k] -= h
        op_p, op_m = solver.solve(model, plus, setpoint), solver.solve(model, minus, setpoint)
        assert op_p.control == op.control
        fd_j = (op_p.current_density - op_m.current_density) / (2 * h)
        fd_u = (op_p.film_voltage - op_m.film_voltage) / (2 * h)
        assert np.allclose(dj[:, k], fd_j, rtol=1e-4, atol=1e-6 * np.max(np.abs(fd_j)))
        assert np.allclose(d_uf[:, k], fd_u, rtol=1e-4, atol=1e-6 * np.max(np.abs(fd_u)))
