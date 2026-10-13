"""시간 적분 엔진.

매 스텝 전류 분포는 대수적으로(준정적) 풀고, 두께만 시간 적분한다.

- 배리어 두께는 고전계 법칙 때문에 매우 stiff 하다 (정상상태 부근 이완 시간 ~0.1 s).
  회로 민감도 dj/dδ 로 야코비안 J_b = ∂f_b/∂δ 를 만들어 선형 암시적 오일러로 푼다:
      Δδ = (I − Δt·J_b)⁻¹·Δt·f_b
  이는 스텝 끝의 선형화 전류 j_lin = j + (dj/dδ)·Δδ 로 f_b 를 평가한 것과 같다.
- 누적 전하·기공층·금속 후퇴도 같은 j_lin 으로 적분한다. 그래서 전하 보존
  (η·k_F·ΔQ = 배리어 증가 + 기공층 증가의 치밀 부피)이 스텝마다 정확히 성립한다.
- 기공층은 회로에 영향을 주지 않으므로 새 배리어에서 푼 동작점이 곧 새 상태의 동작점이다
  (스텝당 회로 해석 1회).

스텝 크기는 배리어 두께 상대 변화(0.5 %), 전류 상대 변화(5 %), 램프 해상도로 제한한다.

모든 계산은 양극 표면 요소에 대한 길이 N 배열이다. 해석 단계(L0: N = 1, L1: 측면도 둘레 N 요소)는
`core.field` 가 만드는 요소 면적과 저항 행렬 R 만 바꾼다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import limits
from .circuit import CircuitSolver
from .field import Field, SurfaceMesh, build_field
from .growth import GrowthRates, make_growth_model
from .power_supply import PowerSupply
from .presets import PresetDB
from .state import ProcessConfig, SimState, Snapshot

MIN_STEP = 1e-9  # s
PROFILE_SAMPLES = 400   # 위치별 이력 h(s, t) 를 기록하는 최대 시각 수 (대략)


class ConfigError(ValueError):
    def __init__(self, errors):
        super().__init__("\n".join(errors))
        self.errors = list(errors)


@dataclass(frozen=True)
class SimulationResult:
    """시간 이력.

    스칼라 시계열은 매 스텝 기록하며, 표면 요소에 대해서는 면적 가중 평균이다 (두께는 최소/최대도 기록).
    위치별 이력 profile_* 은 대략 PROFILE_SAMPLES 개 시각에서 요소별 값을 기록한다 (행 = 시각, 열 = 요소).
    """

    time: np.ndarray                    # s
    voltage: np.ndarray                 # V
    current: np.ndarray                 # A
    control: tuple[str, ...]
    current_density: np.ndarray         # A/m²
    film_voltage: np.ndarray            # V
    cathode_overpotential: np.ndarray   # V
    barrier: np.ndarray                 # m
    porous: np.ndarray                  # m
    total: np.ndarray                   # m
    total_min: np.ndarray               # m
    total_max: np.ndarray               # m
    charge: np.ndarray                  # C/m²
    final: Snapshot
    flags: frozenset[str]
    warnings: tuple[str, ...]
    steps: int
    mesh: SurfaceMesh
    profile_time: np.ndarray            # s, (K,)
    profile_total: np.ndarray           # m, (K, N)
    profile_current_density: np.ndarray  # A/m², (K, N)
    wagner: float                       # 최종 시각의 면적 가중 Wagner 수 (전류 0 이면 nan)

    @property
    def uniformity(self) -> float:
        """최종 두께 편차 (최대 − 최소)/평균. 요소 1개이면 0."""
        h = self.final.total
        mean = float(self.mesh.area @ h / self.mesh.area.sum())
        return float((h.max() - h.min()) / mean) if mean > 0 else 0.0


class _History:
    _FIELDS = ("time", "voltage", "current", "control", "current_density", "film_voltage",
               "cathode_overpotential", "barrier", "porous", "total", "total_min", "total_max", "charge")

    def __init__(self, areas: np.ndarray, duration: float):
        self.weights = areas / areas.sum()
        self.rows = {name: [] for name in self._FIELDS}
        self.profile_interval = duration / PROFILE_SAMPLES
        self.profile = {"time": [], "total": [], "current_density": []}

    def add(self, s: Snapshot, force_profile: bool = False) -> None:
        w = self.weights
        r = self.rows
        p = self.profile
        if force_profile or not p["time"] or s.time - p["time"][-1] >= self.profile_interval:
            self.add_profile(s)
        r["time"].append(s.time)
        r["voltage"].append(s.voltage)
        r["current"].append(s.current)
        r["control"].append(s.control)
        r["current_density"].append(float(w @ s.current_density))
        r["film_voltage"].append(float(w @ s.film_voltage))
        r["cathode_overpotential"].append(s.cathode_overpotential)
        r["barrier"].append(float(w @ s.barrier))
        r["porous"].append(float(w @ s.porous))
        r["total"].append(float(w @ s.total))
        r["total_min"].append(float(s.total.min()))
        r["total_max"].append(float(s.total.max()))
        r["charge"].append(float(w @ s.charge))

    def result(self, final: Snapshot, flags, warnings, steps: int, mesh: SurfaceMesh,
               wagner: float) -> SimulationResult:
        if self.profile["time"][-1] != final.time:   # 마지막 시각은 항상 기록
            self.add_profile(final)
        arrays = {k: np.asarray(v, dtype=float) for k, v in self.rows.items() if k != "control"}
        p = self.profile
        return SimulationResult(control=tuple(self.rows["control"]), final=final, flags=frozenset(flags),
                                warnings=tuple(warnings), steps=steps, mesh=mesh, wagner=wagner,
                                profile_time=np.asarray(p["time"], dtype=float),
                                profile_total=np.vstack(p["total"]),
                                profile_current_density=np.vstack(p["current_density"]), **arrays)

    def add_profile(self, s: Snapshot) -> None:
        self.profile["time"].append(s.time)
        self.profile["total"].append(s.total)
        self.profile["current_density"].append(s.current_density)


class Simulator:
    def __init__(self, config: ProcessConfig, presets: PresetDB | None = None, *,
                 max_step: float | None = None, barrier_tolerance: float = 0.005,
                 current_tolerance: float = 0.05):
        presets = presets or PresetDB.default()
        errors = config.validate(presets)
        if errors:
            raise ConfigError(errors)
        self.config = config
        self.metal = presets.metal(config.metal)
        self.preset = presets.electrolyte(config.electrolyte.preset_id)
        self.cathode = presets.cathode(config.cathode_material)
        self.props = self.preset.properties(config.bath_temperature, config.electrolyte.concentration)
        self.model = make_growth_model(self.metal, self.props)
        self.supply = PowerSupply(config.power)

        geo = config.geometry
        self.field: Field = build_field(geo, config.fidelity, config.elements)
        self.mesh = self.field.mesh
        self.areas = self.mesh.area                                       # m², (N,)
        self.resistance = self.field.resistance(self.props.conductivity)  # Ω, (N, N)
        self.circuit = CircuitSolver(self.areas, self.resistance, self.cathode, geo.cathode_area,
                                     config.offset_voltage, config.external_resistance)
        self.warnings = tuple(limits.range_warnings(config, self.preset, self.cathode))
        self.max_step = max_step if max_step is not None else min(config.duration / 1000.0, 10.0)
        self.barrier_tolerance = barrier_tolerance
        self.current_tolerance = current_tolerance
        self.reset()

    # ── 상태 ────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        n = self.areas.size
        self.state = SimState(time=0.0,
                              barrier=np.full(n, self.metal.native_oxide_thickness),
                              porous=np.zeros(n), metal_loss=np.zeros(n), charge=np.zeros(n))
        self.steps = 0
        self.flags_seen: set[str] = set()
        s = self.state
        self._set_operating_point(self.circuit.solve(self.model, s.barrier, self.supply.setpoint(s.time)))

    @property
    def done(self) -> bool:
        return self.state.time >= self.config.duration

    def snapshot(self) -> Snapshot:
        s, op = self.state, self._op
        return Snapshot(
            time=s.time, voltage=op.voltage, current=op.current, control=op.control,
            cathode_overpotential=op.cathode_overpotential,
            current_density=op.current_density.copy(), film_voltage=op.film_voltage.copy(),
            barrier=s.barrier.copy(), porous=s.porous.copy(), total=s.total.copy(),
            metal_loss=s.metal_loss.copy(), charge=s.charge.copy(), flags=self._flags,
        )

    # ── 적분 ────────────────────────────────────────────────────────────────
    def step(self) -> Snapshot:
        if self.done:
            return self.snapshot()
        s, op0, r0 = self.state, self._op, self._rates
        dj, d_uf = self.circuit.barrier_sensitivity(self.model, s.barrier, op0)
        dt = self._step_size(s, r0, op0, dj)
        t1 = s.time + dt
        if self.config.duration - t1 < MIN_STEP:
            t1 = self.config.duration

        # 1) 배리어: 선형 암시적 오일러
        m = self.model
        f_j, f_uf = m.barrier_rate_partials(op0.current_density, op0.film_voltage, s.barrier)
        jac = f_j[:, np.newaxis] * dj + f_uf[:, np.newaxis] * d_uf      # J_b = ∂f_b/∂δ
        step = np.linalg.solve(np.eye(jac.shape[0]) - dt * jac, dt * r0.barrier)
        barrier1 = np.maximum(s.barrier + step, m.min_barrier)

        # 2) 전하·기공층·금속: 배리어 갱신과 같은 선형화 전류 j_lin 으로 (전하 보존)
        j_lin = np.maximum(op0.current_density + dj @ (barrier1 - s.barrier), 0.0)
        barrier_rate = (barrier1 - s.barrier) / dt
        pores = m.pore_formation(m.formation_rate(j_lin), barrier_rate)
        self.state = SimState(
            time=t1,
            barrier=barrier1,
            porous=np.maximum(s.porous + dt * m.porous_rate(pores, s.porous), 0.0),
            metal_loss=s.metal_loss + dt * m.metal_rate(j_lin),
            charge=s.charge + dt * j_lin,
        )

        # 3) 새 상태의 동작점 (기공층은 회로에 영향 없음)
        op1 = self.circuit.solve(m, barrier1, self.supply.setpoint(t1), op0.log_j, op0.control)
        self.steps += 1
        self._set_operating_point(op1)
        return self.snapshot()

    def run(self, callback=None) -> SimulationResult:
        """끝까지 적분한다. callback(snapshot) 이 False 를 돌려주면 중단한다."""
        history = _History(self.areas, self.config.duration)
        history.add(self.snapshot())
        while not self.done:
            snap = self.step()
            history.add(snap)
            if callback is not None and callback(snap) is False:
                break
        return history.result(self.snapshot(), self.flags_seen, self.warnings, self.steps, self.mesh,
                              self.wagner_number())

    def wagner_number(self) -> float:
        """면적 가중 Wagner 수 Wa = (dU_f/dj) / (R_ii·a_i) — 막 분극 저항 / 전해질 저항.

        Wa ≫ 1 이면 막이 분포를 지배해 균일하고, Wa ≪ 1 이면 형상(전해질 저항)이 지배해 편차가 크다.
        """
        op, s = self._op, self.state
        j = op.current_density
        if op.control == "OFF" or not np.all(j > 0):
            return math.nan
        polarization = self.model.d_film_voltage_d_lnj(j, s.barrier) / j          # Ω·m²
        electrolyte = np.diag(self.resistance) * self.areas                      # Ω·m²
        return float(self.areas @ (polarization / electrolyte) / self.areas.sum())

    # ── 내부 ────────────────────────────────────────────────────────────────
    def _set_operating_point(self, op) -> None:
        s = self.state
        self._op = op
        self._rates = self.model.rates(op.current_density, op.film_voltage, s.barrier, s.porous)
        self._flags = limits.check(op, self.props, s.porous, self._rates)
        self.flags_seen |= self._flags

    def _step_size(self, s: SimState, r: GrowthRates, op, dj: np.ndarray) -> float:
        dt = self.max_step
        rate = np.abs(r.barrier)
        active = rate > 0.0
        if np.any(active):
            dt = min(dt, self.barrier_tolerance * float(np.min(s.barrier[active] / rate[active])))
        # 전류의 상대 변화 (돌입·감쇠 구간 정확도). dj/dt ≈ (dj/dδ)·f_b
        j = op.current_density
        dj_dt = np.abs(dj @ r.barrier)
        flowing = (j > 0.0) & (dj_dt > 0.0)
        if np.any(flowing):
            dt = min(dt, self.current_tolerance * float(np.min(j[flowing] / dj_dt[flowing])))
        if self.props.pore_wall_dissolution > 0.0:
            dt = min(dt, 0.05 / self.props.pore_wall_dissolution)
        ramp = self.config.power.ramp_time
        if s.time < ramp:
            dt = min(dt, ramp / 100.0, ramp - s.time)   # 램프 끝(꺾이는 점)을 건너뛰지 않는다
        dt = max(dt, MIN_STEP)
        return min(dt, self.config.duration - s.time)
