"""셀 회로 해석: 표면 요소별 전류밀도 분포와 동작점(전압·전류)을 구한다.

요소 i 의 전압 수지 (설계서 §1.6–1.7):

    U_app = U_off + U_f,i(j_i, δ_i) + Σ_k R_ik·a_k·j_k + (R_ext + |η_c|)(I)

    I = Σ_k a_k·j_k,  R = 전해질 저항 행렬 [Ω] (L0 에서는 1×1)

미지수를 x_i = ln j_i 로 잡은 Newton 법(감쇠 + backtracking)으로 푼다.
정류기는 전압 제한과 전류 제한 중 먼저 걸리는 쪽으로 동작한다 (CV 해 → 전류 초과 시 CC 해).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .power_supply import Setpoint

J_FLOOR = 1e-12     # A/m² — 사실상 0 전류
J_CEIL = 1e8        # A/m²
_MAX_LOG_STEP = 2.0  # Newton 한 번에 j 가 최대 e² 배까지만 변하도록 제한


class CircuitConvergenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class OperatingPoint:
    voltage: float                  # U_app [V]
    current: float                  # I [A]
    current_density: np.ndarray     # j [A/m²]
    film_voltage: np.ndarray        # U_f [V]
    cathode_overpotential: float    # |η_c| [V]
    control: str                    # "CV" | "CC" | "OFF"
    log_j: np.ndarray               # 다음 스텝 Newton 초기값


class CircuitSolver:
    def __init__(self, areas, resistance, cathode, cathode_area: float,
                 offset_voltage: float = 0.0, external_resistance: float = 0.0,
                 tol: float = 1e-9, max_iter: int = 200):
        self.areas = np.asarray(areas, dtype=float).ravel()
        n = self.areas.size
        self.resistance = np.atleast_2d(np.asarray(resistance, dtype=float))
        if self.resistance.shape != (n, n):
            raise ValueError(f"resistance matrix must be {n}×{n}, got {self.resistance.shape}")
        if np.any(self.areas <= 0) or cathode_area <= 0:
            raise ValueError("electrode areas must be positive")
        self.cathode = cathode
        self.cathode_area = cathode_area
        self.offset_voltage = offset_voltage
        self.external_resistance = external_resistance
        self.tol = tol
        self.max_iter = max_iter
        self._log_floor = math.log(J_FLOOR)
        self._log_ceil = math.log(J_CEIL)
        self._diag = np.arange(n)

    # ── 공개 API ────────────────────────────────────────────────────────────
    def solve(self, model, barrier, setpoint: Setpoint, guess=None, prefer: str = "CV") -> OperatingPoint:
        """전압·전류 제한 중 먼저 걸리는 쪽의 동작점. I(U) 가 단조 증가하므로 결과는 유일하다.

        prefer: 먼저 시도할 제어 모드 (직전 스텝의 모드를 주면 대부분 한 번에 끝난다).
        """
        if setpoint.current_limit <= 0.0 or setpoint.voltage_limit <= self.offset_voltage:
            return self._off(model, barrier, setpoint)
        if prefer == "CC" and math.isfinite(setpoint.current_limit):
            cc = self.solve_current(model, barrier, setpoint.current_limit, guess)
            if cc.voltage <= setpoint.voltage_limit:
                return cc
            return self.solve_voltage(model, barrier, setpoint.voltage_limit, cc.log_j)
        cv = self.solve_voltage(model, barrier, setpoint.voltage_limit, guess)
        if cv.current <= setpoint.current_limit:
            return cv
        cc_guess = cv.log_j - math.log(cv.current / setpoint.current_limit)
        return self.solve_current(model, barrier, setpoint.current_limit, cc_guess)

    def solve_voltage(self, model, barrier, voltage: float, guess=None) -> OperatingPoint:
        """정전압: U_app 고정, 요소별 j 를 구한다."""
        barrier = np.asarray(barrier, dtype=float)
        if guess is None:
            x = self._initial_guess(model, barrier, voltage)
            g, jac, j = self._residual(model, barrier, x, voltage)
        else:
            # 직전 스텝 해(warm start)에서 출발하되, 많이 벗어나 있으면 해석적 상한 추정과 비교한다
            x = self._clip(np.asarray(guess, dtype=float))
            g, jac, j = self._residual(model, barrier, x, voltage)
            if np.max(np.abs(g)) > 0.05 * max(voltage, 1.0):
                xu = self._initial_guess(model, barrier, voltage)
                gu, jacu, ju = self._residual(model, barrier, xu, voltage)
                if np.max(np.abs(gu)) < np.max(np.abs(g)):
                    x, g, jac, j = xu, gu, jacu, ju
        for _ in range(self.max_iter):
            norm = np.max(np.abs(g))
            if norm < self.tol:
                break
            if np.all(g > 0) and np.all(x <= self._log_floor):
                break  # 구동 전압이 거의 0 — 전류 하한에서 멈춤
            dx = np.linalg.solve(jac, -g)
            dx *= min(1.0, _MAX_LOG_STEP / max(np.max(np.abs(dx)), 1e-300))
            x, g, jac, j = self._line_search(model, barrier, x, dx, norm,
                                             lambda xn: self._residual(model, barrier, xn, voltage))
        else:
            raise CircuitConvergenceError(f"CV solve did not converge (|G| = {norm:.3e} V)")
        return self._operating_point(model, barrier, x, j, voltage, "CV")

    def solve_current(self, model, barrier, current: float, guess=None) -> OperatingPoint:
        """정전류: 총전류 고정, 요소별 j 와 U_app 를 함께 구한다."""
        barrier = np.asarray(barrier, dtype=float)
        n = self.areas.size
        if guess is None:
            x = np.full(n, math.log(current / self.areas.sum()))
        else:
            x = self._clip(np.asarray(guess, dtype=float))
        g, jac, j = self._residual(model, barrier, x, 0.0)
        voltage = float(np.mean(g))   # g = U(x) − 0 이므로 평균값이 좋은 초기 전압
        z = np.append(x, voltage)

        def augmented(zn):
            gn, jn_jac, jn = self._residual(model, barrier, zn[:n], zn[n])
            i_el = self.areas * jn
            f = np.append(gn, (i_el.sum() - current) / current)
            big = np.zeros((n + 1, n + 1))
            big[:n, :n] = jn_jac
            big[:n, n] = -1.0
            big[n, :n] = i_el / current
            return f, big, jn

        f, big, j = augmented(z)
        for _ in range(self.max_iter):
            norm = np.max(np.abs(f))
            if norm < self.tol:
                break
            dz = np.linalg.solve(big, -f)
            dz[:n] *= min(1.0, _MAX_LOG_STEP / max(np.max(np.abs(dz[:n])), 1e-300))
            z, f, big, j = self._line_search(model, barrier, z, dz, norm, augmented, clip_slice=slice(0, n))
        else:
            raise CircuitConvergenceError(f"CC solve did not converge (|F| = {norm:.3e})")
        return self._operating_point(model, barrier, z[:n], j, float(z[n]), "CC")

    def barrier_sensitivity(self, model, barrier, op: OperatingPoint):
        """전원 설정을 고정했을 때 배리어 두께에 대한 민감도 (dj/dδ, dU_f/dδ), 각각 N×N.

        G(x, δ) = 0 을 음함수 미분한다: ∂G/∂δ = diag(U_f/δ) (j 고정 시 U_f ∝ δ).
        시간 적분기가 배리어 식을 선형 암시적으로 풀 때 쓴다 (고전계 법칙 때문에 매우 stiff).
        """
        barrier = np.asarray(barrier, dtype=float)
        n = self.areas.size
        if op.control == "OFF":
            return np.zeros((n, n)), np.zeros((n, n))
        j = op.current_density
        _, jac, _ = self._residual(model, barrier, op.log_j, op.voltage)
        explicit = op.film_voltage / barrier
        rhs = -np.diag(explicit)
        if op.control == "CV":
            dx = np.linalg.solve(jac, rhs)
        else:  # CC: 총전류 고정, U_app 도 함께 변한다
            big = np.zeros((n + 1, n + 1))
            big[:n, :n] = jac
            big[:n, n] = -1.0
            big[n, :n] = self.areas * j
            dx = np.linalg.solve(big, np.vstack([rhs, np.zeros((1, n))]))[:n]
        dj = j[:, np.newaxis] * dx
        d_uf = np.diag(explicit) + model.d_film_voltage_d_lnj(j, barrier)[:, np.newaxis] * dx
        return dj, d_uf

    # ── 내부 ────────────────────────────────────────────────────────────────
    def _clip(self, x):
        return np.clip(x, self._log_floor, self._log_ceil)

    def _initial_guess(self, model, barrier, voltage):
        """저항만 있을 때와 막만 있을 때의 전류 중 작은 값 — 해보다 위에서 시작한다."""
        drive = voltage - self.offset_voltage
        lumped = self.resistance @ self.areas + self.external_resistance * self.areas.sum()
        j_resistive = drive / np.maximum(lumped, 1e-300)
        j_film = model.law.current_density(drive, barrier)
        return self._clip(np.log(np.maximum(np.minimum(j_resistive, j_film), J_FLOOR)))

    def _residual(self, model, barrier, x, voltage):
        j = np.exp(x)
        i_el = self.areas * j
        total = i_el.sum()
        series = self.external_resistance * total + self.cathode.overpotential(total, self.cathode_area)
        g = (self.offset_voltage + model.film_voltage(j, barrier) + self.resistance @ i_el + series - voltage)
        d_series = self.external_resistance + self.cathode.d_overpotential(total, self.cathode_area)
        jac = (self.resistance + d_series) * i_el[np.newaxis, :]
        jac[self._diag, self._diag] += model.d_film_voltage_d_lnj(j, barrier)
        return g, jac, j

    def _line_search(self, model, barrier, z, dz, norm, evaluate, clip_slice=None):
        """잔차가 커지지 않는 범위에서 가장 큰 스텝을 받는다.

        전류가 매우 작은 영역에서는 잔차가 거의 평평하므로 '충분한 감소' 조건 대신
        '증가하지 않음' 조건을 쓴다 (스텝 크기는 _MAX_LOG_STEP 로 이미 제한됨).
        """
        alpha = 1.0
        while True:
            zn = z + alpha * dz
            if clip_slice is None:
                zn = self._clip(zn)
            else:
                zn[clip_slice] = self._clip(zn[clip_slice])
            fn, jac, j = evaluate(zn)
            if np.max(np.abs(fn)) <= norm or alpha < 1.0 / 64.0:
                return zn, fn, jac, j
            alpha *= 0.5

    def _operating_point(self, model, barrier, x, j, voltage, control):
        total = float((self.areas * j).sum())
        return OperatingPoint(
            voltage=float(voltage),
            current=total,
            current_density=j,
            film_voltage=model.film_voltage(j, barrier),
            cathode_overpotential=self.cathode.overpotential(total, self.cathode_area),
            control=control,
            log_j=np.asarray(x, dtype=float).copy(),
        )

    def _off(self, model, barrier, setpoint):
        n = self.areas.size
        # 전압 제한이 오프셋 이하이면 그 전압이 걸린 채 전류 0, 전류 제한이 0 이면 개회로(오프셋) 전압
        voltage = setpoint.voltage_limit if setpoint.current_limit > 0 else self.offset_voltage
        return OperatingPoint(
            voltage=float(voltage), current=0.0, current_density=np.zeros(n), film_voltage=np.zeros(n),
            cathode_overpotential=0.0, control="OFF", log_j=np.full(n, self._log_floor),
        )
