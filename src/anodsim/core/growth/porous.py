"""다공성형 성장: Al2O3 (황산·옥살산·인산), TiO2 나노튜브 (EG + NH4F).

    j_ss(U_f)  = j_ss,ref·exp((U_f − U_ref)/U_s)                    정상상태 분극곡선 (프리셋)
    δ_ss(U_f)  = B·U_f / asinh(j_ss/2A)                              정상 배리어 두께 (출력용)

    j > j_ss :  dδ_b/dt = η·k_F·(j − j_ss)                           초과 전류는 배리어를 두껍게
    j < j_ss :  dδ_b/dt = −min(η·k_F·(j_ss − j), v_thin)             전압을 낮추면 배리어가 얇아짐
    dh_p/dt    = η·k_F·min(j, j_ss)/(1 − P) − (v_cd + k_pw·h_p)·[h_p > 0]

정상상태(j = j_ss)에서는 배리어 두께가 δ_ss 로 유지되고 전하는 모두 기공층 성장에 쓰인다.
초기(배리어가 얇아 j ≫ j_ss)에는 거의 모든 전하가 배리어 형성에 쓰인다.
"""

from __future__ import annotations

import numpy as np

from .base import GrowthModel

_MAX_EXPONENT = 700.0


class PorousGrowth(GrowthModel):
    regime = "porous"

    def steady_current_density(self, film_voltage):
        p = self.props
        arg = np.clip((np.asarray(film_voltage, float) - p.U_ref) / p.U_s, -_MAX_EXPONENT, _MAX_EXPONENT)
        return p.j_ss_ref * np.exp(arg)

    def steady_barrier_thickness(self, film_voltage):
        u = np.maximum(np.asarray(film_voltage, float), 0.0)
        j_ss = self.steady_current_density(u)
        delta = self.law.B * u / np.arcsinh(j_ss / (2.0 * self.law.A))
        return np.maximum(delta, self.min_barrier)

    def barrier_rate(self, j, film_voltage, barrier):
        excess = self.formation_rate(np.asarray(j, float) - self.steady_current_density(film_voltage))
        return np.where(excess >= 0.0, excess, -np.minimum(-excess, self.props.barrier_thinning))

    def barrier_rate_partials(self, j, film_voltage, barrier):
        j_ss = self.steady_current_density(film_voltage)
        excess = self.formation_rate(np.asarray(j, float) - j_ss)
        active = (excess >= 0.0) | (-excess < self.props.barrier_thinning)   # v_thin 상한에 걸리지 않은 구간
        k = self.eta * self.k_F
        return np.where(active, k, 0.0), np.where(active, -k * j_ss / self.props.U_s, 0.0)

    def porous_rate(self, pore_formation, porous):
        p = self.props
        porous = np.asarray(porous, float)
        dissolution = np.where(porous > 0.0, p.surface_dissolution + p.pore_wall_dissolution * porous, 0.0)
        return np.asarray(pore_formation, float) / (1.0 - p.porosity) - dissolution
