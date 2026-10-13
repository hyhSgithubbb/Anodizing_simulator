"""치밀(배리어)형 성장: Ti 착색 아노다이징, Al 붕산 배리어형.

    dδ_b/dt = η·k_F·j      (형성된 산화물 전부가 치밀층이 된다)
"""

from __future__ import annotations

import numpy as np

from .base import GrowthModel


class CompactGrowth(GrowthModel):
    regime = "compact"

    def barrier_rate(self, j, film_voltage, barrier):
        return self.formation_rate(j)

    def barrier_rate_partials(self, j, film_voltage, barrier):
        shape = np.shape(barrier)
        return np.full(shape, self.eta * self.k_F), np.zeros(shape)

    def porous_rate(self, pore_formation, porous):
        return np.zeros_like(np.asarray(porous, float))
