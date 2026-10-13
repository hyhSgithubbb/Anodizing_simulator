"""단위 환산 계수.

SI 값 = 표시 값 × 계수.  예) ``thickness_m = 12.0 * UM``, ``thickness_um = thickness_m / UM``
"""

from .constants import ZERO_CELSIUS

# 길이
NM = 1e-9
UM = 1e-6
MM = 1e-3
CM = 1e-2

# 시간
MINUTE = 60.0
HOUR = 3600.0

# 전류밀도 (A/m² 기준)
A_PER_DM2 = 100.0
A_PER_CM2 = 1e4

# 물성
G_PER_MOL = 1e-3      # kg/mol
G_PER_CM3 = 1e3       # kg/m³
KJ_PER_MOL = 1e3      # J/mol

# 속도
UM_PER_HOUR = UM / HOUR    # m/s
NM_PER_SECOND = NM         # m/s
PER_HOUR = 1.0 / HOUR      # 1/s


def celsius_to_kelvin(t_c: float) -> float:
    return t_c + ZERO_CELSIUS


def kelvin_to_celsius(t_k: float) -> float:
    return t_k - ZERO_CELSIUS
