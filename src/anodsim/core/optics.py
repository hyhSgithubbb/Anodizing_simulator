"""Ti 양극산화막(TiO₂)의 간섭색 — 설계서 §1.5 부가 출력.

박막 간섭 (수직 입사, 공기 / TiO₂ 막 / Ti 금속):

    r_ij = (N_i − N_j) / (N_i + N_j),          β = 2π · N₁ · d / λ
    r = (r₀₁ + r₁₂ e^{−2iβ}) / (1 + r₀₁ r₁₂ e^{−2iβ}),   R(λ) = |r|²

- 막: 무정형·아나타제 양극산화 TiO₂, 코시 분산 n(λ) = 2.19 + 0.033/λ²[μm²] (n(550 nm) ≈ 2.30), 흡수 무시
- 기판: Ti 금속 복소 굴절률 N = n − ik (Johnson & Christy 1974 를 매끄럽게 근사한 표)
- 색: D65 조명 아래 반사광 R(λ)·S(λ) 를 CIE 1931 2° 등색함수(Wyman–Sloan–Shirley 2013 해석 근사)로 적분해
  XYZ → sRGB. 완전 반사체가 Y = 1 (흰색)이 되도록 정규화한다.

실제 색은 표면 거칠기, 막 결정상, 보는 각도, 조명에 따라 달라진다. 화면 색은 경향을 보여주는 근사다.
"""

from __future__ import annotations

import colorsys

import numpy as np

WAVELENGTHS = np.arange(380.0, 781.0, 5.0)        # nm

# CIE 표준광 D65 상대 분광 분포 (10 nm 간격, 380–780 nm)
_D65_10NM = np.array([
    49.98, 54.65, 82.75, 91.49, 93.43, 86.68, 104.86, 117.01, 117.81, 114.86, 115.92, 108.81, 109.35,
    107.80, 104.79, 107.69, 104.41, 104.05, 100.00, 96.33, 95.79, 88.69, 90.01, 89.60, 87.70, 83.29,
    83.70, 80.03, 80.21, 82.28, 78.28, 69.72, 71.61, 74.35, 61.60, 69.89, 75.09, 63.59, 46.42, 66.81, 63.38,
])

# Ti 금속 광학 상수 (λ [nm], n, k) — 문헌값을 매끄럽게 근사
_TI_NK = np.array([
    (380.0, 1.80, 2.55), (400.0, 1.86, 2.62), (450.0, 2.00, 2.75), (500.0, 2.16, 2.93), (550.0, 2.37, 3.08),
    (600.0, 2.58, 3.20), (650.0, 2.75, 3.29), (700.0, 2.89, 3.35), (750.0, 3.00, 3.38), (800.0, 3.06, 3.40),
])

# XYZ → 선형 sRGB (D65)
_XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986],
                        [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]])


def _lobe(lam, mu, sigma_low, sigma_high):
    sigma = np.where(lam < mu, sigma_low, sigma_high)
    return np.exp(-0.5 * ((lam - mu) / sigma) ** 2)


def color_matching(lam: np.ndarray) -> np.ndarray:
    """CIE 1931 2° 등색함수 x̄, ȳ, z̄ (Wyman, Sloan, Shirley 2013 의 다중 로브 근사), (3, len(λ))."""
    x = 1.056 * _lobe(lam, 599.8, 37.9, 31.0) + 0.362 * _lobe(lam, 442.0, 16.0, 26.7) \
        - 0.065 * _lobe(lam, 501.1, 20.4, 26.2)
    y = 0.821 * _lobe(lam, 568.8, 46.9, 40.5) + 0.286 * _lobe(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _lobe(lam, 437.0, 11.8, 36.0) + 0.681 * _lobe(lam, 459.0, 26.0, 13.8)
    return np.vstack([x, y, z])


def d65(lam: np.ndarray) -> np.ndarray:
    return np.interp(lam, np.arange(380.0, 781.0, 10.0), _D65_10NM)


def tio2_index(lam: np.ndarray) -> np.ndarray:
    """양극산화 TiO₂ 굴절률 (코시 분산), λ [nm]."""
    um = lam / 1000.0
    return 2.19 + 0.033 / um ** 2


def ti_index(lam: np.ndarray) -> np.ndarray:
    """Ti 금속 복소 굴절률 n − ik, λ [nm]."""
    n = np.interp(lam, _TI_NK[:, 0], _TI_NK[:, 1])
    k = np.interp(lam, _TI_NK[:, 0], _TI_NK[:, 2])
    return n - 1j * k


def reflectance(thickness, lam: np.ndarray = WAVELENGTHS) -> np.ndarray:
    """막 두께 d [m] (스칼라 또는 배열) → 반사율 R(λ), 모양 (len(d), len(λ))."""
    d = np.atleast_1d(np.asarray(thickness, dtype=float))[:, np.newaxis] * 1e9   # nm
    n0, n1, n2 = 1.0, tio2_index(lam)[np.newaxis, :], ti_index(lam)[np.newaxis, :]
    r01 = (n0 - n1) / (n0 + n1)
    r12 = (n1 - n2) / (n1 + n2)
    phase = np.exp(-2j * (2.0 * np.pi * n1 * d / lam[np.newaxis, :]))
    r = (r01 + r12 * phase) / (1.0 + r01 * r12 * phase)
    return np.abs(r) ** 2


def spectrum_to_xyz(spectrum: np.ndarray, lam: np.ndarray = WAVELENGTHS) -> np.ndarray:
    """반사 스펙트럼 (…, len(λ)) → XYZ (…, 3). 완전 반사체(R = 1)가 Y = 1."""
    weights = color_matching(lam) * d65(lam)[np.newaxis, :]          # (3, nλ)
    return (np.asarray(spectrum) @ weights.T) / weights[1].sum()


def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    """XYZ → sRGB (0–1, 감마 적용, 색역 밖은 자름)."""
    linear = np.clip(np.asarray(xyz) @ _XYZ_TO_RGB.T, 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)


def interference_color(thickness) -> np.ndarray:
    """TiO₂ 막 두께 d [m] (스칼라 또는 배열) → sRGB (len(d), 3), 0–1."""
    return xyz_to_srgb(spectrum_to_xyz(reflectance(thickness)))


def color_name(rgb) -> str:
    """sRGB (0–1) → 대략적인 한국어 색 이름 (화면 판독용)."""
    h, s, v = colorsys.rgb_to_hsv(*(float(c) for c in rgb))
    if s < 0.12:
        return "은회색" if v > 0.45 else "회색"
    hue = 360.0 * h
    if hue < 15.0 or hue >= 330.0:
        return "자홍·분홍"
    if hue < 40.0:
        return "청동·갈색" if v < 0.6 else "주황"
    if hue < 70.0:
        return "금색·노랑"
    if hue < 160.0:
        return "초록"
    if hue < 200.0:
        return "청록"
    if hue < 255.0:
        return "파랑"
    if hue < 290.0:
        return "남보라"
    return "보라"


def to_hex(rgb) -> str:
    r, g, b = (int(round(255 * float(c))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"
