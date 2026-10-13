"""필드 해석: 양극 표면 요소와 전해질 저항 행렬 R 을 만든다 (설계서 §1.7).

해석 단계마다 바뀌는 것은 표면 요소(면적 a)와 R 뿐이고, 성장 모델·회로 해석·시간 적분은 그대로다.

    L0 — 양극 전체가 한 요소, R = Geometry.lumped_resistance (1×1)
    L1 — 측면도 둘레를 따라 N 개 요소, 경로 근사 대각 행렬 R_ii = L_eff,i/(κ a_i)

κ 가 균일하면 R ∝ 1/κ 이므로 경로 길이 L_eff 만 형상에서 한 번 구해 두면 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import Geometry
from .mesh import FACE_NAMES, LUMPED, SurfaceMesh, build_surface_mesh
from .path_model import effective_path_length, path_lengths

FIDELITY_LEVELS = {
    "L0": "L0 — 0D (양극 평균, 빠른 추정)",
    "L1": "L1 — 위치별 두께 (경로 저항 근사)",
}
DEFAULT_ELEMENTS = 120


@dataclass(frozen=True)
class Field:
    """한 형상에 대한 해석 준비물."""

    mesh: SurfaceMesh
    path_length: np.ndarray   # L_eff,i [m] — R_ii·κ·a_i (L0 은 등가 길이)

    def resistance(self, conductivity: float) -> np.ndarray:
        """전해질 저항 행렬 R [Ω], (N, N)."""
        return np.diag(self.path_length / (conductivity * self.mesh.area))


def build_field(geometry: Geometry, fidelity: str = "L1", elements: int = DEFAULT_ELEMENTS) -> Field:
    if fidelity == "L0":
        mesh = SurfaceMesh.lumped(geometry)
        # lumped_resistance 는 κ 에 반비례하므로 κ = 1 로 등가 길이를 얻는다
        length = geometry.lumped_resistance(1.0) * mesh.area
        return Field(mesh, np.asarray(length, dtype=float))
    if fidelity == "L1":
        mesh = build_surface_mesh(geometry, elements)
        return Field(mesh, effective_path_length(geometry, mesh))
    raise ValueError(f"지원하지 않는 해석 단계: {fidelity}")


__all__ = ["DEFAULT_ELEMENTS", "FACE_NAMES", "FIDELITY_LEVELS", "LUMPED", "Field", "SurfaceMesh",
           "build_field", "build_surface_mesh", "effective_path_length", "path_lengths"]
