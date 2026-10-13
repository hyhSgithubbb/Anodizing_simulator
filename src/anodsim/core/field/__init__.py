"""필드 해석: 양극 표면 요소와 전해질 저항 행렬 R 을 만든다 (설계서 §1.7).

해석 단계마다 바뀌는 것은 표면 요소(면적 a)와 R 뿐이고, 성장 모델·회로 해석·시간 적분은 그대로다.

    L0 — 양극 전체가 한 요소, R = Geometry.lumped_resistance (1×1)
    L1 — 측면도 둘레를 따라 N 개 요소, 경로 근사 대각 행렬 R_ii = L_eff,i/(κ a_i)
    L2 — 같은 요소, 측면도 단면 라플라스 방정식 FDM 으로 만든 조밀한 대칭 행렬 R (요소 사이 결합 포함)

κ 가 균일하면 R ∝ 1/κ 이므로 κ = 1 S/m 일 때의 R 만 형상에서 한 번 구해 둔다.
같은 형상·단계·요소 수로 다시 계산하면 캐시된 결과를 쓴다 (L2 격자 풀이를 반복하지 않도록).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..geometry import Geometry
from .fdm import FDMSolution, solve_fdm
from .mesh import FACE_NAMES, LUMPED, SurfaceMesh, build_surface_mesh
from .path_model import effective_path_length, path_lengths

FIDELITY_LEVELS = {
    "L0": "L0 — 0D (양극 평균, 빠른 추정)",
    "L1": "L1 — 위치별 두께 (경로 저항 근사)",
    "L2": "L2 — 위치별 두께 (전해질 전위장 FDM)",
}
DEFAULT_ELEMENTS = 120


@dataclass(frozen=True, eq=False)
class Field:
    """한 형상에 대한 해석 준비물."""

    level: str
    mesh: SurfaceMesh
    unit_resistance: np.ndarray        # κ = 1 S/m 일 때의 전해질 저항 행렬 [Ω], (N, N)
    fdm: FDMSolution | None = None     # L2 만: 격자·전위 기저 (전위장 표시용)

    def resistance(self, conductivity: float) -> np.ndarray:
        """전해질 저항 행렬 R [Ω], (N, N)."""
        return self.unit_resistance / conductivity

    @property
    def has_potential(self) -> bool:
        return self.fdm is not None

    def potential_image(self, element_current, conductivity: float):
        """요소 전류 I [A] 일 때 전해질 전위 이미지 (L2 만, 아니면 None). FDMSolution.potential_image 참고."""
        if self.fdm is None:
            return None
        return self.fdm.potential_image(element_current, conductivity)


def build_field(geometry: Geometry, fidelity: str = "L1", elements: int = DEFAULT_ELEMENTS,
                grid_scale: float = 1.0) -> Field:
    if fidelity not in FIDELITY_LEVELS:
        raise ValueError(f"지원하지 않는 해석 단계: {fidelity}")
    if fidelity == "L0":
        elements = 1
    return _build_field(geometry, fidelity, int(elements), float(grid_scale))


@lru_cache(maxsize=3)
def _build_field(geometry: Geometry, fidelity: str, elements: int, grid_scale: float) -> Field:
    if fidelity == "L0":
        mesh = SurfaceMesh.lumped(geometry)
        # lumped_resistance 는 κ 에 반비례하므로 κ = 1 로 구한다
        return Field("L0", mesh, np.array([[geometry.lumped_resistance(1.0)]]))
    mesh = build_surface_mesh(geometry, elements)
    if fidelity == "L1":
        return Field("L1", mesh, np.diag(effective_path_length(geometry, mesh) / mesh.area))
    fdm = solve_fdm(geometry, mesh, grid_scale)
    return Field("L2", mesh, fdm.unit_resistance, fdm)


__all__ = ["DEFAULT_ELEMENTS", "FACE_NAMES", "FIDELITY_LEVELS", "LUMPED", "FDMSolution", "Field", "SurfaceMesh",
           "build_field", "build_surface_mesh", "effective_path_length", "path_lengths", "solve_fdm"]
