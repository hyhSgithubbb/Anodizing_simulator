"""L1 경로 저항 근사 (설계서 §1.7).

각 양극 요소 i 는 음극 k 마다 독립된 전해질 경로(길이 L_ik)로 이어져 있다고 본다.

    1/L_eff,i = Σ_k 1/L_ik,      R_ii = L_eff,i / (κ·a_i),      R_ik = 0 (i ≠ k)

L_ik 는 측면도(x–z) 단면에서 요소 중심부터 음극 k 의 잠긴 단면까지의 최단 경로다.

- 양극 판 자체는 장애물이다. 음극이 보이지 않는 면(반대쪽 음극을 등진 면)은 판 둘레를 따라
  모서리까지 간 뒤 그 모서리에서 직선으로 간다 (우회 경로).
- 액면은 절연 경계라서, 윗면이 액면 위로 나온 양극은 아래쪽(바닥 모서리)으로만 돌아갈 수 있다.
- 음극 단면은 잠긴 부분만 쓴다. 다른 음극에 의한 가림, 전해조 벽의 영향은 무시한다.

이 근사가 잡는 효과: 음극까지의 거리·높이 차이(음극이 덮지 못하는 부분은 얇아짐),
한쪽에만 음극이 있을 때 등진 면이 얇아지는 현상, 양쪽 음극에 가까운 판 끝(바닥·윗면 부근)이 두꺼워지는 현상.
잡지 못하는 효과: 모서리 전류 집중(1차 전류 분포의 특이성)과 요소 사이의 전위 결합 — L2(FDM, P3)의 몫이다.
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry import Geometry, PlateElectrode
from .mesh import FACE_NORMALS, SurfaceMesh, anode_outline

MIN_PATH = 1e-3   # m — 거리 하한 (음극이 판에 거의 붙은 경우의 발산 방지)

Rect = tuple[float, float, float, float]   # (x0, x1, z0, z1)


def _wet_rect(plate: PlateElectrode, level: float) -> Rect:
    return plate.x_min, plate.x_max, plate.z_bottom, plate.z_bottom + plate.immersed_height(level)


def _clip(rect: Rect, face: int, px: float, pz: float) -> Rect:
    """직사각형을 점 (px, pz) 를 지나는 면 법선 쪽 반평면으로 자른다 (법선이 축 방향이라 직사각형으로 남는다)."""
    x0, x1, z0, z1 = rect
    nx, nz = FACE_NORMALS[face]
    if nx < 0:
        x1 = min(x1, px)
    elif nx > 0:
        x0 = max(x0, px)
    if nz < 0:
        z1 = min(z1, pz)
    elif nz > 0:
        z0 = max(z0, pz)
    return x0, x1, z0, z1


def _distance(rect: Rect, px: float, pz: float) -> float:
    x0, x1, z0, z1 = rect
    if x0 > x1 or z0 > z1:
        return math.inf   # 보이는 부분이 없다
    dx = max(x0 - px, 0.0, px - x1)
    dz = max(z0 - pz, 0.0, pz - z1)
    return math.hypot(dx, dz)


class _Outline:
    """양극 단면 둘레 — 꼭짓점에서 보이는 영역과 둘레를 따라 걷는 거리."""

    def __init__(self, geometry: Geometry):
        self.vertices, self.faces = anode_outline(geometry)
        self.lengths = np.linalg.norm(np.diff(self.vertices, axis=0), axis=1)
        self.n_edges = len(self.faces)
        self.closed = self.n_edges == 4   # 완전히 잠긴 판: 윗면으로도 돌아갈 수 있다

    def vertex_faces(self, m: int) -> list[int]:
        e = self.n_edges
        if self.closed:
            return [self.faces[(m - 1) % e], self.faces[m % e]]
        return [self.faces[k] for k in (m - 1, m) if 0 <= k < e]

    def vertex_distance(self, m: int, rect: Rect) -> float:
        vx, vz = self.vertices[m % len(self.vertices)] if self.closed else self.vertices[m]
        return min(_distance(_clip(rect, f, vx, vz), vx, vz) for f in self.vertex_faces(m))

    def walks(self, edge: int, offset: float):
        """요소(변 edge, 변 시작점에서 offset)에서 둘레를 따라 각 꼭짓점까지의 (꼭짓점, 거리)."""
        e = self.n_edges
        # 정방향 (s 증가)
        d = self.lengths[edge] - offset
        stop = edge + e if self.closed else e
        for m in range(edge + 1, stop + 1):
            yield m % e if self.closed else m, d
            if m < stop:
                d += self.lengths[m % e]
        # 역방향 (s 감소)
        d = offset
        stop = edge - e + 1 if self.closed else 0
        for m in range(edge, stop - 1, -1):
            yield m % e if self.closed else m, d
            if m > stop:
                d += self.lengths[(m - 1) % e]


def path_lengths(geometry: Geometry, mesh: SurfaceMesh) -> np.ndarray:
    """요소 i 에서 음극 k 까지의 최단 경로 길이 L_ik, (N, K) [m]."""
    level = geometry.tank.liquid_level
    outline = _Outline(geometry)
    rects = [_wet_rect(c, level) for c in geometry.cathodes]
    edge_of = {f: k for k, f in enumerate(outline.faces)}
    out = np.empty((mesh.size, len(rects)))
    for i in range(mesh.size):
        face = int(mesh.face[i])
        edge = edge_of[face]
        offset = float(mesh.s[i] - np.sum(outline.lengths[:edge]))
        px, pz = float(mesh.x[i]), float(mesh.z[i])
        for k, rect in enumerate(rects):
            best = _distance(_clip(rect, face, px, pz), px, pz)        # 직접 보이는 경우
            for m, walked in outline.walks(edge, offset):              # 판 둘레를 돌아가는 경우
                if walked >= best:
                    continue
                best = min(best, walked + outline.vertex_distance(m, rect))
            out[i, k] = max(best, MIN_PATH)
    return out


def effective_path_length(geometry: Geometry, mesh: SurfaceMesh) -> np.ndarray:
    """L_eff,i = 1 / Σ_k (1/L_ik) [m]."""
    return 1.0 / np.sum(1.0 / path_lengths(geometry, mesh), axis=1)
