"""양극 표면 이산화 — 측면도(x–z) 단면의 둘레를 따라가는 1D 좌표 s.

판 양극의 잠긴 단면은 직사각형 [x_min, x_max] × [z_bottom, z_wet] 이다
(z_wet = min(윗면, 액면)). 둘레를 다음 순서로 돈다.

    시작 (x_min, z_wet) ─ −x 면(아래로) ─ 바닥(+x 방향) ─ +x 면(위로) ─ [윗면(−x 방향), 완전히 잠긴 경우만]

깊이(y) 방향으로는 균일하다고 본다(측면도 가정). 요소 면적은 길이 × 폭이고,
y 쪽 두 끝면(두께 × 높이)은 같은 높이의 −x/+x 면 요소에 절반씩 붙인다.
그래서 요소 면적의 합은 항상 `PlateElectrode.wetted_area` 와 같다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import Geometry

FACE_NAMES = ("−x 면", "바닥", "+x 면", "윗면")
FACE_NORMALS = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
LUMPED = -1   # L0: 양극 전체가 한 요소


@dataclass(frozen=True)
class SurfaceMesh:
    """양극 표면 요소 (길이 N 배열, SI 단위)."""

    s: np.ndarray          # 요소 중심의 둘레 좌표 [m]
    length: np.ndarray     # 요소 길이 (둘레 방향) [m]
    area: np.ndarray       # 요소 면적 [m²]
    x: np.ndarray          # 요소 중심 x [m]
    z: np.ndarray          # 요소 중심 z [m]
    face: np.ndarray       # 면 번호 (FACE_NAMES 인덱스, L0 은 LUMPED)
    edges: np.ndarray      # 면 경계의 s 좌표 (길이 = 면 수 + 1) [m]
    faces: tuple[int, ...]  # 둘레 순서대로 나타나는 면 번호

    @property
    def size(self) -> int:
        return int(self.area.size)

    @property
    def normal(self) -> np.ndarray:
        """(N, 2) 바깥 법선 (x, z)."""
        return FACE_NORMALS[self.face]

    @property
    def perimeter(self) -> float:
        return float(self.edges[-1])

    @classmethod
    def lumped(cls, geometry: Geometry) -> SurfaceMesh:
        a = geometry.anode
        level = geometry.tank.liquid_level
        zc = a.z_bottom + 0.5 * a.immersed_height(level)
        one = np.ones(1)
        return cls(s=np.zeros(1), length=one * 0.0, area=one * geometry.anode_area, x=one * a.x_center,
                   z=one * zc, face=np.array([LUMPED]), edges=np.zeros(1), faces=())


def anode_outline(geometry: Geometry):
    """잠긴 양극 단면의 둘레 꼭짓점 (s 순서)과 면 번호.

    돌려주는 값: (꼭짓점 (M, 2) 배열, 면 번호 튜플). 완전히 잠기면 닫힌 둘레(마지막 꼭짓점 = 첫 꼭짓점).
    """
    a = geometry.anode
    level = geometry.tank.liquid_level
    x0, x1 = a.x_min, a.x_max
    z0 = a.z_bottom
    z1 = z0 + a.immersed_height(level)
    vertices = [(x0, z1), (x0, z0), (x1, z0), (x1, z1)]
    faces = (0, 1, 2)
    if a.fully_immersed(level):
        vertices.append((x0, z1))
        faces = (0, 1, 2, 3)
    return np.array(vertices, dtype=float), faces


def build_surface_mesh(geometry: Geometry, target_elements: int = 120) -> SurfaceMesh:
    """둘레를 대략 같은 길이의 요소로 나눈다 (각 면 최소 1개)."""
    if target_elements < 1:
        raise ValueError("target_elements must be ≥ 1")
    a = geometry.anode
    vertices, faces = anode_outline(geometry)
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    perimeter = float(lengths.sum())
    if perimeter <= 0.0:
        raise ValueError("양극이 전해질에 잠기지 않았습니다.")
    ds = perimeter / target_elements
    edges = np.concatenate([[0.0], np.cumsum(lengths)])

    cols = {k: [] for k in ("s", "length", "area", "x", "z", "face")}
    for f, (p0, p1), s0, length in zip(faces, zip(vertices[:-1], vertices[1:], strict=True), edges[:-1], lengths,
                                       strict=True):
        n = max(1, int(round(length / ds)))
        frac = (np.arange(n) + 0.5) / n
        seg = np.full(n, length / n)
        # −x/+x 면 요소는 y 끝면(두께 × 같은 높이 구간)의 절반씩을 함께 가진다
        depth = a.width + (a.thickness if f in (0, 2) else 0.0)
        cols["s"].append(s0 + frac * length)
        cols["length"].append(seg)
        cols["area"].append(seg * depth)
        cols["x"].append(p0[0] + frac * (p1[0] - p0[0]))
        cols["z"].append(p0[1] + frac * (p1[1] - p0[1]))
        cols["face"].append(np.full(n, f))
    arrays = {k: np.concatenate(v) for k, v in cols.items()}
    arrays["face"] = arrays["face"].astype(int)
    return SurfaceMesh(edges=edges, faces=tuple(faces), **arrays)
