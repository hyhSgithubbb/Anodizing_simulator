"""L2: 측면도(x–z) 단면 전해질의 라플라스 방정식 FDM → 전해질 저항 행렬 R (설계서 §1.7).

    ∇·(κ∇φ) = 0           전해질 영역 [0, W] × [0, 액면]
    φ = 0                  음극 표면 (디리클레, 기준 전위)
    ∂φ/∂n = 0              전해조 벽·바닥·액면 (절연)
    −κ ∂φ/∂n = 주입 전류     양극 요소 k 표면 (요소마다 단위 전류를 따로 넣는다)

격자: 직교 비균일 격자. 모든 전극 표면과 양극 요소 경계가 격자선 위에 오도록 하고,
전극 표면에서는 간격 h_min 으로 시작해 멀어질수록 키운다 (최대 h_max).

이산화: 꼭짓점(노드) 기반 유한체적, 5점 스텐실. 전해질 칸 하나가 네 변에 있는 노드 쌍마다
κ·(칸의 반쪽 폭)/(노드 간 거리) 만큼의 전도도를 보탠다. 그래서 전극 내부·전해조 밖은 저절로 빠지고,
절연 경계 조건은 따로 처리할 필요가 없다.

    - 전해질 칸에 하나라도 닿는 노드만 미지수다. 음극 칸에도 닿는 노드는 φ = 0 으로 고정한다.
    - 양극 표면의 격자 변은 하나의 요소에 속한다. 요소 k 의 단위 전류를 접촉 길이에 균일하게 나눠
      변의 양 끝 노드에 반씩 준다 → 주입 행렬 B (노드 × 요소).
    - 요소 평균 전위의 가중치도 같은 B 이므로 R = Bᵀ K⁻¹ B 는 대칭 양의 정부호다.

2D 단면은 깊이(y) 방향으로 균일하다고 가정한다. 단위 깊이당 저항을 등가 깊이
D = (요소 면적 합)/(둘레 길이) 로 나눠 3D 저항으로 바꾼다:  R = Bᵀ K⁻¹ B / (κ D).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from ..geometry import Geometry, PlateElectrode
from .mesh import SurfaceMesh

ELECTROLYTE, ANODE, CATHODE = 0, 1, 2
H_MIN = 0.5e-3        # m — 전극 표면의 격자 간격
H_MAX = 5e-3          # m — 최대 격자 간격 (작은 전해조에서는 크기/40 으로 줄인다)
GROWTH = 0.3          # 전극에서 멀어질 때 간격 증가율 (간격 ≈ h_min + 0.3 × 거리)
MERGE_TOL = 1e-6      # m — 이보다 가까운 격자선은 하나로 합친다
DISPLAY_PIXELS = 400  # 전위장 표시 이미지의 긴 변 화소 수


def _merge(values) -> np.ndarray:
    v = np.sort(np.asarray(values, dtype=float))
    keep = np.concatenate([[True], np.diff(v) > MERGE_TOL])
    return v[keep]


def grid_lines(keys, refine, lo: float, hi: float, h_min: float, h_max: float) -> np.ndarray:
    """격자선 좌표.

    keys: 반드시 격자선이 지나야 하는 좌표 (전극 표면, 요소 경계, 액면 …)
    refine: 간격을 h_min 부터 키우기 시작하는 좌표 (전극 표면 — 모서리 특이점 근처를 촘촘히)

    키 사이 구간마다 양 끝에서 가운데로 걸어 들어가며 선을 놓는다. 그래서 거울 대칭인 배치는
    격자도 정확히 거울 대칭이 된다 (대칭 배치 → 대칭 해).
    """
    keys = _merge(np.clip(np.concatenate([[lo, hi], np.asarray(keys, dtype=float)]), lo, hi))
    refine = np.asarray(refine, dtype=float)

    def spacing(p: float) -> float:
        d = float(np.min(np.abs(refine - p))) if refine.size else math.inf
        return min(h_max, h_min + GROWTH * d)

    def walk(start: float, end: float) -> list[float]:
        """start 에서 구간 가운데를 넘지 않을 때까지 간격대로 걸은 점들."""
        mid, direction, p, points = 0.5 * (start + end), math.copysign(1.0, end - start), start, []
        while True:
            nxt = p + direction * spacing(p)
            if direction * (mid - nxt) <= 0.0:
                return points
            points.append(nxt)
            p = nxt

    out = [keys[0]]
    for a, b in zip(keys[:-1], keys[1:], strict=True):
        left, right = walk(a, b), walk(b, a)
        inner_a = left[-1] if left else a
        inner_b = right[-1] if right else b
        h_mid = spacing(0.5 * (a + b))
        middle = []
        if inner_b - inner_a > 1.3 * h_mid:          # 가운데 빈 곳이 한 칸보다 넓으면 중점을 넣는다
            middle = [0.5 * (a + b)]
        elif left and right and inner_b - inner_a < 0.3 * h_mid:   # 너무 좁으면 두 점을 중점 하나로
            left, right, middle = left[:-1], right[:-1], [0.5 * (a + b)]
        out.extend(left + middle + right[::-1] + [b])
    return np.asarray(out)


def _wet_rect(plate: PlateElectrode, geometry: Geometry):
    level = geometry.tank.liquid_level
    return (max(plate.x_min, 0.0), min(plate.x_max, geometry.tank.width),
            max(plate.z_bottom, 0.0), min(plate.z_top, level))


def _index(lines: np.ndarray, value: float) -> int:
    k = int(np.argmin(np.abs(lines - value)))
    if abs(lines[k] - value) > 10 * MERGE_TOL:
        raise ValueError("FDM 격자선이 전극 표면과 맞지 않습니다.")
    return k


@dataclass(frozen=True, eq=False)
class FDMSolution:
    """L2 해석 결과. 전위는 '단위 깊이당 단위 전류, κ = 1 S/m' 기준이다."""

    x: np.ndarray                 # 격자선 x [m], (nx,)
    z: np.ndarray                 # 격자선 z [m], (nz,)
    cells: np.ndarray             # 칸 종류 (nz−1, nx−1): ELECTROLYTE / ANODE / CATHODE
    unknown: np.ndarray           # 미지 노드의 평탄 인덱스 (j·nx + i), (n_u,)
    basis: np.ndarray             # 요소 k 단위 전류에 대한 미지 노드 전위 (n_u, N), float32
    unit_resistance_2d: np.ndarray  # Bᵀ K⁻¹ B [Ω·m] (κ = 1, 단위 깊이), (N, N)
    depth: float                  # 등가 깊이 D [m]
    contact: np.ndarray           # 요소별 전해질 접촉 길이 [m], (N,)

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.size, self.x.size

    @property
    def unit_resistance(self) -> np.ndarray:
        """κ = 1 S/m 일 때의 3D 저항 행렬 [Ω]."""
        return self.unit_resistance_2d / self.depth

    def node_potential(self, element_current, conductivity: float) -> np.ndarray:
        """요소 전류 I [A] 일 때 격자 노드 전위 [V] (nz, nx). 전해질에 닿지 않는 노드는 nan, 음극은 0."""
        phi = np.full(self.x.size * self.z.size, np.nan)
        phi[self._fixed] = 0.0
        phi[self.unknown] = self.basis @ (np.asarray(element_current, dtype=float) / (conductivity * self.depth))
        return phi.reshape(self.shape)

    def potential_image(self, element_current, conductivity: float):
        """표시용 균일 격자 전위 이미지 [V] (행 = z 아래→위, 열 = x), 전극 안은 nan.

        돌려주는 값: (이미지, (x0, x1, z0, z1) [m]).
        """
        weights, outside, shape = self._display
        values = self.basis @ (np.asarray(element_current, dtype=float) / (conductivity * self.depth))
        image = np.asarray(weights @ values, dtype=float)
        image[outside] = np.nan
        return image.reshape(shape), (float(self.x[0]), float(self.x[-1]), float(self.z[0]), float(self.z[-1]))

    @cached_property
    def _fixed(self) -> np.ndarray:
        nz, nx = self.shape
        touched_e = _corner_mask(self.cells == ELECTROLYTE, nz, nx)
        touched_c = _corner_mask(self.cells == CATHODE, nz, nx)
        return np.flatnonzero((touched_e & touched_c).ravel())

    @cached_property
    def _display(self):
        """노드 전위 → 표시 화소의 쌍선형 보간 행렬 (고정 노드는 φ = 0 이라 빠진다)."""
        x, z = self.x, self.z
        width, height = x[-1] - x[0], z[-1] - z[0]
        size = max(width, height) / DISPLAY_PIXELS
        nxp, nzp = max(int(round(width / size)), 2), max(int(round(height / size)), 2)
        xp = x[0] + (np.arange(nxp) + 0.5) * width / nxp
        zp = z[0] + (np.arange(nzp) + 0.5) * height / nzp
        i = np.clip(np.searchsorted(x, xp) - 1, 0, x.size - 2)
        j = np.clip(np.searchsorted(z, zp) - 1, 0, z.size - 2)
        tx = (xp - x[i]) / (x[i + 1] - x[i])
        tz = (zp - z[j]) / (z[j + 1] - z[j])
        jj, ii = np.meshgrid(j, i, indexing="ij")
        tzz, txx = np.meshgrid(tz, tx, indexing="ij")
        nx = x.size
        pixel = np.arange(nzp * nxp)
        position = np.full(x.size * z.size, -1)
        position[self.unknown] = np.arange(self.unknown.size)
        rows, cols, vals = [], [], []
        for dj, di, w in ((0, 0, (1 - tzz) * (1 - txx)), (0, 1, (1 - tzz) * txx),
                          (1, 0, tzz * (1 - txx)), (1, 1, tzz * txx)):
            node = ((jj + dj) * nx + (ii + di)).ravel()
            col = position[node]
            ok = col >= 0
            rows.append(pixel[ok])
            cols.append(col[ok])
            vals.append(w.ravel()[ok])
        weights = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                    shape=(pixel.size, self.unknown.size))
        outside = (self.cells[jj, ii] != ELECTROLYTE).ravel()
        return weights, outside, (nzp, nxp)


def _corner_mask(cell_mask: np.ndarray, nz: int, nx: int) -> np.ndarray:
    """칸 마스크 → 그 칸의 네 꼭짓점 노드 마스크 (nz, nx)."""
    nodes = np.zeros((nz, nx), dtype=bool)
    nodes[:-1, :-1] |= cell_mask
    nodes[:-1, 1:] |= cell_mask
    nodes[1:, :-1] |= cell_mask
    nodes[1:, 1:] |= cell_mask
    return nodes


def _element_cuts(mesh: SurfaceMesh):
    """요소 경계 좌표 (세로 면 → z, 가로 면 → x)."""
    vertical = np.isin(mesh.face, (0, 2))
    half = 0.5 * mesh.length
    z_cuts = np.concatenate([mesh.z[vertical] - half[vertical], mesh.z[vertical] + half[vertical]])
    x_cuts = np.concatenate([mesh.x[~vertical] - half[~vertical], mesh.x[~vertical] + half[~vertical]])
    return x_cuts, z_cuts


def solve_fdm(geometry: Geometry, mesh: SurfaceMesh, grid_scale: float = 1.0) -> FDMSolution:
    """양극 요소별 단위 전류 주입 문제를 풀어 저항 행렬과 전위 기저를 만든다.

    grid_scale: 격자 간격 배율 (0.5 = 두 배 촘촘히; 격자 수렴 확인용).
    """
    tank = geometry.tank
    width, level = tank.width, tank.liquid_level
    h_max = min(H_MAX, min(width, level) / 40.0) * grid_scale
    h_min = min(H_MIN * grid_scale, 0.5 * h_max)

    anode = _wet_rect(geometry.anode, geometry)
    cathodes = [_wet_rect(c, geometry) for c in geometry.cathodes]
    x_cuts, z_cuts = _element_cuts(mesh)
    rects = [anode] + cathodes
    x_surfaces = [v for r in rects for v in r[:2]]
    z_surfaces = [v for r in rects for v in r[2:]]
    x = grid_lines(np.concatenate([x_surfaces, x_cuts]), x_surfaces, 0.0, width, h_min, h_max)
    z = grid_lines(np.concatenate([z_surfaces, z_cuts]), z_surfaces, 0.0, level, h_min, h_max)
    nx, nz = x.size, z.size

    # ── 칸 분류 ──────────────────────────────────────────────────────────────
    xc, zc = 0.5 * (x[:-1] + x[1:]), 0.5 * (z[:-1] + z[1:])
    zz, xx = np.meshgrid(zc, xc, indexing="ij")

    def inside(r):
        return (xx > r[0]) & (xx < r[1]) & (zz > r[2]) & (zz < r[3])

    cells = np.full(zz.shape, ELECTROLYTE, dtype=np.int8)
    for r in cathodes:
        cells[inside(r)] = CATHODE
    cells[inside(anode)] = ANODE

    electrolyte = cells == ELECTROLYTE
    touched_e = _corner_mask(electrolyte, nz, nx)
    touched_c = _corner_mask(cells == CATHODE, nz, nx)
    unknown_mask = (touched_e & ~touched_c).ravel()
    unknown = np.flatnonzero(unknown_mask)
    position = np.full(nx * nz, -1)
    position[unknown] = np.arange(unknown.size)

    # ── 전도도 행렬 K (κ = 1, 단위 깊이) ────────────────────────────────────
    jj, ii = np.nonzero(electrolyte)
    dx, dz = np.diff(x)[ii], np.diff(z)[jj]
    n00 = jj * nx + ii
    n10, n01, n11 = n00 + 1, n00 + nx, n00 + nx + 1
    gx, gz = 0.5 * dz / dx, 0.5 * dx / dz
    p = np.concatenate([n00, n01, n00, n10])
    q = np.concatenate([n10, n11, n01, n11])
    g = np.concatenate([gx, gx, gz, gz])
    pp, qq = position[p], position[q]
    rows, cols, vals = [], [], []
    for a, b in ((pp, qq), (qq, pp)):
        ok = a >= 0
        rows += [a[ok]]
        cols += [a[ok]]
        vals += [g[ok]]                       # 대각: 고정 노드(φ = 0)로 가는 링크도 포함
        both = ok & (b >= 0)
        rows += [a[both]]
        cols += [b[both]]
        vals += [-g[both]]
    k_matrix = sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(unknown.size, unknown.size))

    # ── 양극 표면 변 → 요소, 주입 행렬 B ────────────────────────────────────
    ia0, ia1 = _index(x, anode[0]), _index(x, anode[1])
    jb, jt = _index(z, anode[2]), _index(z, anode[3])
    bounds = np.concatenate([[0.0], np.cumsum(mesh.length)])
    face_start = dict(zip(mesh.faces, mesh.edges[:-1], strict=True))
    edges = []   # (노드 a, 노드 b, 길이, s 중점)
    for face in mesh.faces:
        s0 = face_start[face]
        if face in (0, 2):
            i_line = ia0 if face == 0 else ia1
            i_cell = ia0 - 1 if face == 0 else ia1
            for j in range(jb, jt):
                if not (0 <= i_cell < nx - 1) or cells[j, i_cell] != ELECTROLYTE:
                    continue
                zm = 0.5 * (z[j] + z[j + 1])
                s = s0 + (anode[3] - zm if face == 0 else zm - anode[2])
                edges.append((j * nx + i_line, (j + 1) * nx + i_line, z[j + 1] - z[j], s))
        else:
            j_line = jb if face == 1 else jt
            j_cell = jb - 1 if face == 1 else jt
            for i in range(ia0, ia1):
                if not (0 <= j_cell < nz - 1) or cells[j_cell, i] != ELECTROLYTE:
                    continue
                xm = 0.5 * (x[i] + x[i + 1])
                s = s0 + (xm - anode[0] if face == 1 else anode[1] - xm)
                edges.append((j_line * nx + i, j_line * nx + i + 1, x[i + 1] - x[i], s))
    if not edges:
        raise ValueError("양극이 전해질에 닿지 않습니다.")
    na, nb, length, s_mid = (np.array(col) for col in zip(*edges, strict=True))
    element = np.clip(np.searchsorted(bounds, s_mid, side="right") - 1, 0, mesh.size - 1)
    contact = np.bincount(element, weights=length, minlength=mesh.size)
    if np.any(contact <= 0.0):
        raise ValueError("전해질에 닿지 않는 양극 요소가 있습니다 (벽·바닥·음극에 붙어 있음).")
    if np.any(position[na] < 0) or np.any(position[nb] < 0):
        raise ValueError("양극과 음극이 맞닿아 있습니다.")
    share = 0.5 * length / contact[element]
    b_matrix = np.zeros((unknown.size, mesh.size))
    np.add.at(b_matrix, (position[na], element), share)
    np.add.at(b_matrix, (position[nb], element), share)

    # ── 풀이: 요소마다 단위 전류 → 전위 기저, R = Bᵀ K⁻¹ B ──────────────────
    try:
        lu = splu(k_matrix)
    except RuntimeError as exc:   # 음극과 이어지지 않은 전해질 영역 (특이 행렬)
        raise ValueError("음극과 이어지지 않은 전해질 영역이 있어 전위를 정할 수 없습니다.") from exc
    basis = lu.solve(b_matrix)
    r2d = b_matrix.T @ basis
    r2d = 0.5 * (r2d + r2d.T)   # 반올림 오차 제거 (이론상 대칭)
    depth = float(mesh.area.sum() / mesh.length.sum())
    return FDMSolution(x=x, z=z, cells=cells, unknown=unknown, basis=basis.astype(np.float32),
                       unit_resistance_2d=r2d, depth=depth, contact=contact)
