"""전해조·전극 형상.

좌표계 (측면도 기준):
    x — 전해조 가로 (측면도의 가로축)
    z — 높이, 위쪽이 + (측면도의 세로축, 바닥 z = 0)
    y — 깊이 (측면도 화면 안쪽 방향, 3D 보기에서만 보임)

전극은 y–z 평면에 평행한 판(plate)이며 ±x 방향 면이 서로 마주본다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

_MIN_PATH = 1e-3  # m — 경로 저항 근사에서 쓰는 최소 거리


@dataclass(frozen=True)
class Box:
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float

    def overlaps(self, other: Box) -> bool:
        return (self.x0 < other.x1 and other.x0 < self.x1
                and self.y0 < other.y1 and other.y0 < self.y1
                and self.z0 < other.z1 and other.z0 < self.z1)


@dataclass(frozen=True)
class Tank:
    width: float          # x [m]
    depth: float          # y [m]
    height: float         # z [m]
    liquid_level: float   # z [m]


@dataclass(frozen=True)
class PlateElectrode:
    x_center: float
    z_bottom: float
    height: float
    thickness: float      # x 방향 두께 [m]
    width: float          # y 방향 폭 [m]
    y_center: float | None = None   # None 이면 전해조 깊이 중앙

    @property
    def z_top(self) -> float:
        return self.z_bottom + self.height

    @property
    def x_min(self) -> float:
        return self.x_center - 0.5 * self.thickness

    @property
    def x_max(self) -> float:
        return self.x_center + 0.5 * self.thickness

    def immersed_height(self, liquid_level: float) -> float:
        return max(0.0, min(self.z_top, liquid_level) - self.z_bottom)

    def face_area(self, liquid_level: float) -> float:
        """잠긴 한쪽 넓은 면의 면적 [m²]."""
        return self.immersed_height(liquid_level) * self.width

    def wetted_area(self, liquid_level: float) -> float:
        """전해질에 닿는 전체 면적 [m²] — 앞뒤 두 면 + 옆 모서리 + 바닥."""
        h = self.immersed_height(liquid_level)
        if h <= 0.0:
            return 0.0
        bottom = self.thickness * self.width if self.z_bottom < liquid_level else 0.0
        return 2.0 * h * self.width + 2.0 * h * self.thickness + bottom

    def box(self, tank: Tank) -> Box:
        yc = 0.5 * tank.depth if self.y_center is None else self.y_center
        return Box(self.x_min, self.x_max, yc - 0.5 * self.width, yc + 0.5 * self.width,
                   self.z_bottom, self.z_top)


@dataclass(frozen=True)
class Geometry:
    tank: Tank
    anode: PlateElectrode
    cathodes: tuple[PlateElectrode, ...]

    @classmethod
    def default(cls) -> Geometry:
        """40 × 25 × 35 cm 전해조, 가운데 양극, 좌우 음극 2장."""
        tank = Tank(width=0.40, depth=0.25, height=0.35, liquid_level=0.30)
        anode = PlateElectrode(x_center=0.20, z_bottom=0.08, height=0.15, thickness=0.002, width=0.10)
        cathode = PlateElectrode(x_center=0.08, z_bottom=0.05, height=0.30, thickness=0.003, width=0.15)
        return cls(tank, anode, (cathode, replace(cathode, x_center=0.32)))

    @property
    def anode_area(self) -> float:
        return self.anode.wetted_area(self.tank.liquid_level)

    @property
    def cathode_area(self) -> float:
        return sum(c.wetted_area(self.tank.liquid_level) for c in self.cathodes)

    def validate(self) -> list[str]:
        t = self.tank
        errors: list[str] = []
        if min(t.width, t.depth, t.height) <= 0:
            errors.append("전해조 크기는 0보다 커야 합니다.")
            return errors
        if not 0 < t.liquid_level <= t.height:
            errors.append("액면 높이는 0보다 크고 전해조 높이 이하여야 합니다.")
        if not self.cathodes:
            errors.append("음극이 최소 1개 필요합니다.")
        named = [("양극", self.anode)] + [(f"음극 {i + 1}", c) for i, c in enumerate(self.cathodes)]
        for name, e in named:
            if min(e.height, e.thickness, e.width) <= 0:
                errors.append(f"{name}: 높이·두께·폭은 0보다 커야 합니다.")
                continue
            b = e.box(t)
            if b.x0 < 0 or b.x1 > t.width or b.y0 < 0 or b.y1 > t.depth:
                errors.append(f"{name}: 전해조 밖으로 벗어났습니다.")
            if e.z_bottom < 0:
                errors.append(f"{name}: 바닥 높이는 0 이상이어야 합니다.")
            if e.immersed_height(t.liquid_level) <= 0:
                errors.append(f"{name}: 전해질에 잠기지 않았습니다.")
        boxes = [(name, e.box(t)) for name, e in named if min(e.height, e.thickness, e.width) > 0]
        for i in range(len(boxes)):
            for k in range(i + 1, len(boxes)):
                if boxes[i][1].overlaps(boxes[k][1]):
                    errors.append(f"{boxes[i][0]}와(과) {boxes[k][0]}이(가) 겹칩니다.")
        return errors

    def lumped_resistance(self, conductivity: float) -> float:
        """L0 근사: 양극 전체와 음극 사이의 전해질 저항 [Ω].

        양극의 좌우 두 면 각각에 대해, 같은 쪽 음극은 면 사이 거리, 반대쪽 음극은
        판을 돌아가는 우회 경로(거리 + 판 두께 + 잠긴 높이/2)로 경로 길이 L 을 잡고
        면 전도도 κ·A_face/L 을 모두 병렬로 더한다.
        """
        level = self.tank.liquid_level
        a = self.anode
        face = a.face_area(level)
        detour = a.thickness + 0.5 * a.immersed_height(level)
        conductance = 0.0
        for side in (-1.0, 1.0):
            for c in self.cathodes:
                gap = abs(c.x_center - a.x_center) - 0.5 * (a.thickness + c.thickness)
                same_side = (c.x_center - a.x_center) * side > 0
                path = gap if same_side else gap + detour
                conductance += conductivity * face / max(path, _MIN_PATH)
        if conductance <= 0:
            raise ValueError("전극 배치로부터 전해질 저항을 계산할 수 없습니다.")
        return 1.0 / conductance
