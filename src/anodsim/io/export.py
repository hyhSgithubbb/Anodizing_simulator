"""계산 결과 CSV 내보내기 (UTF-8, 사람이 읽는 단위를 열 이름에 붙임)."""

from __future__ import annotations

import csv

from ..core import units as u
from ..core.field import FACE_NAMES, LUMPED


def write_timeseries_csv(path, result) -> None:
    """시간 이력 (면적 가중 평균 + 위치별 최소·최대 두께, 욕 온도, 전원 단계)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_min", "voltage_V", "current_A", "control", "step", "current_density_A_dm2",
                    "film_voltage_V", "barrier_nm", "porous_um", "total_um", "total_min_um", "total_max_um",
                    "charge_C_cm2", "bath_temperature_C"])
        for k in range(result.time.size):
            w.writerow([f"{result.time[k] / u.MINUTE:.6g}", f"{result.voltage[k]:.6g}", f"{result.current[k]:.6g}",
                        result.control[k], int(result.step[k]) + 1,
                        f"{result.current_density[k] / u.A_PER_DM2:.6g}",
                        f"{result.film_voltage[k]:.6g}", f"{result.barrier[k] / u.NM:.6g}",
                        f"{result.porous[k] / u.UM:.6g}", f"{result.total[k] / u.UM:.6g}",
                        f"{result.total_min[k] / u.UM:.6g}", f"{result.total_max[k] / u.UM:.6g}",
                        f"{result.charge[k] / u.A_PER_CM2:.6g}",
                        f"{u.kelvin_to_celsius(result.temperature[k]):.5g}"])


def write_profile_csv(path, result, index: int | None = None) -> None:
    """한 시각의 요소별 값 (s: 측면도 단면 둘레 좌표). index 는 profile_time 의 순번 (None = 최종)."""
    m = result.mesh
    if index is None:
        f = result.final
        time, total, barrier = f.time, f.total, f.barrier
        porous, current_density = f.porous, f.current_density
    else:
        time, total = result.profile_time[index], result.profile_total[index]
        current_density = result.profile_current_density[index]
        barrier = porous = None
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["s_cm", "face", "x_cm", "z_cm", "area_cm2", "total_um"]
        if barrier is not None:
            header += ["barrier_nm", "porous_um"]
        w.writerow(header + ["current_density_A_dm2", "time_min"])
        for i in range(m.size):
            face = "전체" if m.face[i] == LUMPED else FACE_NAMES[m.face[i]]
            row = [f"{m.s[i] / u.CM:.5g}", face, f"{m.x[i] / u.CM:.5g}", f"{m.z[i] / u.CM:.5g}",
                   f"{m.area[i] / u.CM ** 2:.5g}", f"{total[i] / u.UM:.6g}"]
            if barrier is not None:
                row += [f"{barrier[i] / u.NM:.6g}", f"{porous[i] / u.UM:.6g}"]
            w.writerow(row + [f"{current_density[i] / u.A_PER_DM2:.6g}", f"{time / u.MINUTE:.6g}"])
