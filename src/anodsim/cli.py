"""명령줄 실행 (GUI 없이 L0 계산).

예)
    python -m anodsim.cli --list
    python -m anodsim.cli --metal Al --electrolyte al_sulfuric_typeII --mode CCCV --voltage 15 --current 5 --minutes 60
    python -m anodsim.cli --metal Ti --electrolyte ti_color_acid --mode CV --voltage 50 --minutes 10 --csv ti.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys

import numpy as np

from .core import units as u
from .core.geometry import Geometry
from .core.limits import FLAG_MESSAGES, WARNING_FLAGS
from .core.power_supply import PowerMode, PowerSetting
from .core.presets import PresetDB
from .core.simulator import ConfigError, Simulator
from .core.state import ElectrolyteRecipe, ProcessConfig

DEFAULT_ELECTROLYTE = {"Al": "al_sulfuric_typeII", "Ti": "ti_color_acid"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anodsim-cli", description="아노다이징 산화막 성장 L0 시뮬레이션")
    p.add_argument("--list", action="store_true", help="프리셋 목록을 출력하고 끝낸다")
    p.add_argument("--metal", choices=["Al", "Ti"], default="Al")
    p.add_argument("--electrolyte", help="전해질 프리셋 id (기본: 금속별 대표 프리셋)")
    p.add_argument("--concentration", type=float, help="농도 (프리셋 단위, 기본: 프리셋 기준값)")
    p.add_argument("--cathode", help="음극재 id (기본: 전해질 추천 음극)")
    p.add_argument("--mode", choices=[m.value for m in PowerMode], default="CCCV")
    p.add_argument("--voltage", type=float, help="전압 설정/제한 [V] (CC 기본: 프리셋 전압 범위 상한)")
    p.add_argument("--current", type=float, help="전류 설정/제한 [A] (CV 에서 생략하면 제한 없음)")
    p.add_argument("--ramp", type=float, default=30.0, help="소프트 스타트 램프 시간 [s]")
    p.add_argument("--minutes", type=float, default=60.0, help="공정 시간 [min]")
    p.add_argument("--temperature", type=float, help="욕 온도 [°C] (기본: 프리셋 기준 온도)")
    p.add_argument("--csv", help="시간 이력을 CSV 로 저장할 경로")
    return p


def list_presets(db: PresetDB) -> None:
    print("금속:")
    for m in db.metals.values():
        print(f"  {m.symbol:3s} {m.name}")
    print("\n전해질:")
    for e in db.electrolytes.values():
        print(f"  {e.id:20s} [{e.metal}, {e.regime:7s}] {e.name}")
    print("\n음극재:")
    for c in db.cathodes.values():
        print(f"  {c.id:9s} {c.name}")


def config_from_args(args, db: PresetDB) -> ProcessConfig:
    preset = db.electrolyte(args.electrolyte or DEFAULT_ELECTROLYTE[args.metal])
    mode = PowerMode(args.mode)
    voltage = args.voltage if args.voltage is not None else preset.voltage_range.high
    if args.current is not None:
        current = args.current
    elif mode is PowerMode.CV:
        current = math.inf
    else:
        raise SystemExit("CC / CCCV 모드에는 --current 가 필요합니다.")
    t_c = args.temperature if args.temperature is not None else u.kelvin_to_celsius(preset.temperature_ref)
    return ProcessConfig(
        metal=args.metal,
        electrolyte=ElectrolyteRecipe(preset.id, args.concentration),
        cathode_material=args.cathode or preset.recommended_cathodes[0],
        power=PowerSetting(mode, voltage, current, args.ramp),
        geometry=Geometry.default(),
        bath_temperature=u.celsius_to_kelvin(t_c),
        duration=args.minutes * u.MINUTE,
    )


def write_csv(path: str, result) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_min", "voltage_V", "current_A", "control", "current_density_A_dm2",
                    "film_voltage_V", "barrier_nm", "porous_um", "total_um", "charge_C_cm2"])
        for k in range(result.time.size):
            w.writerow([f"{result.time[k] / u.MINUTE:.6g}", f"{result.voltage[k]:.6g}", f"{result.current[k]:.6g}",
                        result.control[k], f"{result.current_density[k] / u.A_PER_DM2:.6g}",
                        f"{result.film_voltage[k]:.6g}", f"{result.barrier[k] / u.NM:.6g}",
                        f"{result.porous[k] / u.UM:.6g}", f"{result.total[k] / u.UM:.6g}",
                        f"{result.charge[k] / u.A_PER_CM2:.6g}"])


def print_result(sim: Simulator, result) -> None:
    cfg = sim.config
    print(f"{sim.metal.symbol} / {sim.preset.name} / 음극 {sim.cathode.name}")
    print(f"양극 면적 {sim.areas.sum() / u.CM ** 2:.1f} cm², 전해질 저항 {sim.resistance[0, 0] * 1e3:.1f} mΩ, "
          f"κ = {sim.props.conductivity:.3g} S/m, 욕 온도 {u.kelvin_to_celsius(cfg.bath_temperature):.1f} °C")
    print(f"{'t [min]':>8} {'U [V]':>8} {'I [A]':>8} {'j [A/dm²]':>10} {'U_f [V]':>8} "
          f"{'δ_b [nm]':>9} {'h_p [μm]':>9} {'h [μm]':>9}  제어")
    for t_target in np.linspace(0.0, cfg.duration, 11):
        k = min(int(np.searchsorted(result.time, t_target)), result.time.size - 1)
        print(f"{result.time[k] / u.MINUTE:8.2f} {result.voltage[k]:8.2f} {result.current[k]:8.3f} "
              f"{result.current_density[k] / u.A_PER_DM2:10.3f} {result.film_voltage[k]:8.2f} "
              f"{result.barrier[k] / u.NM:9.2f} {result.porous[k] / u.UM:9.3f} {result.total[k] / u.UM:9.3f}  "
              f"{result.control[k]}")
    for w in result.warnings:
        print(f"[주의] {w}")
    for flag in WARNING_FLAGS:
        if flag in result.flags:
            print(f"[경고] {FLAG_MESSAGES[flag]}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    db = PresetDB.default()
    if args.list:
        list_presets(db)
        return 0
    try:
        sim = Simulator(config_from_args(args, db), db)
    except ConfigError as e:
        for err in e.errors:
            print(f"[설정 오류] {err}", file=sys.stderr)
        return 2
    result = sim.run()
    print_result(sim, result)
    if args.csv:
        write_csv(args.csv, result)
        print(f"CSV 저장: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
