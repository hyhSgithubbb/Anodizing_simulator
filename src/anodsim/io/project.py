"""프로젝트 파일 (*.anod.json) — 공정 설정(ProcessConfig) 저장·불러오기.

사람이 읽고 고칠 수 있도록 단위를 키 이름에 붙인 JSON 이다 (cm, mm, °C, min, s, V, A, W/K).
'제한 없음'(무한대) 전류는 null 로 쓴다. 형식이 바뀌면 version 을 올리고 이전 버전 읽기를 유지한다.

    {
      "format": "anodsim-project", "version": 1,
      "metal": "Ti",
      "electrolyte": {"preset": "ti_color_acid", "concentration": null},
      "cathode": "ti",
      "power": {"mode": "CV", "voltage_V": 50, "current_A": null, "ramp_s": 10}
               또는 {"program": {"repeat": 1, "steps": [{"mode": "CV", "voltage_V": 15, "current_A": null,
                                                        "duration_min": 10, "ramp_s": 30}, …]}},
      "geometry": {"tank": {…_cm}, "anode": {…}, "cathodes": [{…}, …]},
      "bath_temperature_C": 20, "duration_min": 10,
      "fidelity": "L1", "elements": 120,
      "offset_voltage_V": 0, "external_resistance_ohm": 0,
      "thermal": null 또는 {"cooling_W_per_K": 10, "coolant_temperature_C": 20}
    }
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..core import units as u
from ..core.geometry import Geometry, PlateElectrode, Tank
from ..core.power_supply import PowerMode, PowerProgram, PowerSetting, PowerStep
from ..core.state import ElectrolyteRecipe, ProcessConfig
from ..core.thermal import ThermalSetting

FORMAT = "anodsim-project"
VERSION = 1
PROJECT_SUFFIX = ".anod.json"


class ProjectError(ValueError):
    pass


def _num(value: float) -> float:
    """단위 환산의 반올림 찌꺼기(20.000000000000004)를 지워 사람이 읽기 좋게."""
    return float(f"{value:.12g}")


def _current(value: float) -> float | None:
    return None if math.isinf(value) else _num(value)


# ── 저장 ────────────────────────────────────────────────────────────────────
def _plate(p: PlateElectrode) -> dict:
    return {"x_center_cm": _num(p.x_center / u.CM), "z_bottom_cm": _num(p.z_bottom / u.CM),
            "height_cm": _num(p.height / u.CM), "thickness_mm": _num(p.thickness / u.MM),
            "width_cm": _num(p.width / u.CM),
            "y_center_cm": None if p.y_center is None else _num(p.y_center / u.CM)}


def _power(power) -> dict:
    if isinstance(power, PowerProgram):
        steps = [{"mode": s.mode.value, "voltage_V": _num(s.voltage), "current_A": _current(s.current),
                  "duration_min": _num(s.duration / u.MINUTE), "ramp_s": _num(s.ramp_time)} for s in power.steps]
        return {"program": {"repeat": power.repeat, "steps": steps}}
    return {"mode": power.mode.value, "voltage_V": _num(power.voltage), "current_A": _current(power.current),
            "ramp_s": _num(power.ramp_time)}


def config_to_dict(config: ProcessConfig) -> dict:
    g = config.geometry
    t = g.tank
    thermal = None
    if config.thermal is not None:
        thermal = {"cooling_W_per_K": _num(config.thermal.cooling_ua),
                   "coolant_temperature_C": _num(u.kelvin_to_celsius(config.thermal.coolant_temperature))}
    return {
        "format": FORMAT,
        "version": VERSION,
        "metal": config.metal,
        "electrolyte": {"preset": config.electrolyte.preset_id, "concentration": config.electrolyte.concentration},
        "cathode": config.cathode_material,
        "power": _power(config.power),
        "geometry": {
            "tank": {"width_cm": _num(t.width / u.CM), "depth_cm": _num(t.depth / u.CM),
                     "height_cm": _num(t.height / u.CM), "liquid_level_cm": _num(t.liquid_level / u.CM)},
            "anode": _plate(g.anode),
            "cathodes": [_plate(c) for c in g.cathodes],
        },
        "bath_temperature_C": _num(u.kelvin_to_celsius(config.bath_temperature)),
        "duration_min": _num(config.duration / u.MINUTE),
        "fidelity": config.fidelity,
        "elements": config.elements,
        "offset_voltage_V": _num(config.offset_voltage),
        "external_resistance_ohm": _num(config.external_resistance),
        "thermal": thermal,
    }


def save_project(path, config: ProcessConfig) -> Path:
    path = Path(path)
    path.write_text(json.dumps(config_to_dict(config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# ── 불러오기 ────────────────────────────────────────────────────────────────
def _get(d: dict, key: str, where: str):
    if not isinstance(d, dict) or key not in d:
        raise ProjectError(f"프로젝트 파일에 '{where}{key}' 항목이 없습니다.")
    return d[key]


def _float(d: dict, key: str, where: str, optional: bool = False) -> float | None:
    value = d.get(key) if optional else _get(d, key, where)
    if value is None:
        if optional:
            return None
        raise ProjectError(f"'{where}{key}' 값이 비어 있습니다.")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProjectError(f"'{where}{key}' 는 숫자여야 합니다 (값: {value!r}).")
    return float(value)


def _limit(d: dict, key: str, where: str) -> float:
    """전류: null 이면 제한 없음."""
    value = _float(d, key, where, optional=True)
    return math.inf if value is None else value


def _mode(d: dict, where: str) -> PowerMode:
    mode = _get(d, "mode", where)
    try:
        return PowerMode(mode)
    except ValueError:
        raise ProjectError(f"'{where}mode' 는 CV, CC, CCCV 중 하나여야 합니다 (값: {mode!r}).") from None


def _plate_from(d: dict, where: str) -> PlateElectrode:
    y = _float(d, "y_center_cm", where, optional=True)
    return PlateElectrode(x_center=_float(d, "x_center_cm", where) * u.CM,
                          z_bottom=_float(d, "z_bottom_cm", where) * u.CM,
                          height=_float(d, "height_cm", where) * u.CM,
                          thickness=_float(d, "thickness_mm", where) * u.MM,
                          width=_float(d, "width_cm", where) * u.CM,
                          y_center=None if y is None else y * u.CM)


def _power_from(d: dict):
    if "program" in d:
        program = d["program"]
        steps_data = _get(program, "steps", "power.program.")
        if not isinstance(steps_data, list):
            raise ProjectError("'power.program.steps' 는 목록이어야 합니다.")
        steps = []
        for k, s in enumerate(steps_data, start=1):
            where = f"power.program.steps[{k}]."
            steps.append(PowerStep(_mode(s, where), _float(s, "voltage_V", where), _limit(s, "current_A", where),
                                   _float(s, "duration_min", where) * u.MINUTE, _float(s, "ramp_s", where)))
        repeat = program.get("repeat", 1)
        if not isinstance(repeat, int) or isinstance(repeat, bool):
            raise ProjectError("'power.program.repeat' 는 정수여야 합니다.")
        return PowerProgram(tuple(steps), repeat)
    return PowerSetting(_mode(d, "power."), _float(d, "voltage_V", "power."), _limit(d, "current_A", "power."),
                        _float(d, "ramp_s", "power."))


def config_from_dict(d: dict) -> ProcessConfig:
    if not isinstance(d, dict) or d.get("format") != FORMAT:
        raise ProjectError("아노다이징 시뮬레이터 프로젝트 파일이 아닙니다 (format 항목 확인).")
    version = d.get("version")
    if version != VERSION:
        raise ProjectError(f"지원하지 않는 프로젝트 파일 버전입니다: {version} (지원: {VERSION})")
    electrolyte = _get(d, "electrolyte", "")
    concentration = _float(electrolyte, "concentration", "electrolyte.", optional=True)
    geometry = _get(d, "geometry", "")
    tank = _get(geometry, "tank", "geometry.")
    cathodes = _get(geometry, "cathodes", "geometry.")
    if not isinstance(cathodes, list):
        raise ProjectError("'geometry.cathodes' 는 목록이어야 합니다.")
    thermal_data = d.get("thermal")
    thermal = None
    if thermal_data is not None:
        thermal = ThermalSetting(_float(thermal_data, "cooling_W_per_K", "thermal."),
                                 u.celsius_to_kelvin(_float(thermal_data, "coolant_temperature_C", "thermal.")))
    elements = d.get("elements", 120)
    if not isinstance(elements, int) or isinstance(elements, bool):
        raise ProjectError("'elements' 는 정수여야 합니다.")
    return ProcessConfig(
        metal=str(_get(d, "metal", "")),
        electrolyte=ElectrolyteRecipe(str(_get(electrolyte, "preset", "electrolyte.")), concentration),
        cathode_material=str(_get(d, "cathode", "")),
        power=_power_from(_get(d, "power", "")),
        geometry=Geometry(
            Tank(width=_float(tank, "width_cm", "geometry.tank.") * u.CM,
                 depth=_float(tank, "depth_cm", "geometry.tank.") * u.CM,
                 height=_float(tank, "height_cm", "geometry.tank.") * u.CM,
                 liquid_level=_float(tank, "liquid_level_cm", "geometry.tank.") * u.CM),
            _plate_from(_get(geometry, "anode", "geometry."), "geometry.anode."),
            tuple(_plate_from(c, f"geometry.cathodes[{k}].") for k, c in enumerate(cathodes, start=1)),
        ),
        bath_temperature=u.celsius_to_kelvin(_float(d, "bath_temperature_C", "")),
        duration=_float(d, "duration_min", "") * u.MINUTE,
        fidelity=str(d.get("fidelity", "L1")),
        elements=elements,
        offset_voltage=_float(d, "offset_voltage_V", "", optional=True) or 0.0,
        external_resistance=_float(d, "external_resistance_ohm", "", optional=True) or 0.0,
        thermal=thermal,
    )


def load_project(path) -> ProcessConfig:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProjectError(f"JSON 형식 오류 ({path.name}, {exc.lineno}행): {exc.msg}") from exc
    return config_from_dict(data)
