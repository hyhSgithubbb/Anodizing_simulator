"""프로젝트 파일 저장·불러오기와 결과 CSV 내보내기 (anodsim.io)."""

import csv
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from anodsim import cli
from anodsim.core import units as u
from anodsim.core.geometry import Geometry
from anodsim.core.power_supply import PowerMode, PowerProgram, PowerStep
from anodsim.core.simulator import Simulator
from anodsim.core.thermal import ThermalSetting
from anodsim.io import (ProjectError, config_from_dict, config_to_dict, load_project, save_project,
                        write_profile_csv, write_timeseries_csv)

from conftest import make_config


def _rich_config():
    g = Geometry.default()
    program = PowerProgram((PowerStep(PowerMode.CC, 40.0, 2.0, 300.0, 20.0),
                            PowerStep(PowerMode.CV, 25.0, math.inf, 600.0, 60.0)), repeat=2)
    return replace(make_config(metal="Ti", preset="ti_color_acid", cathode="ti", minutes=30.0, temp_c=25.0),
                   power=program, fidelity="L2", elements=80, external_resistance=0.01,
                   geometry=replace(g, anode=replace(g.anode, x_center=0.213, y_center=0.1), cathodes=g.cathodes[:1]),
                   thermal=ThermalSetting(15.0, u.celsius_to_kelvin(18.0)))


def _assert_same(a, b):
    """단위 환산 반올림(1e-12) 안에서 같은 설정."""
    da, db = config_to_dict(a), config_to_dict(b)
    assert da == db
    assert b.duration == pytest.approx(a.duration)
    assert b.geometry.anode.x_center == pytest.approx(a.geometry.anode.x_center)


def test_round_trip_preserves_everything(tmp_path):
    config = _rich_config()
    path = save_project(tmp_path / "ti.anod.json", config)
    loaded = load_project(path)
    _assert_same(config, loaded)
    assert loaded.power.repeat == 2 and math.isinf(loaded.power.steps[1].current)
    assert loaded.thermal.coolant_temperature == pytest.approx(u.celsius_to_kelvin(18.0))
    assert loaded.geometry.anode.y_center == pytest.approx(0.1)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["power"]["program"]["steps"][1]["current_A"] is None      # 제한 없음 = null
    assert data["geometry"]["anode"]["x_center_cm"] == 21.3                 # 반올림 찌꺼기 없이


def test_round_trip_simple_setting():
    config = make_config(mode="CV", current=math.inf, ramp=12.0)
    loaded = config_from_dict(config_to_dict(config))
    assert loaded.power == config.power and loaded.thermal is None
    _assert_same(config, loaded)


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.update(format="something"), "프로젝트 파일이 아닙니다"),
    (lambda d: d.update(version=99), "버전"),
    (lambda d: d.pop("metal"), "'metal'"),
    (lambda d: d["power"].update(mode="DC"), "mode"),
    (lambda d: d["geometry"]["tank"].update(width_cm="넓게"), "숫자"),
    (lambda d: d["geometry"].update(cathodes={}), "목록"),
])
def test_bad_project_files_are_reported(mutate, message):
    data = config_to_dict(make_config())
    mutate(data)
    with pytest.raises(ProjectError, match=message):
        config_from_dict(data)


def test_invalid_json_reports_line(tmp_path):
    path = tmp_path / "bad.anod.json"
    path.write_text('{"format": "anodsim-project",\n "version": 1,\n oops}', encoding="utf-8")
    with pytest.raises(ProjectError, match="3행"):
        load_project(path)


def test_csv_exports(tmp_path, presets):
    config = replace(make_config(minutes=2.0, preset="al_sulfuric_typeII"), fidelity="L1",
                     thermal=ThermalSetting(5.0, u.celsius_to_kelvin(20.0)))
    result = Simulator(config, presets).run()
    series = tmp_path / "series.csv"
    write_timeseries_csv(series, result)
    rows = list(csv.DictReader(series.open(encoding="utf-8")))
    assert len(rows) == result.time.size
    assert {"bath_temperature_C", "step", "total_min_um"} <= set(rows[0])
    assert float(rows[-1]["time_min"]) == pytest.approx(2.0)
    assert float(rows[-1]["bath_temperature_C"]) > 20.0
    profile = tmp_path / "profile.csv"
    write_profile_csv(profile, result)
    rows = list(csv.DictReader(profile.open(encoding="utf-8")))
    assert len(rows) == result.mesh.size and "barrier_nm" in rows[0]
    mid = tmp_path / "mid.csv"
    k = result.profile_time.size // 2
    write_profile_csv(mid, result, index=k)
    rows = list(csv.DictReader(mid.open(encoding="utf-8")))
    assert float(rows[0]["time_min"]) == pytest.approx(result.profile_time[k] / u.MINUTE, rel=1e-5)
    assert np.isclose(float(rows[3]["total_um"]), result.profile_total[k][3] / u.UM, rtol=1e-5)


def test_cli_project_round_trip(tmp_path, capsys):
    project = tmp_path / "hard.anod.json"
    args = ["--electrolyte", "al_sulfuric_hard", "--mode", "CCCV", "--voltage", "60", "--current", "10",
            "--temperature", "0", "--minutes", "3", "--cooling", "20", "--fidelity", "L0"]
    assert cli.main(args + ["--save-project", str(project)]) == 0
    first = capsys.readouterr().out
    assert "욕 온도 0.0 →" in first and project.exists()
    assert cli.main(["--project", str(project)]) == 0
    second = capsys.readouterr().out
    assert first.split("\n", 1)[1] == second                       # 저장 안내 한 줄 말고는 같은 결과


def test_cli_reports_bad_project(tmp_path, capsys):
    path = tmp_path / "x.anod.json"
    path.write_text("{}", encoding="utf-8")
    assert cli.main(["--project", str(path)]) == 2
    assert "프로젝트 오류" in capsys.readouterr().err


def test_cli_prints_ti_interference_colour(capsys):
    assert cli.main(["--metal", "Ti", "--mode", "CV", "--voltage", "20", "--minutes", "3", "--fidelity", "L0"]) == 0
    assert "예상 간섭색" in capsys.readouterr().out
