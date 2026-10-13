"""P4 GUI: 전원 프로그램 편집, 열수지, Ti 간섭색, 프로젝트 저장·열기, 내보내기."""

import csv
import json

import numpy as np
import pytest

pytest.importorskip("pytestqt")

from anodsim.core.power_supply import PowerMode
from anodsim.gui.panels.process_panel import PROGRAM


def _run(window, qtbot, timeout=60000):
    window.run_button.click()
    qtbot.waitUntil(lambda: window._thread is None, timeout=timeout)
    return window.results._result


def _cell(editor, row, column):
    return editor.table.cellWidget(row, column)


def test_program_editor_and_run(window, qtbot):
    process = window.process
    process.mode.setCurrentIndex(process.mode.findData(PowerMode.CV))
    process.voltage.setValue(15.0)
    process.unlimited.setChecked(True)
    process.duration.setValue(4.0)
    process.mode.setCurrentIndex(process.mode.findData(PROGRAM))
    editor = process.program
    assert editor.table.rowCount() == 1                       # 지금 단일 설정이 첫 단계로
    assert _cell(editor, 0, 1).value() == pytest.approx(15.0)
    assert _cell(editor, 0, 2).specialValueText() == _cell(editor, 0, 2).text()   # 전류 '제한 없음'
    assert _cell(editor, 0, 3).value() == pytest.approx(4.0)
    assert not process.duration.isEnabled()
    editor._add_after_selection()
    _cell(editor, 1, 1).setValue(20.0)
    _cell(editor, 1, 3).setValue(2.0)
    _cell(editor, 0, 3).setValue(2.0)
    assert process.duration.value() == pytest.approx(4.0)     # 2 + 2 min
    program = process.power_setting()
    assert [s.voltage for s in program.steps] == [15.0, 20.0]
    result = _run(window, qtbot)
    assert set(result.step) == {0, 1}
    assert "단계" in window.results.readout.text()
    k = np.searchsorted(result.time, 230.0)
    assert result.voltage[k] == pytest.approx(20.0)


def test_pulse_program_with_repeat(window, qtbot):
    process = window.process
    process.mode.setCurrentIndex(process.mode.findData(PROGRAM))
    editor = process.program
    _cell(editor, 0, 0).setCurrentIndex(_cell(editor, 0, 0).findData(PowerMode.CV))
    _cell(editor, 0, 3).setValue(0.5)
    editor._add_after_selection()
    _cell(editor, 1, 1).setValue(0.0)                          # 0 V = 꺼짐
    _cell(editor, 1, 4).setValue(0.0)
    editor.repeat.setValue(3)
    assert process.duration.value() == pytest.approx(3.0)
    result = _run(window, qtbot)
    assert "OFF" in result.control and result.step.max() == 5


def test_thermal_run_shows_temperature(window, qtbot):
    process = window.process
    process.duration.setValue(5.0)
    process.thermal_box.setChecked(True)
    process.cooling.setValue(0.0)
    assert "열용량" in process.thermal_info.text() and "단열" in process.thermal_info.text()
    config = window.build_config()
    assert config.thermal is not None and config.thermal.cooling_ua == 0.0
    result = _run(window, qtbot)
    assert result.thermal and result.temperature[-1] > result.temperature[0]
    assert window.results.temperature.isVisible()
    assert "욕" in window.results.readout.text()
    process.thermal_box.setChecked(False)
    process.duration.setValue(1.0)
    _run(window, qtbot)
    assert not window.results.temperature.isVisible()


def test_ti_interference_colour(window, qtbot):
    process = window.process
    process.metal.setCurrentIndex(process.metal.findData("Ti"))
    window.electrolyte.preset.setCurrentIndex(window.electrolyte.preset.findData("ti_color_acid"))
    assert "예상 간섭색" in process.colour_preview.text()
    process.duration.setValue(2.0)
    _run(window, qtbot)
    panel = window.side_panel
    assert panel.colour_mode.isEnabled() and panel.interference
    strips = window.side_view._strips
    assert strips and strips[0][0].image.ndim == 3                # RGB 색 띠
    assert "간섭색" in window.results.readout.text()
    panel.colour_mode.setCurrentIndex(0)                          # 두께 색상표로 되돌리기
    assert window.side_view._strips[0][0].image.ndim == 2
    # Al 다공성 결과에서는 간섭색을 고를 수 없다
    process.metal.setCurrentIndex(process.metal.findData("Al"))
    process.duration.setValue(1.0)
    _run(window, qtbot)
    assert not panel.colour_mode.isEnabled() and not panel.interference


def test_project_save_and_open(window, qtbot, tmp_path, monkeypatch):
    process = window.process
    process.metal.setCurrentIndex(process.metal.findData("Ti"))
    window.electrolyte.preset.setCurrentIndex(window.electrolyte.preset.findData("ti_color_acid"))
    process.mode.setCurrentIndex(process.mode.findData(PowerMode.CV))
    process.voltage.setValue(42.0)
    process.duration.setValue(7.5)
    process.thermal_box.setChecked(True)
    process.cooling.setValue(12.0)
    process.fidelity.setCurrentIndex(process.fidelity.findData("L2"))
    window.geometry.anode.x.setValue(21.5)
    window.geometry.cathode2.setChecked(False)
    saved = window.build_config()

    path = tmp_path / "ti.anod.json"
    monkeypatch.setattr(window, "_ask_save_path", lambda *a: str(path))
    assert window.save_project_as()
    assert path.exists() and window.windowTitle().startswith("ti.anod.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["power"]["voltage_V"] == 42.0 and data["fidelity"] == "L2"

    window.new_project()                                          # 기본값으로
    assert window.process.metal_symbol() == "Al" and window.geometry.cathode2.isChecked()
    monkeypatch.setattr(window, "_ask_open_path", lambda: str(path))
    assert window.open_project()
    loaded = window.build_config()
    assert loaded == saved
    assert not window.windowTitle().startswith("*")
    process.voltage.setValue(43.0)                                # 고치면 제목에 * 표시
    assert window.windowTitle().startswith("*")


def test_open_bad_project_shows_message(window, tmp_path, monkeypatch):
    path = tmp_path / "bad.anod.json"
    path.write_text("{}", encoding="utf-8")
    messages = []
    monkeypatch.setattr("anodsim.gui.main_window.QMessageBox.warning", lambda *a: messages.append(a[2]))
    assert not window.open_project(str(path))
    assert messages and "프로젝트" in messages[0]


def test_failed_open_keeps_current_inputs(window, tmp_path, monkeypatch):
    """파일은 읽혔지만 화면에 넣다 실패하면 (음극 3장) 반쯤 바뀐 입력을 되돌린다."""
    from anodsim.io import config_to_dict
    before = window.build_config()
    data = config_to_dict(before)
    data["metal"], data["electrolyte"] = "Ti", {"preset": "ti_color_acid", "concentration": None}
    data["cathode"] = "ti"
    data["geometry"]["cathodes"] *= 2                             # 음극 4장 — 화면은 2장까지
    path = tmp_path / "four.anod.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    messages = []
    monkeypatch.setattr("anodsim.gui.main_window.QMessageBox.warning", lambda *a: messages.append(a[2]))
    assert not window.open_project(str(path))
    assert messages and "2장" in messages[0]
    assert window.build_config() == before and window.process.metal_symbol() == "Al"
    assert not window.windowTitle().startswith("*")


def test_exports_after_run(window, qtbot, tmp_path, monkeypatch):
    assert not window.export_actions["series"].isEnabled()
    window.process.duration.setValue(1.0)
    _run(window, qtbot)
    assert window.export_actions["series"].isEnabled()
    targets = iter([tmp_path / "s.csv", tmp_path / "p.csv", tmp_path / "r.png", tmp_path / "side.png"])
    monkeypatch.setattr(window, "_ask_save_path", lambda *a: str(next(targets)))
    assert window.export_timeseries() and window.export_profile()
    assert window.export_png("results") and window.export_png("side")
    rows = list(csv.DictReader((tmp_path / "s.csv").open(encoding="utf-8")))
    assert len(rows) == window.results._result.time.size
    assert (tmp_path / "r.png").stat().st_size > 1000 and (tmp_path / "side.png").stat().st_size > 1000
