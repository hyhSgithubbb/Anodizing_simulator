import math
import os
from dataclasses import replace

import pytest

# GUI 테스트: 디스플레이가 없으면 offscreen 플랫폼을 쓴다 (xvfb-run 으로 실행하면 실제 X 서버 사용)
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from anodsim.core import units as u
from anodsim.core.geometry import Geometry
from anodsim.core.power_supply import PowerMode, PowerSetting
from anodsim.core.presets import PresetDB
from anodsim.core.state import ElectrolyteRecipe, ProcessConfig


@pytest.fixture(scope="session")
def presets() -> PresetDB:
    return PresetDB.default()


def make_config(metal="Al", preset="al_sulfuric_typeII", cathode="pb", mode="CV", voltage=15.0,
                current=math.inf, ramp=0.0, minutes=10.0, temp_c=20.0, geometry=None, concentration=None):
    return ProcessConfig(
        metal=metal,
        electrolyte=ElectrolyteRecipe(preset, concentration),
        cathode_material=cathode,
        power=PowerSetting(PowerMode(mode), voltage, current, ramp),
        geometry=geometry or Geometry.default(),
        bath_temperature=u.celsius_to_kelvin(temp_c),
        duration=minutes * u.MINUTE,
    )


def ideal_presets(db: PresetDB, preset_id: str, **changes) -> PresetDB:
    """η=1, P=0, 용해·박화 없음 등으로 바꾼 프리셋 DB (보존 법칙 검증용)."""
    preset = db.electrolyte(preset_id)
    if preset.dissolution is not None:
        changes.setdefault("dissolution", replace(preset.dissolution, surface_rate=0.0, pore_wall_rate=0.0,
                                                  barrier_thinning=0.0))
    modified = replace(preset, **changes)
    return PresetDB(db.metals, {**db.electrolytes, preset_id: modified}, db.cathodes)


@pytest.fixture
def window(qtbot):
    """GUI 메인 창 (pytest-qt). GUI 테스트 파일들이 함께 쓴다."""
    from anodsim.gui.app import configure_pyqtgraph
    from anodsim.gui.main_window import MainWindow

    configure_pyqtgraph()
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    return w
