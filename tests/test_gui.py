"""GUI 스모크 테스트 (pytest-qt). 디스플레이가 없으면 offscreen 플랫폼으로 실행된다."""

import pytest

pytest.importorskip("pytestqt")

from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from anodsim.core.power_supply import PowerMode
from anodsim.gui.app import configure_pyqtgraph
from anodsim.gui.main_window import MainWindow


@pytest.fixture
def window(qtbot):
    configure_pyqtgraph()
    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    qtbot.waitExposed(w)
    return w


def test_window_starts_with_valid_defaults(window):
    assert window.run_button.isEnabled()
    assert window.process.metal_symbol() == "Al"
    assert window.electrolyte.recipe().preset_id == "al_sulfuric_typeII"
    assert window.view_tabs.currentIndex() == 0          # 측면도(2D)가 기본
    assert window.view_tabs.tabText(0).startswith("측면도")
    assert len(window.side_view._items) > 5


def test_switching_metal_filters_presets_and_cathode(window):
    window.process.metal.setCurrentIndex(window.process.metal.findData("Ti"))
    ids = [window.electrolyte.preset.itemData(i) for i in range(window.electrolyte.preset.count())]
    assert set(ids) == {"ti_color_acid", "ti_nanotube_eg"}
    window.electrolyte.preset.setCurrentIndex(window.electrolyte.preset.findData("ti_nanotube_eg"))
    cathode = window.presets.cathode(window.cathode.material_id())
    assert cathode.is_compatible(window.presets.electrolyte("ti_nanotube_eg").tags)
    assert window.process.voltage.value() == pytest.approx(60.0)


def test_invalid_geometry_blocks_run(window):
    window.geometry.anode.x.setValue(39.99)
    assert not window.run_button.isEnabled()
    assert "전해조 밖" in window.messages.text()
    window.geometry.reset()
    assert window.run_button.isEnabled()


def test_run_simulation_end_to_end(window, qtbot):
    window.process.mode.setCurrentIndex(window.process.mode.findData(PowerMode.CV))
    window.process.duration.setValue(3.0)
    window.run_button.click()
    qtbot.waitUntil(lambda: window._thread is None, timeout=30000)
    assert window.results.thickness.curves
    assert window.results._result.time[-1] == pytest.approx(180.0)
    assert "최종" in window.results.readout.text()


def test_3d_view_rotates_with_mouse_drag(window, qtbot):
    view = window.view3d.view
    if view is None:
        pytest.skip("OpenGL 을 사용할 수 없는 환경")
    window.view_tabs.setCurrentIndex(1)
    qtbot.wait(50)
    before = view.opts["azimuth"], view.opts["elevation"]
    start, button = QPointF(200, 150), Qt.MouseButton.LeftButton
    app = QApplication.instance()
    app.sendEvent(view, QMouseEvent(QEvent.Type.MouseButtonPress, start, view.mapToGlobal(start), button, button,
                                    Qt.KeyboardModifier.NoModifier))
    for k in range(1, 11):
        p = QPointF(200 + 8 * k, 150 - 3 * k)
        app.sendEvent(view, QMouseEvent(QEvent.Type.MouseMove, p, view.mapToGlobal(p), Qt.MouseButton.NoButton,
                                        button, Qt.KeyboardModifier.NoModifier))
    after = view.opts["azimuth"], view.opts["elevation"]
    assert after != before
    assert len(view._labels) == 6    # 양극 1 + 음극 2 + 축 3
