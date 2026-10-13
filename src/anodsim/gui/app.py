"""QApplication 생성과 메인 창 실행."""

from __future__ import annotations

import sys

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication

from . import style


def configure_pyqtgraph() -> None:
    pg.setConfigOptions(antialias=True, background=style.SURFACE, foreground=style.INK_SECONDARY)


def run_app(argv=None) -> int:
    configure_pyqtgraph()
    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("Anodizing Simulator")
    from .main_window import MainWindow   # QApplication 이후에 import (OpenGL 위젯)

    window = MainWindow()
    window.show()
    return app.exec()
