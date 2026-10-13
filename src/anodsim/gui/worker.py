"""시뮬레이션을 별도 스레드에서 실행 (GUI 가 멈추지 않도록)."""

from __future__ import annotations

import time

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class SimulationWorker(QObject):
    progress = pyqtSignal(float)            # 0–1
    finished = pyqtSignal(object, bool)     # (SimulationResult, 중지 여부)
    failed = pyqtSignal(str)

    def __init__(self, simulator):
        super().__init__()
        self.simulator = simulator
        self._stop_requested = False
        self._last_emit = 0.0

    def request_stop(self) -> None:
        self._stop_requested = True

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self.simulator.run(callback=self._on_step)
        except Exception as exc:  # 계산 오류는 GUI 에 메시지로 보여준다
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.progress.emit(1.0)
        self.finished.emit(result, self._stop_requested)

    def _on_step(self, snapshot) -> bool:
        if self._stop_requested:
            return False
        now = time.monotonic()
        if now - self._last_emit > 0.05:
            self._last_emit = now
            self.progress.emit(snapshot.time / self.simulator.config.duration)
        return True
