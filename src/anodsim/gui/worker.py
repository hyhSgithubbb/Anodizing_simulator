"""시뮬레이션을 별도 스레드에서 실행 (GUI 가 멈추지 않도록).

- 실시간 갱신은 '당겨 가기(pull)' 방식이다. 워커는 매 스텝 최신 스냅샷의 참조만 바꿔 두고,
  GUI 가 타이머(최대 30 Hz)로 take_snapshot() 을 불러 가져간다. 그리기가 느려도 신호가
  이벤트 큐에 쌓이지 않으므로 마우스·키보드 입력이 밀리지 않는다.
- 배속: 공정 시간 / 실제 시간. math.inf 이면 최대 속도로 계산한다.
  실행 중에도 set_speed 로 바꿀 수 있다 (바꾼 시점부터 새 배속으로 맞춘다).
"""

from __future__ import annotations

import math
import time

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

_SLEEP_CHUNK = 0.02   # s — 배속 대기 중에도 중지 요청에 빨리 반응하도록


class SimulationWorker(QObject):
    finished = pyqtSignal(object, bool)     # (SimulationResult, 중지 여부)
    failed = pyqtSignal(str)

    def __init__(self, simulator, speed: float = math.inf):
        super().__init__()
        self.simulator = simulator
        self._stop_requested = False
        self._speed = speed
        self._anchor = None      # (실제 시각, 공정 시각) — 배속 기준점
        self._latest = None      # 가장 최근 스냅샷 (참조 대입은 스레드 사이에서 원자적)
        self._taken = None

    @property
    def speed(self) -> float:
        return self._speed

    def request_stop(self) -> None:
        self._stop_requested = True

    def set_speed(self, speed: float) -> None:
        """배속 변경 (GUI 스레드에서 호출해도 안전: 값 대입만 한다)."""
        self._anchor = None
        self._speed = speed

    def take_snapshot(self):
        """GUI 스레드용: 지난번에 가져간 뒤 새 스냅샷이 있으면 돌려주고, 없으면 None."""
        snap = self._latest
        if snap is None or snap is self._taken:
            return None
        self._taken = snap
        return snap

    @property
    def progress(self) -> float:
        snap = self._latest
        return 0.0 if snap is None else snap.time / self.simulator.config.duration

    @pyqtSlot()
    def run(self) -> None:
        try:
            self._latest = self.simulator.snapshot()
            result = self.simulator.run(callback=self._on_step)
        except Exception as exc:  # 계산 오류는 GUI 에 메시지로 보여준다
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self._latest = result.final
        self.finished.emit(result, self._stop_requested)

    def _on_step(self, snapshot) -> bool:
        if self._stop_requested:
            return False
        self._latest = snapshot
        self._pace(snapshot.time)
        return not self._stop_requested

    def _pace(self, sim_time: float) -> None:
        """배속에 맞춰 기다린다: 실제 경과 시간 ≥ 공정 경과 시간 / 배속."""
        speed = self._speed
        if not math.isfinite(speed):
            return
        anchor = self._anchor
        if anchor is None:
            self._anchor = anchor = (time.monotonic(), sim_time)
        target = anchor[0] + (sim_time - anchor[1]) / speed
        while not self._stop_requested and self._speed == speed:
            remaining = target - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, _SLEEP_CHUNK))
