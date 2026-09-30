"""Test helpers for driving Pipecat observers without a live pipeline."""

from __future__ import annotations

import asyncio

from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class DummyProcessor(FrameProcessor):
    """Minimal processor used as FramePushed source/destination."""

    def __init__(self, name: str = "dummy"):
        super().__init__()
        self._name = name

    @property
    def name(self) -> str:
        return self._name


class ControllableClock:
    """Deterministic time source for observers that accept ``time_source``."""

    def __init__(self, start: float = 0.0):
        self.t = start

    def __call__(self) -> float:
        return self.t


async def drain_observer_events(observer: BaseObserver) -> None:
    """Wait until async event-handler tasks registered on the observer finish."""
    while observer._event_tasks:
        tasks = [task for _, task in list(observer._event_tasks)]
        if not tasks:
            break
        await asyncio.gather(*tasks, return_exceptions=True)


async def push_frame(
    observer: BaseObserver,
    frame,
    *,
    source: FrameProcessor | None = None,
    destination: FrameProcessor | None = None,
    timestamp: int = 1,
    clock: ControllableClock | None = None,
    now: float | None = None,
) -> None:
    """Push one frame through an observer and drain resulting event tasks."""
    if clock is not None and now is not None:
        clock.t = now
    src = source or DummyProcessor("source")
    dst = destination or DummyProcessor("destination")
    await observer.on_push_frame(
        FramePushed(
            source=src,
            destination=dst,
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
            timestamp=timestamp,
        )
    )
    await drain_observer_events(observer)
