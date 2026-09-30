"""Pipecat lifecycle observers and payload mapping helpers.

Observers translate Pipecat events into plain dicts via the ``*_to_dict``
helpers. Those helpers exist so the future SessionMetricsAccumulator can
consume framework-neutral records without importing Pipecat types.
"""

from __future__ import annotations

import inspect
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    TranscriptionFrame,
)
from pipecat.observers.service_metrics_observer import (
    ServiceLatencyRecord,
    ServiceMetricsObserver,
    ServiceUsageRecord,
)
from pipecat.observers.startup_timing_observer import (
    StartupTimingObserver,
    StartupTimingReport,
    TransportTimingReport,
)
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.observers.user_bot_latency_observer import (
    FunctionCallMetrics,
    LatencyBreakdown,
    UserBotLatencyObserver,
)


def latency_record_to_dict(record: ServiceLatencyRecord) -> dict[str, Any]:
    """Normalize ``on_service_latency`` payload fields."""
    return {
        "kind": str(record.kind),
        "processor": record.processor,
        "model": record.model,
        "timestamp": record.timestamp,
        "seconds": record.seconds,
        "ttfb_secs": record.ttfb_secs,
        "leading_silence_secs": record.leading_silence_secs,
        "thinking_time_secs": record.thinking_time_secs,
    }


def usage_record_to_dict(record: ServiceUsageRecord) -> dict[str, Any]:
    """Normalize ``on_service_usage`` payload fields.

    STT audio is ``audio_seconds`` (not ``seconds``). LLM cache reads are
    ``cache_read_input_tokens``. Unused kind-specific fields are left as None.
    """
    return {
        "kind": str(record.kind),
        "processor": record.processor,
        "model": record.model,
        "timestamp": record.timestamp,
        "audio_seconds": record.audio_seconds,
        "characters": record.characters,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "total_tokens": record.total_tokens,
        "cache_read_input_tokens": record.cache_read_input_tokens,
        "cache_creation_input_tokens": record.cache_creation_input_tokens,
        "reasoning_tokens": record.reasoning_tokens,
        "input_audio_tokens": record.input_audio_tokens,
        "output_audio_tokens": record.output_audio_tokens,
        "cache_read_input_audio_tokens": record.cache_read_input_audio_tokens,
    }


def startup_report_to_dict(report: StartupTimingReport) -> dict[str, Any]:
    """Normalize ``on_startup_timing_report`` payload fields."""
    return {
        "start_time": report.start_time,
        "total_duration_secs": report.total_duration_secs,
        "setup_phase_secs": report.setup_phase_secs,
        "start_phase_secs": report.start_phase_secs,
        "processor_timings": [
            {
                "processor_name": timing.processor_name,
                "start_offset_secs": timing.start_offset_secs,
                "duration_secs": timing.duration_secs,
                "setup_duration_secs": timing.setup_duration_secs,
                "start_duration_secs": timing.start_duration_secs,
            }
            for timing in report.processor_timings
        ],
        "warmup": (
            {
                "duration_secs": report.warmup.duration_secs,
                "blocking_duration_secs": report.warmup.blocking_duration_secs,
            }
            if report.warmup is not None
            else None
        ),
    }


def transport_report_to_dict(report: TransportTimingReport) -> dict[str, Any]:
    """Normalize ``on_transport_timing_report`` payload fields."""
    return {
        "start_time": report.start_time,
        "bot_connected_secs": report.bot_connected_secs,
        "client_connected_secs": report.client_connected_secs,
    }


def turn_ended_to_dict(
    turn_count: int,
    duration: float,
    was_interrupted: bool,
) -> dict[str, Any]:
    """Normalize ``on_turn_ended`` handler arguments."""
    return {
        "turn_count": turn_count,
        "duration_secs": duration,
        "was_interrupted": was_interrupted,
        "status": "interrupted" if was_interrupted else "completed",
    }


def latency_measured_to_dict(latency_seconds: float) -> dict[str, Any]:
    """Normalize ``on_latency_measured`` (user silence → bot speech)."""
    return {"latency_seconds": latency_seconds}


def function_call_metrics_to_dict(metrics: FunctionCallMetrics) -> dict[str, Any]:
    """Normalize one ``LatencyBreakdown.function_calls`` entry."""
    return {
        "function_name": metrics.function_name,
        "start_time": metrics.start_time,
        "duration_secs": metrics.duration_secs,
    }


def latency_breakdown_to_dict(breakdown: LatencyBreakdown) -> dict[str, Any]:
    """Normalize ``on_latency_breakdown`` payload fields."""
    return {
        "total_secs": breakdown.total_secs,
        "measured_from": (
            str(breakdown.measured_from) if breakdown.measured_from is not None else None
        ),
        "user_turn_start_time": breakdown.user_turn_start_time,
        "user_turn_secs": breakdown.user_turn_secs,
        "ttfb": [
            {
                "processor": item.processor,
                "model": item.model,
                "start_time": item.start_time,
                "duration_secs": item.duration_secs,
            }
            for item in breakdown.ttfb
        ],
        "text_aggregation": (
            {
                "processor": breakdown.text_aggregation.processor,
                "start_time": breakdown.text_aggregation.start_time,
                "duration_secs": breakdown.text_aggregation.duration_secs,
            }
            if breakdown.text_aggregation is not None
            else None
        ),
        "function_calls": [
            function_call_metrics_to_dict(call) for call in breakdown.function_calls
        ],
        "contributions": [
            {
                "key": item.key,
                "label": item.label,
                "owner": item.owner,
                "owner_kind": str(item.owner_kind),
                "start_time": item.start_time,
                "duration_secs": item.duration_secs,
            }
            for item in breakdown.contributions
        ],
    }


def tool_started_to_dict(frame: FunctionCallInProgressFrame) -> dict[str, Any]:
    """Normalize a tool-start frame for future accumulator wiring."""
    return {
        "event": "started",
        "function_name": frame.function_name,
        "tool_call_id": frame.tool_call_id,
        "cancel_on_interruption": frame.cancel_on_interruption,
        "group_id": frame.group_id,
    }


def tool_result_to_dict(frame: FunctionCallResultFrame) -> dict[str, Any]:
    """Normalize a tool-result frame. ``error`` set means failure."""
    return {
        "event": "result",
        "function_name": frame.function_name,
        "tool_call_id": frame.tool_call_id,
        "error": frame.error,
        "ok": frame.error is None,
    }


def tool_cancelled_to_dict(frame: FunctionCallCancelFrame) -> dict[str, Any]:
    """Normalize a tool-cancel frame."""
    return {
        "event": "cancelled",
        "function_name": frame.function_name,
        "tool_call_id": frame.tool_call_id,
        "run_llm": frame.run_llm,
        "ok": False,
    }


def transcription_to_dict(frame: TranscriptionFrame) -> dict[str, Any]:
    """Normalize an STT transcription frame for utterance counting."""
    return {
        "text": frame.text,
        "user_id": frame.user_id,
        "timestamp": frame.timestamp,
        "finalized": frame.finalized,
        "language": str(frame.language) if frame.language is not None else None,
    }


def setup_observers(
    accumulator: Any | None = None,
    on_checkpoint: Any | None = None,
):
    """Set up observers for startup, service metrics, latency, and turns.

    Returns a list of observers for ``PipelineWorker``. When ``accumulator``
    is provided, observer events are automatically dispatched to the recorder.
    When ``on_checkpoint`` is provided, it is invoked on each completed turn.
    """
    startup_observer = StartupTimingObserver()

    @startup_observer.event_handler("on_startup_timing_report")
    async def on_startup_timing_report(observer, report):
        payload = startup_report_to_dict(report)
        if accumulator is not None:
            accumulator.note_startup(payload)
        logger.info(f"Total startup duration: {payload['total_duration_secs']:.3f}s")
        for timing in payload["processor_timings"]:
            logger.info(f"  {timing['processor_name']}: {timing['duration_secs']:.3f}s")

    @startup_observer.event_handler("on_transport_timing_report")
    async def on_transport_timing_report(observer, report):
        payload = transport_report_to_dict(report)
        if accumulator is not None:
            accumulator.note_transport(payload)
        if payload["client_connected_secs"] is not None:
            logger.info(f"Client connection time: {payload['client_connected_secs']:.3f}s")

    service_observer = ServiceMetricsObserver()

    @service_observer.event_handler("on_service_latency")
    async def on_service_latency(observer, record):
        payload = latency_record_to_dict(record)
        if accumulator is not None:
            accumulator.note_service_latency(payload)
        logger.info(
            f"Service Latency [{payload['processor']}]: "
            f"{payload['kind']} = {payload['seconds']:.3f}s"
        )

    @service_observer.event_handler("on_service_usage")
    async def on_service_usage(observer, record):
        payload = usage_record_to_dict(record)
        if accumulator is not None:
            accumulator.note_service_usage(payload)
        if payload["kind"] == "llm":
            logger.info(
                f"LLM Token Usage [{payload['processor']}]: "
                f"{payload['total_tokens']} tokens"
            )
        elif payload["kind"] == "tts":
            logger.info(
                f"TTS Usage [{payload['processor']}]: {payload['characters']} characters"
            )
        elif payload["kind"] == "stt":
            audio = payload["audio_seconds"] if payload["audio_seconds"] is not None else 0.0
            logger.info(f"STT Usage [{payload['processor']}]: {audio:.2f}s audio")

    latency_observer = UserBotLatencyObserver()

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency_seconds):
        payload = latency_measured_to_dict(latency_seconds)
        if accumulator is not None:
            accumulator.note_user_bot_latency(payload)
        logger.info(f"User-to-bot latency: {payload['latency_seconds']:.3f}s")

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech_latency(observer, latency_seconds):
        if accumulator is not None:
            accumulator.note_first_bot_speech_latency(latency_seconds)
        logger.info(f"First bot speech latency: {latency_seconds:.3f}s")

    @latency_observer.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        payload = latency_breakdown_to_dict(breakdown)
        if accumulator is not None:
            accumulator.note_latency_breakdown(payload)

    turn_observer = TurnTrackingObserver(turn_end_timeout_secs=2.5)

    @turn_observer.event_handler("on_turn_started")
    async def on_turn_started(observer, turn_count):
        if accumulator is not None:
            accumulator.note_turn_started(turn_count)
        logger.info(f"Turn {turn_count} started")

    @turn_observer.event_handler("on_turn_ended")
    async def on_turn_ended(observer, turn_count, duration, was_interrupted):
        payload = turn_ended_to_dict(turn_count, duration, was_interrupted)
        if accumulator is not None:
            accumulator.note_turn_ended(payload)
            checkpoint_fn = on_checkpoint if on_checkpoint is not None else accumulator.checkpoint_after_turn
            if checkpoint_fn is not None:
                try:
                    res = checkpoint_fn(accumulator.summary_dict())
                    if inspect.isawaitable(res):
                        await res
                except Exception as err:
                    logger.warning(f"Turn checkpoint hook failed: {err}")
        logger.info(
            f"Turn {payload['turn_count']} {payload['status']} "
            f"after {payload['duration_secs']:.2f}s"
        )

    return [startup_observer, latency_observer, turn_observer, service_observer]
