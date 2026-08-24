from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from livekit.agents.metrics import (
    EOUMetrics,
    EOTInferenceMetrics,
    InterruptionMetrics,
    LLMMetrics,
    RealtimeModelMetrics,
    STTMetrics,
    TTSMetrics,
    VADMetrics,
)


@dataclass
class _Statistics:
    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, value: Any) -> None:
        if value is None:
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        self.count += 1
        self.total += number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)

    def summary(self, digits: int = 3) -> dict[str, float | int | None]:
        average = self.total / self.count if self.count else None
        return {
            "count": self.count,
            "average": round(average, digits) if average is not None else None,
            "minimum": round(self.minimum, digits) if self.minimum is not None else None,
            "maximum": round(self.maximum, digits) if self.maximum is not None else None,
            "total": round(self.total, digits),
        }


@dataclass
class SessionMetricsAccumulator:
    """Accumulates one running LiveKit agent session without reading exports."""

    metric_event_count: int = 0
    llm_requests: int = 0
    llm_prompt_tokens: int = 0
    llm_cached_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    llm_cancelled: int = 0
    llm_duration: _Statistics = field(default_factory=_Statistics)
    llm_ttft: _Statistics = field(default_factory=_Statistics)
    llm_tokens_per_second: _Statistics = field(default_factory=_Statistics)
    llm_models: Counter[str] = field(default_factory=Counter)
    tts_requests: int = 0
    tts_characters: int = 0
    tts_audio_duration: _Statistics = field(default_factory=_Statistics)
    tts_ttfb: _Statistics = field(default_factory=_Statistics)
    tts_duration: _Statistics = field(default_factory=_Statistics)
    tts_models: Counter[str] = field(default_factory=Counter)
    tts_cancelled: int = 0
    tts_streamed: int = 0
    stt_metric_events: int = 0
    stt_audio_duration: _Statistics = field(default_factory=_Statistics)
    stt_duration: _Statistics = field(default_factory=_Statistics)
    stt_models: Counter[str] = field(default_factory=Counter)
    eou_events: int = 0
    eou_delay: _Statistics = field(default_factory=_Statistics)
    eou_transcription_delay: _Statistics = field(default_factory=_Statistics)
    eou_probability: _Statistics = field(default_factory=_Statistics)
    eou_callback_delay: _Statistics = field(default_factory=_Statistics)
    eot_inference_events: int = 0
    eot_inference_duration: _Statistics = field(default_factory=_Statistics)
    interruption_events: int = 0
    interruption_events_with_interruptions: int = 0
    interruption_probability: _Statistics = field(default_factory=_Statistics)
    interruption_detection_delay: _Statistics = field(default_factory=_Statistics)
    interruption_prediction_duration: _Statistics = field(default_factory=_Statistics)
    interruption_total_duration: _Statistics = field(default_factory=_Statistics)
    interruption_count: int = 0
    backchannel_count: int = 0
    vad_events: int = 0
    vad_inference_duration: _Statistics = field(default_factory=_Statistics)

    def collect(self, metric: Any) -> None:
        self.metric_event_count += 1
        if isinstance(metric, LLMMetrics):
            self._collect_llm(metric)
        elif isinstance(metric, RealtimeModelMetrics):
            self._collect_realtime_model(metric)
        elif isinstance(metric, TTSMetrics):
            self._collect_tts(metric)
        elif isinstance(metric, STTMetrics):
            self._collect_stt(metric)
        elif isinstance(metric, EOUMetrics):
            self._collect_eou(metric)
        elif isinstance(metric, EOTInferenceMetrics):
            self._collect_eot_inference(metric)
        elif isinstance(metric, InterruptionMetrics):
            self._collect_interruption(metric)
        elif isinstance(metric, VADMetrics):
            self._collect_vad(metric)

    def _collect_llm(self, metric: LLMMetrics) -> None:
        self.llm_requests += 1
        self.llm_prompt_tokens += int(getattr(metric, "prompt_tokens", 0) or 0)
        self.llm_cached_prompt_tokens += int(getattr(metric, "prompt_cached_tokens", 0) or 0)
        self.llm_completion_tokens += int(getattr(metric, "completion_tokens", 0) or 0)
        self.llm_total_tokens += int(getattr(metric, "total_tokens", 0) or 0)
        self.llm_cancelled += int(bool(getattr(metric, "cancelled", False)))
        self.llm_duration.add(getattr(metric, "duration", None))
        self.llm_ttft.add(getattr(metric, "ttft", None))
        self.llm_tokens_per_second.add(getattr(metric, "tokens_per_second", None))
        self._count_model(self.llm_models, metric)

    def _collect_realtime_model(self, metric: RealtimeModelMetrics) -> None:
        self.llm_requests += 1
        self.llm_prompt_tokens += int(getattr(metric, "input_tokens", 0) or 0)
        self.llm_completion_tokens += int(getattr(metric, "output_tokens", 0) or 0)
        self.llm_total_tokens += int(getattr(metric, "total_tokens", 0) or 0)
        details = getattr(metric, "input_token_details", None)
        self.llm_cached_prompt_tokens += int(getattr(details, "cached_tokens", 0) or 0)
        self.llm_cancelled += int(bool(getattr(metric, "cancelled", False)))
        self.llm_duration.add(getattr(metric, "duration", None))
        self.llm_ttft.add(getattr(metric, "ttft", None))
        self.llm_tokens_per_second.add(getattr(metric, "tokens_per_second", None))
        self._count_model(self.llm_models, metric)

    def _collect_tts(self, metric: TTSMetrics) -> None:
        self.tts_requests += 1
        self.tts_characters += int(getattr(metric, "characters_count", 0) or 0)
        self.tts_audio_duration.add(getattr(metric, "audio_duration", None))
        self.tts_ttfb.add(getattr(metric, "ttfb", None))
        self.tts_duration.add(getattr(metric, "duration", None))
        self.tts_cancelled += int(bool(getattr(metric, "cancelled", False)))
        self.tts_streamed += int(bool(getattr(metric, "streamed", False)))
        self._count_model(self.tts_models, metric)

    def _collect_stt(self, metric: STTMetrics) -> None:
        self.stt_metric_events += 1
        self.stt_audio_duration.add(getattr(metric, "audio_duration", None))
        self.stt_duration.add(getattr(metric, "duration", None))
        self._count_model(self.stt_models, metric)

    def _collect_eou(self, metric: EOUMetrics) -> None:
        self.eou_events += 1
        self.eou_delay.add(getattr(metric, "end_of_utterance_delay", None))
        self.eou_transcription_delay.add(getattr(metric, "transcription_delay", None))
        self.eou_callback_delay.add(getattr(metric, "on_user_turn_completed_delay", None))

    def _collect_eot_inference(self, metric: EOTInferenceMetrics) -> None:
        self.eot_inference_events += 1
        self.eot_inference_duration.add(getattr(metric, "total_duration", None))

    def _collect_interruption(self, metric: InterruptionMetrics) -> None:
        self.interruption_events += 1
        self.interruption_detection_delay.add(getattr(metric, "detection_delay", None))
        self.interruption_prediction_duration.add(getattr(metric, "prediction_duration", None))
        self.interruption_total_duration.add(getattr(metric, "total_duration", None))
        interruptions = int(getattr(metric, "num_interruptions", 0) or 0)
        self.interruption_count += interruptions
        self.interruption_events_with_interruptions += int(interruptions > 0)
        self.backchannel_count += int(getattr(metric, "num_backchannels", 0) or 0)

    def _collect_vad(self, metric: VADMetrics) -> None:
        self.vad_events += 1
        self.vad_inference_duration.add(getattr(metric, "inference_duration_total", None))

    @staticmethod
    def _count_model(counter: Counter[str], metric: Any) -> None:
        metadata = getattr(metric, "metadata", None)
        model = getattr(metric, "label", None)
        if isinstance(metadata, dict):
            model = metadata.get("model_name") or model
        if model:
            counter[str(model)] += 1

    @staticmethod
    def _count_value(counter: Counter[str], value: Any) -> None:
        if value:
            counter[str(value)] += 1

    @staticmethod
    def _percentage(part: int, whole: int) -> float | None:
        return round(part * 100 / whole, 2) if whole else None

    def summary(self) -> dict[str, Any]:
        total_prompt_tokens = self.llm_prompt_tokens
        uncached_prompt_tokens = max(
            total_prompt_tokens - self.llm_cached_prompt_tokens, 0
        )
        return {
            "events": {"metric_event_count": self.metric_event_count},
            "llm": {
                "request_count": self.llm_requests,
                "prompt_tokens": total_prompt_tokens,
                "cached_prompt_tokens": self.llm_cached_prompt_tokens,
                "uncached_prompt_tokens": uncached_prompt_tokens,
                "completion_tokens": self.llm_completion_tokens,
                "total_tokens": self.llm_total_tokens or total_prompt_tokens + self.llm_completion_tokens,
                "ttft_seconds": self.llm_ttft.summary(),
                "tokens_per_second": self.llm_tokens_per_second.summary(),
                "models": dict(self.llm_models),
            },
            "tts": {
                "request_count": self.tts_requests,
                "characters": self.tts_characters,
                "audio_duration_seconds": self.tts_audio_duration.summary(),
                "ttfb_seconds": self.tts_ttfb.summary(),
                "duration_seconds": self.tts_duration.summary(),
                "cancelled_count": self.tts_cancelled,
                "streamed_count": self.tts_streamed,
                "streamed_percentage": self._percentage(self.tts_streamed, self.tts_requests),
                "models": dict(self.tts_models),
            },
            "stt": {
                "metric_event_count": self.stt_metric_events,
                "audio_duration_seconds": self.stt_audio_duration.summary(),
                "duration_seconds": self.stt_duration.summary(),
                "models": dict(self.stt_models),
            },
            "vad": {
                "event_count": self.vad_events,
                "inference_duration_seconds": self.vad_inference_duration.summary(),
            },
            "eou": {
                "event_count": self.eou_events,
                "end_of_utterance_delay_seconds": self.eou_delay.summary(),
                "transcription_delay_seconds": self.eou_transcription_delay.summary(),
                "callback_delay_seconds": self.eou_callback_delay.summary(),
                "eot_inference_event_count": self.eot_inference_events,
                "eot_inference_duration_seconds": self.eot_inference_duration.summary(),
            },
            "interruptions": {
                "event_count": self.interruption_events,
                "detected_count": self.interruption_count,
                "events_with_interruptions": self.interruption_events_with_interruptions,
                "event_rate_percentage": self._percentage(
                    self.interruption_events_with_interruptions, self.interruption_events
                ),
                "backchannel_count": self.backchannel_count,
                "detection_delay_seconds": self.interruption_detection_delay.summary(),
                "prediction_duration_seconds": self.interruption_prediction_duration.summary(),
                "total_duration_seconds": self.interruption_total_duration.summary(),
            },
        }


def format_summary(summary: dict[str, Any]) -> str:
    """Return a compact, readable representation of a session summary."""
    lines = ["SESSION METRICS SUMMARY"]
    for category, values in summary.items():
        lines.append(f"\n{category.upper()}")
        for name, value in values.items():
            lines.append(f"  {name}: {value}")
    return "\n".join(lines)
