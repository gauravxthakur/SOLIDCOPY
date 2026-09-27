from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import time
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
from metrics.costs import CostCalculator, CreditAccount, RateCard


@dataclass
class _Statistics:
    count: int = 0
    sum: float = 0.0
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
        self.sum += number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)

    def summary(self, digits: int = 3) -> dict[str, float | int | None]:
        average = self.sum / self.count if self.count else None
        return {
            "count": self.count,
            "average": round(average, digits) if average is not None else None,
            "minimum": round(self.minimum, digits) if self.minimum is not None else None,
            "maximum": round(self.maximum, digits) if self.maximum is not None else None,
            "total": round(self.sum, digits),
        }


@dataclass
class SessionMetricsAccumulator:
    """Accumulates one running LiveKit agent session without reading exports."""

    session_id: str | None = None
    rate_card: RateCard | None = None
    credit_account: CreditAccount | None = None
    llm_model: str | None = None
    stt_model: str | None = None
    tts_model: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _started_monotonic: float = field(default_factory=time.monotonic, repr=False)
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
    # Filled from session events (not metrics_collected alone)
    user_utterance_count: int = 0
    turn_count: int = 0
    completed_turns: int = 0
    interrupted_turns: int = 0
    turn_e2e: _Statistics = field(default_factory=_Statistics)
    turn_end_of_turn_delay: _Statistics = field(default_factory=_Statistics)
    turn_llm_ttft: _Statistics = field(default_factory=_Statistics)
    turn_tts_ttfb: _Statistics = field(default_factory=_Statistics)
    ttfa: _Statistics = field(default_factory=_Statistics)
    tool_calls: int = 0
    tool_success: int = 0
    tool_failed: int = 0
    tool_by_name: Counter[str] = field(default_factory=Counter)
    tool_duration: _Statistics = field(default_factory=_Statistics)
    preemptive_started: int = 0
    preemptive_invalidated: int = 0
    turns: list[dict[str, Any]] = field(default_factory=list)
    _tool_started_at: dict[str, float] = field(default_factory=dict, repr=False)
    _tool_names: dict[str, str] = field(default_factory=dict, repr=False)
    _pending_turn: dict[str, Any] | None = field(default=None, repr=False)
    _cost_calculator: CostCalculator = field(init=False, repr=False)
    _checkpointed_turn_keys: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self.rate_card = self.rate_card or RateCard.from_environment()
        self.credit_account = self.credit_account or CreditAccount.from_environment()
        self._cost_calculator = CostCalculator(self.rate_card)

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
        turn = self._pending_turn_for_metrics()
        self.llm_requests += 1
        prompt_tokens = int(getattr(metric, "prompt_tokens", 0) or 0)
        cached_prompt_tokens = int(getattr(metric, "prompt_cached_tokens", 0) or 0)
        completion_tokens = int(getattr(metric, "completion_tokens", 0) or 0)
        self.llm_prompt_tokens += prompt_tokens
        self.llm_cached_prompt_tokens += cached_prompt_tokens
        self.llm_completion_tokens += completion_tokens
        self.llm_total_tokens += int(getattr(metric, "total_tokens", 0) or 0)
        self.llm_cancelled += int(bool(getattr(metric, "cancelled", False)))
        self.llm_duration.add(getattr(metric, "duration", None))
        self.llm_ttft.add(getattr(metric, "ttft", None))
        self.llm_tokens_per_second.add(getattr(metric, "tokens_per_second", None))
        self._count_model(self.llm_models, metric)
        self._add_turn_llm(
            turn,
            metric,
            prompt_tokens,
            cached_prompt_tokens,
            completion_tokens,
        )

    def _collect_realtime_model(self, metric: RealtimeModelMetrics) -> None:
        turn = self._pending_turn_for_metrics()
        self.llm_requests += 1
        prompt_tokens = int(getattr(metric, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(metric, "output_tokens", 0) or 0)
        self.llm_total_tokens += int(getattr(metric, "total_tokens", 0) or 0)
        details = getattr(metric, "input_token_details", None)
        cached_prompt_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        self.llm_prompt_tokens += prompt_tokens
        self.llm_cached_prompt_tokens += cached_prompt_tokens
        self.llm_completion_tokens += completion_tokens
        self.llm_cancelled += int(bool(getattr(metric, "cancelled", False)))
        self.llm_duration.add(getattr(metric, "duration", None))
        self.llm_ttft.add(getattr(metric, "ttft", None))
        self.llm_tokens_per_second.add(getattr(metric, "tokens_per_second", None))
        turn["llm_model"] = self.llm_model or self._model_name(metric) or turn["llm_model"]
        self._add_turn_llm(
            turn,
            metric,
            prompt_tokens,
            cached_prompt_tokens,
            completion_tokens,
        )

    def _collect_tts(self, metric: TTSMetrics) -> None:
        turn = self._pending_turn_for_metrics()
        characters = int(getattr(metric, "characters_count", 0) or 0)
        audio_duration = getattr(metric, "audio_duration", None)
        self.tts_requests += 1
        self.tts_characters += characters
        self.tts_audio_duration.add(audio_duration)
        self.tts_ttfb.add(getattr(metric, "ttfb", None))
        self.tts_duration.add(getattr(metric, "duration", None))
        self.tts_cancelled += int(bool(getattr(metric, "cancelled", False)))
        self.tts_streamed += int(bool(getattr(metric, "streamed", False)))
        self._count_model(self.tts_models, metric)
        turn["tts_requests"] += 1
        turn["tts_model"] = self.tts_model or self._model_name(metric) or turn["tts_model"]
        self._add_turn_value(turn, "tts_characters", characters)
        self._add_turn_seconds(turn, "tts_audio_seconds", audio_duration)

    def _collect_stt(self, metric: STTMetrics) -> None:
        turn = self._pending_turn_for_metrics()
        audio_duration = getattr(metric, "audio_duration", None)
        self.stt_metric_events += 1
        self.stt_audio_duration.add(audio_duration)
        self.stt_duration.add(getattr(metric, "duration", None))
        self._count_model(self.stt_models, metric)
        turn["stt_requests"] += 1
        turn["stt_model"] = self.stt_model or self._model_name(metric) or turn["stt_model"]
        self._add_turn_seconds(turn, "stt_audio_seconds", audio_duration)

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

    def _new_pending_turn(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": None,
            "text": None,
            "llm_model": None,
            "llm_requests": 0,
            "prompt_tokens": 0,
            "cached_prompt_tokens": 0,
            "completion_tokens": 0,
            "ttft_seconds": None,
            "tokens_per_second": None,
            "tts_requests": 0,
            "tts_model": None,
            "tts_characters": 0,
            "tts_audio_seconds": 0.0,
            "stt_requests": 0,
            "stt_model": None,
            "stt_audio_seconds": 0.0,
            "tool_calls_by_id": {},
            "started_monotonic": time.monotonic(),
            "interrupted": False,
            "turn_duration_seconds": None,
        }

    def _pending_turn_for_metrics(self) -> dict[str, Any]:
        if self._pending_turn is None:
            self._pending_turn = self._new_pending_turn()
        return self._pending_turn

    def _add_turn_llm(
        self,
        turn: dict[str, Any],
        metric: Any,
        prompt_tokens: int,
        cached_prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        turn["llm_requests"] += 1
        turn["prompt_tokens"] += prompt_tokens
        turn["cached_prompt_tokens"] += cached_prompt_tokens
        turn["completion_tokens"] += completion_tokens
        turn["llm_model"] = self._model_name(metric) or turn["llm_model"]
        if getattr(metric, "ttft", None) is not None:
            turn["ttft_seconds"] = float(metric.ttft)
        if getattr(metric, "tokens_per_second", None) is not None:
            turn["tokens_per_second"] = float(metric.tokens_per_second)

    @staticmethod
    def _add_turn_value(turn: dict[str, Any], key: str, value: Any) -> None:
        if value is not None:
            turn[key] += value

    @staticmethod
    def _add_turn_seconds(turn: dict[str, Any], key: str, value: Any) -> None:
        if value is not None:
            turn[key] += float(value)

    @staticmethod
    def _model_name(metric: Any) -> str | None:
        metadata = getattr(metric, "metadata", None)
        model = getattr(metric, "label", None)
        if isinstance(metadata, dict):
            model = metadata.get("model_name") or model
        return str(model) if model else None

    @staticmethod
    def _item_text(item: Any) -> str | None:
        text = getattr(item, "text", None)
        if text is None:
            text = getattr(item, "content", None)
        if text is None:
            return None
        if isinstance(text, str):
            return text
        if isinstance(text, (list, tuple)):
            parts = []
            for part in text:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("text") is not None:
                    parts.append(str(part["text"]))
            return "".join(parts) or None
        return str(text)

    @staticmethod
    def _turn_duration(turn: dict[str, Any], report: Any) -> float | None:
        if isinstance(report, dict) and report.get("e2e_latency") is not None:
            return float(report["e2e_latency"])
        return round(time.monotonic() - turn["started_monotonic"], 3)

    @staticmethod
    def _finalize_turn(turn: dict[str, Any]) -> dict[str, Any]:
        tool_calls = list(turn.pop("tool_calls_by_id", {}).values())
        turn.pop("started_monotonic", None)
        turn["uncached_prompt_tokens"] = max(
            turn["prompt_tokens"] - turn["cached_prompt_tokens"], 0
        ) if turn["llm_requests"] else None
        turn["prompt_tokens"] = turn["prompt_tokens"] if turn["llm_requests"] else None
        turn["cached_prompt_tokens"] = (
            turn["cached_prompt_tokens"] if turn["llm_requests"] else None
        )
        turn["completion_tokens"] = (
            turn["completion_tokens"] if turn["llm_requests"] else None
        )
        turn["tts_characters"] = turn["tts_characters"] if turn["tts_requests"] else None
        turn["tts_audio_seconds"] = (
            round(turn["tts_audio_seconds"], 3) if turn["tts_requests"] else None
        )
        turn["stt_audio_seconds"] = (
            round(turn["stt_audio_seconds"], 3) if turn["stt_requests"] else None
        )
        turn["tool_names"] = [call["name"] for call in tool_calls]
        turn["tool_latency_seconds"] = [
            call["latency_seconds"] for call in tool_calls
        ]
        turn["tool_calls"] = tool_calls
        turn.pop("llm_requests", None)
        turn.pop("tts_requests", None)
        turn.pop("stt_requests", None)
        return turn

    def note_final_transcript(self) -> None:
        """Count one user utterance from a final STT transcript event."""
        self.user_utterance_count += 1

    def note_assistant_message(self, item: Any) -> None:
        """Record one agent turn from a conversation_item_added assistant message."""
        if getattr(item, "role", None) != "assistant":
            return
        self.turn_count += 1
        if bool(getattr(item, "interrupted", False)):
            self.interrupted_turns += 1
        else:
            self.completed_turns += 1
        report = getattr(item, "metrics", None) or {}
        if isinstance(report, dict):
            self.turn_e2e.add(report.get("e2e_latency"))
            self.turn_end_of_turn_delay.add(report.get("end_of_turn_delay"))
            self.turn_llm_ttft.add(report.get("llm_node_ttft"))
            self.turn_tts_ttfb.add(report.get("tts_node_ttfb"))
        turn = self._pending_turn or self._new_pending_turn()
        turn["turn_id"] = f"turn-{self.turn_count:04d}"
        turn["text"] = self._item_text(item)
        turn["interrupted"] = bool(getattr(item, "interrupted", False))
        turn["turn_duration_seconds"] = self._turn_duration(turn, report)
        turn["cost_breakdown"] = self._cost_calculator.calculate_turn(turn)
        turn["credit_simulation"] = self.credit_account.record_connected_seconds(
            turn["turn_duration_seconds"] or 0
        )
        self.turns.append(self._finalize_turn(turn))
        self._pending_turn = None

    def checkpoint_after_turn(self, directory: Path | str | None = None) -> Path | None:
        """Persist the current summary after a turn, deduplicated by session and turn."""
        if not self.turns:
            return None
        latest = self.turns[-1]
        key = self._turn_key(latest)
        if key in self._checkpointed_turn_keys:
            return self._checkpoint_path(directory)
        self._checkpointed_turn_keys.add(key)
        return persist_summary(self.summary(), directory=directory)

    def _checkpoint_path(self, directory: Path | str | None = None) -> Path:
        session_id = self.session_id or "unknown"
        safe_id = _SAFE_SESSION_ID.sub("_", str(session_id)).strip("._") or "unknown"
        target = Path(directory) if directory else DEFAULT_SUMMARY_DIR
        return target / f"{safe_id}.json"

    def _turn_key(self, turn: dict[str, Any]) -> str:
        return f"{self.session_id or 'unknown'}:{turn.get('turn_id')}"

    def note_ttfa(self, seconds: float | None) -> None:
        """Time from EOU decision to agent first audio (speaking)."""
        self.ttfa.add(seconds)

    def note_function_tools_executed(self, calls: Any, outputs: Any) -> None:
        """Record a finished tool batch from function_tools_executed."""
        turn = self._pending_turn_for_metrics()
        pairs = list(zip(list(calls or []), list(outputs or []), strict=False))
        for call, output in pairs:
            name = str(getattr(call, "name", None) or "unknown")
            call_id = str(getattr(call, "call_id", None) or name)
            turn["tool_calls_by_id"].setdefault(
                call_id,
                {"name": name, "latency_seconds": None},
            )["name"] = name
            self.tool_calls += 1
            self.tool_by_name[name] += 1
            if output is not None and bool(getattr(output, "is_error", False)):
                self.tool_failed += 1
            else:
                self.tool_success += 1

    def note_tool_started(self, call_id: str, name: str | None = None) -> None:
        turn = self._pending_turn_for_metrics()
        self._tool_started_at[call_id] = time.monotonic()
        if name:
            self._tool_names[call_id] = name
        turn["tool_calls_by_id"][call_id] = {
            "name": name or "unknown",
            "latency_seconds": None,
        }

    def note_tool_ended(self, call_id: str, status: str | None = None) -> None:
        started = self._tool_started_at.pop(call_id, None)
        self._tool_names.pop(call_id, None)
        if started is not None:
            latency = time.monotonic() - started
            self.tool_duration.add(latency)
            if self._pending_turn is not None:
                tool = self._pending_turn["tool_calls_by_id"].setdefault(
                    call_id,
                    {"name": "unknown", "latency_seconds": None},
                )
                tool["latency_seconds"] = round(latency, 3)

    def note_preemptive_started(self) -> None:
        self.preemptive_started += 1

    def note_preemptive_invalidated(self) -> None:
        self.preemptive_invalidated += 1

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
        ended_at = datetime.now(timezone.utc)
        turn_costs = [record.get("cost_breakdown") for record in self.turns]
        line_values = {"llm": [], "stt": [], "tts": []}
        line_statuses = {"llm": [], "stt": [], "tts": []}
        for breakdown in turn_costs:
            if not breakdown:
                continue
            for name, line in breakdown["lines"].items():
                line_statuses[name].append(line["status"])
                if line["status"] == "measured":
                    line_values[name].append(line["cost_usd"])
        cost_breakdown = {
            "currency": "USD",
            "total_cost_usd": round(sum(
                value for values in line_values.values() for value in values
            ), 6),
            "lines": {
                name: {
                    "status": (
                        "missing_rate" if "missing_rate" in line_statuses[name]
                        else "measured" if line_values[name]
                        else "not_applicable"
                    ),
                    "cost_usd": round(sum(line_values[name]), 6) if line_values[name] else None,
                }
                for name in line_values
            },
            "turns_with_missing_rates": sum(
                bool(breakdown and breakdown["status"] == "missing_rate")
                for breakdown in turn_costs
            ),
        }
        credit_summary = {
            "plan_name": self.credit_account.plan_name,
            "customer_rate_inr_per_second": float(
                self.credit_account.customer_rate_per_second
            ),
            "credit_unit": "1 connected second",
            "connected_seconds_source": "completed_turn_duration_simulation",
            **self.credit_account.snapshot(),
        }
        return {
            "session": {
                "session_id": self.session_id,
                "started_at": self.started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "duration_seconds": round(time.monotonic() - self._started_monotonic, 3),
            },
            "events": {"metric_event_count": self.metric_event_count},
            "cost_breakdown": cost_breakdown,
            "credit_simulation": credit_summary,
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
                "utterance_count": self.user_utterance_count,
                "audio_duration_seconds": self.stt_audio_duration.summary(),
                "duration_seconds": self.stt_duration.summary(),
                "models": dict(self.stt_models),
            },
            "eou": {
                "event_count": self.eou_events,
                "end_of_utterance_delay_seconds": self.eou_delay.summary(),
                "transcription_delay_seconds": self.eou_transcription_delay.summary(),
                "callback_delay_seconds": self.eou_callback_delay.summary(),
                "eot_inference_event_count": self.eot_inference_events,
                "eot_inference_duration_seconds": self.eot_inference_duration.summary(),
            },
            "turns": {
                "count": self.turn_count,
                "completed_count": self.completed_turns,
                "interrupted_count": self.interrupted_turns,
                "interruption_rate_percentage": self._percentage(
                    self.interrupted_turns, self.turn_count
                ),
                "end_to_end_latency_seconds": self.turn_e2e.summary(),
                "end_of_turn_delay_seconds": self.turn_end_of_turn_delay.summary(),
                "llm_ttft_seconds": self.turn_llm_ttft.summary(),
                "tts_ttfb_seconds": self.turn_tts_ttfb.summary(),
                "ttfa_seconds": self.ttfa.summary(),
                "records": list(self.turns),
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
            "tools": {
                "count": self.tool_calls,
                "successful_count": self.tool_success,
                "failed_count": self.tool_failed,
                "by_name": dict(self.tool_by_name),
                "duration_seconds": self.tool_duration.summary(),
            },
            "runtime": {
                "llm_cancelled_count": self.llm_cancelled,
                "tts_cancelled_count": self.tts_cancelled,
                "vad_event_count": self.vad_events,
                "vad_inference_duration_seconds": self.vad_inference_duration.summary(),
                "preemptive_started_count": self.preemptive_started,
                "preemptive_invalidated_count": self.preemptive_invalidated,
                "preemptive_invalidation_rate_percentage": self._percentage(
                    self.preemptive_invalidated, self.preemptive_started
                ),
            },
        }


_SAFE_SESSION_ID = re.compile(r"[^A-Za-z0-9._-]+")
DEFAULT_SUMMARY_DIR = Path("metrics/sessions")


def persist_summary(
    summary: dict[str, Any],
    directory: Path | str | None = None,
) -> Path:
    """Write the structured session summary to disk. No prompts or transcripts."""
    session_id = (summary.get("session") or {}).get("session_id") or "unknown"
    safe_id = _SAFE_SESSION_ID.sub("_", str(session_id)).strip("._") or "unknown"
    target = Path(directory) if directory else DEFAULT_SUMMARY_DIR
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{safe_id}.json"
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _fmt_num(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.2f}" if abs(value) >= 10 else f"{value:.2f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _avg(stats: dict[str, Any] | None) -> Any:
    return None if not isinstance(stats, dict) else stats.get("average")


def _total(stats: dict[str, Any] | None) -> Any:
    return None if not isinstance(stats, dict) else stats.get("total")


def format_summary(summary: dict[str, Any]) -> str:
    """Return a compact, readable representation of a session summary."""
    session = summary.get("session", {})
    events = summary.get("events", {})
    llm = summary.get("llm", {})
    tts = summary.get("tts", {})
    stt = summary.get("stt", {})
    eou = summary.get("eou", {})
    turns = summary.get("turns", {})
    tools = summary.get("tools", {})

    lines = [
        "SESSION METRICS SUMMARY",
        f"Session duration: {_fmt_duration(session.get('duration_seconds'))}",
        f"Metric events: {_fmt_num(events.get('metric_event_count'))}",
        "",
        "LLM",
        f"  Requests: {_fmt_num(llm.get('request_count'))}",
        f"  Prompt tokens: {_fmt_num(llm.get('prompt_tokens'))}",
        f"  Cached prompt tokens: {_fmt_num(llm.get('cached_prompt_tokens'))}",
        f"  Completion tokens: {_fmt_num(llm.get('completion_tokens'))}",
        f"  Average TTFT: {_fmt_num(_avg(llm.get('ttft_seconds')))} s",
        f"  Average tokens/sec: {_fmt_num(_avg(llm.get('tokens_per_second')))}",
        "",
        "TTS",
        f"  Requests: {_fmt_num(tts.get('request_count'))}",
        f"  Total characters: {_fmt_num(tts.get('characters'))}",
        f"  Total audio: {_fmt_num(_total(tts.get('audio_duration_seconds')))} s",
        f"  Average TTFB: {_fmt_num(_avg(tts.get('ttfb_seconds')))} s",
        "",
        "STT",
        f"  Metric events: {_fmt_num(stt.get('metric_event_count'))}",
        f"  User utterances: {_fmt_num(stt.get('utterance_count'))}",
        f"  Total audio: {_fmt_num(_total(stt.get('audio_duration_seconds')))} s",
        f"  Average transcription delay: {_fmt_num(_avg(eou.get('transcription_delay_seconds')))} s",
        "",
        "TURNS",
        f"  Turns: {_fmt_num(turns.get('count'))}",
        f"  Completed: {_fmt_num(turns.get('completed_count'))}",
        f"  Interrupted turns: {_fmt_num(turns.get('interrupted_count'))}",
        f"  Average end-to-end latency: {_fmt_num(_avg(turns.get('end_to_end_latency_seconds')))} s",
        f"  Average TTFA: {_fmt_num(_avg(turns.get('ttfa_seconds')))} s",
        "",
        "TOOLS",
        f"  Total calls: {_fmt_num(tools.get('count'))}",
        f"  Successful: {_fmt_num(tools.get('successful_count'))}",
        f"  Failed: {_fmt_num(tools.get('failed_count'))}",
    ]
    for name, count in (tools.get("by_name") or {}).items():
        lines.append(f"  {name}: {_fmt_num(count)}")
    return "\n".join(lines)
