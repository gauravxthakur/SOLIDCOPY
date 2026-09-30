"""Session metrics accumulator and legacy MetricsFrame logger.

``SessionMetricsAccumulator`` has no Pipecat imports: observers pass plain
dicts from ``metrics.observers`` mapping helpers into ``collect`` / ``note_*``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import time
from typing import Any, Mapping

from metrics.types import (
    CostBreakdown,
    CreditSimulation,
    EndpointingSection,
    EventsSection,
    InterruptionsSection,
    LlmSection,
    ProcessorStartupTiming,
    RuntimeSection,
    SessionMeta,
    SessionSummary,
    StartupSection,
    StatSummary,
    SttSection,
    ToolCallRecord,
    ToolsSection,
    TurnRecord,
    TurnsSection,
    TtsSection,
    empty_session_summary,
    session_summary_to_dict,
)

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class RunningStats:
    """Mutable count / average / min / max / total accumulator."""

    count: int = 0
    total: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, value: Any) -> None:
        if value is None or isinstance(value, bool):
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if math.isnan(number) or math.isinf(number):
            return
        self.count += 1
        self.total += number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)

    def summary(self, digits: int = 3) -> StatSummary:
        average = self.total / self.count if self.count else None
        return StatSummary(
            count=self.count,
            average=round(average, digits) if average is not None else None,
            minimum=round(self.minimum, digits) if self.minimum is not None else None,
            maximum=round(self.maximum, digits) if self.maximum is not None else None,
            total=round(self.total, digits),
        )


_LATENCY_KINDS = frozenset({"ttfb", "ttfa", "ttfat"})
_USAGE_KINDS = frozenset({"llm", "stt", "tts"})
_LLM_PROCESSOR_TOKENS = (
    "llm",
    "openai",
    "google",
    "anthropic",
    "gemini",
    "bedrock",
    "groq",
    "ollama",
    "mistral",
    "deepseek",
    "together",
    "cerebras",
    "fireworks",
)
_TTS_PROCESSOR_TOKENS = (
    "tts",
    "cartesia",
    "eleven",
    "deepgram-tts",
    "azure",
    "playht",
    "lmnt",
    "rime",
    "piper",
    "polly",
    "openai-tts",
)


def _as_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return 0
        return int(number)
    except (TypeError, ValueError):
        return 0


def _as_optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _count_model(counter: Counter[str], model: Any) -> None:
    if model:
        counter[str(model)] += 1


def _percentage(part: int, whole: int) -> float | None:
    return round(part * 100 / whole, 2) if whole else None


def _looks_like_llm_processor(processor: str | None) -> bool:
    name = (processor or "").lower()
    return any(token in name for token in _LLM_PROCESSOR_TOKENS)


def _looks_like_tts_processor(processor: str | None) -> bool:
    name = (processor or "").lower()
    return any(token in name for token in _TTS_PROCESSOR_TOKENS)


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


@dataclass
class SessionMetricsAccumulator:
    """In-memory session recorder. Accepts framework-neutral dict payloads only."""

    session_id: str | None = None
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
    llm_cache_creation_input_tokens: int = 0
    llm_reasoning_tokens: int = 0
    llm_input_audio_tokens: int = 0
    llm_output_audio_tokens: int = 0
    llm_cache_read_input_audio_tokens: int = 0
    llm_ttfb: RunningStats = field(default_factory=RunningStats)
    llm_ttfat: RunningStats = field(default_factory=RunningStats)
    llm_models: Counter[str] = field(default_factory=Counter)

    tts_requests: int = 0
    tts_characters: int = 0
    tts_ttfb: RunningStats = field(default_factory=RunningStats)
    tts_ttfa: RunningStats = field(default_factory=RunningStats)
    tts_leading_silence: RunningStats = field(default_factory=RunningStats)
    tts_models: Counter[str] = field(default_factory=Counter)

    stt_metric_events: int = 0
    stt_audio_duration: RunningStats = field(default_factory=RunningStats)
    stt_models: Counter[str] = field(default_factory=Counter)
    user_utterance_count: int = 0

    user_turn_seconds: RunningStats = field(default_factory=RunningStats)
    turn_prediction_event_count: int = 0
    turn_prediction_duration: RunningStats = field(default_factory=RunningStats)

    turn_count: int = 0
    completed_turns: int = 0
    interrupted_turns: int = 0
    user_bot_latency: RunningStats = field(default_factory=RunningStats)
    first_bot_speech_latency_seconds: float | None = None
    turn_llm_ttfb: RunningStats = field(default_factory=RunningStats)
    turn_llm_ttfat: RunningStats = field(default_factory=RunningStats)
    turn_tts_ttfb: RunningStats = field(default_factory=RunningStats)
    turn_tts_ttfa: RunningStats = field(default_factory=RunningStats)
    turns: list[TurnRecord] = field(default_factory=list)

    tool_calls: int = 0
    tool_success: int = 0
    tool_failed: int = 0
    tool_cancelled: int = 0
    tool_by_name: Counter[str] = field(default_factory=Counter)
    tool_duration: RunningStats = field(default_factory=RunningStats)

    client_connected_secs: float | None = None
    bot_connected_secs: float | None = None
    startup: StartupSection = field(default_factory=StartupSection)
    interrupted_cycle_count: int = 0

    _pending_turn: dict[str, Any] | None = field(default=None, repr=False)
    _tool_started_at: dict[str, float] = field(default_factory=dict, repr=False)
    _open_tools: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    # --- collect / note API -------------------------------------------------

    def collect(self, record: Mapping[str, Any]) -> None:
        """Dispatch a service latency or usage dict from observer helpers."""
        kind = str(record.get("kind") or "")
        if kind in _LATENCY_KINDS:
            self.note_service_latency(record)
            return
        if kind in _USAGE_KINDS:
            self.note_service_usage(record)
            return
        raise ValueError(
            f"Unsupported collect record kind={kind!r}; expected latency "
            f"{sorted(_LATENCY_KINDS)} or usage {sorted(_USAGE_KINDS)}"
        )

    def note_service_latency(self, record: Mapping[str, Any]) -> None:
        """Record one ``latency_record_to_dict`` payload."""
        self.metric_event_count += 1
        kind = str(record.get("kind") or "")
        seconds = _as_optional_float(record.get("seconds"))
        processor = record.get("processor")
        model = record.get("model")
        turn = self._pending_turn_for_metrics()

        if kind == "ttfat":
            self.llm_ttfat.add(seconds)
            self.turn_llm_ttfat.add(seconds)
            _count_model(self.llm_models, model or self.llm_model)
            if seconds is not None:
                turn["ttfat_seconds"] = seconds
            ttfb = _as_optional_float(record.get("ttfb_secs"))
            if ttfb is not None:
                self.llm_ttfb.add(ttfb)
                self.turn_llm_ttfb.add(ttfb)
                turn["ttfb_seconds"] = ttfb
            turn["llm_model"] = model or self.llm_model or turn.get("llm_model")
            return

        if kind == "ttfa":
            self.tts_ttfa.add(seconds)
            self.turn_tts_ttfa.add(seconds)
            self.tts_leading_silence.add(record.get("leading_silence_secs"))
            _count_model(self.tts_models, model or self.tts_model)
            if seconds is not None:
                turn["tts_ttfa_seconds"] = seconds
            ttfb = _as_optional_float(record.get("ttfb_secs"))
            if ttfb is not None:
                self.tts_ttfb.add(ttfb)
                self.turn_tts_ttfb.add(ttfb)
                turn["tts_ttfb_seconds"] = ttfb
            turn["tts_model"] = model or self.tts_model or turn.get("tts_model")
            return

        if kind == "ttfb":
            if _looks_like_tts_processor(str(processor) if processor else None) and not (
                _looks_like_llm_processor(str(processor) if processor else None)
            ):
                self.tts_ttfb.add(seconds)
                self.turn_tts_ttfb.add(seconds)
                _count_model(self.tts_models, model or self.tts_model)
                if seconds is not None:
                    turn["tts_ttfb_seconds"] = seconds
                turn["tts_model"] = model or self.tts_model or turn.get("tts_model")
            else:
                # Default standalone TTFB to LLM when processor is ambiguous.
                self.llm_ttfb.add(seconds)
                self.turn_llm_ttfb.add(seconds)
                _count_model(self.llm_models, model or self.llm_model)
                if seconds is not None:
                    turn["ttfb_seconds"] = seconds
                turn["llm_model"] = model or self.llm_model or turn.get("llm_model")
            return

    def note_service_usage(self, record: Mapping[str, Any]) -> None:
        """Record one ``usage_record_to_dict`` payload."""
        self.metric_event_count += 1
        kind = str(record.get("kind") or "")
        model = record.get("model")
        turn = self._pending_turn_for_metrics()

        if kind == "llm":
            self.llm_requests += 1
            prompt = _as_int(record.get("prompt_tokens"))
            cached = _as_int(record.get("cache_read_input_tokens"))
            completion = _as_int(record.get("completion_tokens"))
            total = _as_int(record.get("total_tokens"))
            cache_create = _as_int(record.get("cache_creation_input_tokens"))
            reasoning = _as_int(record.get("reasoning_tokens"))
            input_audio = _as_int(record.get("input_audio_tokens"))
            output_audio = _as_int(record.get("output_audio_tokens"))
            cache_read_audio = _as_int(record.get("cache_read_input_audio_tokens"))
            self.llm_prompt_tokens += prompt
            self.llm_cached_prompt_tokens += cached
            self.llm_completion_tokens += completion
            self.llm_total_tokens += total or (prompt + completion)
            self.llm_cache_creation_input_tokens += cache_create
            self.llm_reasoning_tokens += reasoning
            self.llm_input_audio_tokens += input_audio
            self.llm_output_audio_tokens += output_audio
            self.llm_cache_read_input_audio_tokens += cache_read_audio
            _count_model(self.llm_models, model or self.llm_model)
            turn["llm_requests"] = int(turn.get("llm_requests") or 0) + 1
            turn["prompt_tokens"] = int(turn.get("prompt_tokens") or 0) + prompt
            turn["cached_prompt_tokens"] = int(turn.get("cached_prompt_tokens") or 0) + cached
            turn["completion_tokens"] = int(turn.get("completion_tokens") or 0) + completion
            turn["llm_model"] = model or self.llm_model or turn.get("llm_model")
            return

        if kind == "tts":
            characters = _as_int(record.get("characters"))
            self.tts_requests += 1
            self.tts_characters += characters
            _count_model(self.tts_models, model or self.tts_model)
            turn["tts_requests"] = int(turn.get("tts_requests") or 0) + 1
            turn["tts_characters"] = int(turn.get("tts_characters") or 0) + characters
            turn["tts_model"] = model or self.tts_model or turn.get("tts_model")
            return

        if kind == "stt":
            audio = _as_optional_float(record.get("audio_seconds"))
            self.stt_metric_events += 1
            self.stt_audio_duration.add(audio)
            _count_model(self.stt_models, model or self.stt_model)
            turn["stt_requests"] = int(turn.get("stt_requests") or 0) + 1
            if audio is not None:
                turn["stt_audio_seconds"] = float(turn.get("stt_audio_seconds") or 0.0) + audio
            turn["stt_model"] = model or self.stt_model or turn.get("stt_model")
            return

    def note_turn_started(self, turn_count: int) -> None:
        """Open a pending turn bucket for subsequent metrics."""
        turn = self._new_pending_turn()
        turn["turn_id"] = f"turn-{int(turn_count):04d}"
        self._pending_turn = turn

    def note_turn_ended(self, record: Mapping[str, Any]) -> None:
        """Finalize one turn from ``turn_ended_to_dict``."""
        self.turn_count += 1
        interrupted = bool(record.get("was_interrupted"))
        if interrupted:
            self.interrupted_turns += 1
            self.interrupted_cycle_count += 1
        else:
            self.completed_turns += 1

        turn = self._pending_turn or self._new_pending_turn()
        turn_count = record.get("turn_count")
        if turn_count is not None:
            turn["turn_id"] = f"turn-{int(turn_count):04d}"
        elif not turn.get("turn_id"):
            turn["turn_id"] = f"turn-{self.turn_count:04d}"
        turn["interrupted"] = interrupted
        turn["turn_duration_seconds"] = _as_optional_float(record.get("duration_secs"))
        self.turns.append(self._finalize_turn(turn))
        self._pending_turn = None

    def note_assistant_text(self, text: str | None) -> None:
        """Attach assistant response text to the active pending turn."""
        if text is None:
            return
        turn = self._pending_turn_for_metrics()
        existing = turn.get("text")
        turn["text"] = text if not existing else f"{existing}{text}"

    def note_final_transcript(self, record: Mapping[str, Any] | None = None) -> None:
        """Count a finalized user utterance (``transcription_to_dict`` or bare)."""
        if record is not None and record.get("finalized") is False:
            return
        self.user_utterance_count += 1

    def note_user_bot_latency(self, record: Mapping[str, Any] | float) -> None:
        """Record user-silence → bot-speech latency."""
        if isinstance(record, Mapping):
            seconds = _as_optional_float(record.get("latency_seconds"))
        else:
            seconds = _as_optional_float(record)
        self.user_bot_latency.add(seconds)
        if seconds is not None and self._pending_turn is not None:
            self._pending_turn["user_bot_latency_seconds"] = seconds

    def note_first_bot_speech_latency(self, seconds: float | None) -> None:
        """Record client-connect → first bot speech (once)."""
        value = _as_optional_float(seconds)
        if value is None:
            return
        if self.first_bot_speech_latency_seconds is None:
            self.first_bot_speech_latency_seconds = value

    def note_latency_breakdown(self, record: Mapping[str, Any]) -> None:
        """Ingest ``latency_breakdown_to_dict`` for endpointing + tool spans."""
        self.user_turn_seconds.add(record.get("user_turn_secs"))
        for call in record.get("function_calls") or []:
            if not isinstance(call, Mapping):
                continue
            name = str(call.get("function_name") or "unknown")
            duration = _as_optional_float(call.get("duration_secs"))
            self.tool_duration.add(duration)
            # Durations here are observational; counts come from tool frame notes.
            if self._pending_turn is not None and duration is not None:
                by_id = self._pending_turn.setdefault("tool_calls_by_id", {})
                # Match by name if call id unknown in breakdown.
                for tool in by_id.values():
                    if tool.get("name") == name and tool.get("latency_seconds") is None:
                        tool["latency_seconds"] = round(duration, 3)
                        break

    def note_turn_prediction(self, *, is_complete: bool | None = None, duration_ms: Any = None) -> None:
        """Optional smart-turn / TurnMetricsData style event (if collected later)."""
        self.turn_prediction_event_count += 1
        if duration_ms is not None:
            try:
                self.turn_prediction_duration.add(float(duration_ms) / 1000.0)
            except (TypeError, ValueError):
                pass
        _ = is_complete  # retained for future filtering; counting all events for now

    def note_tool_started(self, record: Mapping[str, Any]) -> None:
        """Record ``tool_started_to_dict``."""
        call_id = str(record.get("tool_call_id") or "")
        name = str(record.get("function_name") or "unknown")
        if not call_id:
            call_id = name
        self._tool_started_at[call_id] = time.monotonic()
        self._open_tools[call_id] = {"name": name, "tool_call_id": call_id, "latency_seconds": None, "ok": None, "error": None}
        turn = self._pending_turn_for_metrics()
        turn.setdefault("tool_calls_by_id", {})[call_id] = {
            "name": name,
            "tool_call_id": call_id,
            "latency_seconds": None,
            "ok": None,
            "error": None,
        }
        self.tool_calls += 1
        self.tool_by_name[name] += 1

    def note_tool_result(self, record: Mapping[str, Any]) -> None:
        """Record ``tool_result_to_dict``."""
        call_id = str(record.get("tool_call_id") or "")
        name = str(record.get("function_name") or self._open_tools.get(call_id, {}).get("name") or "unknown")
        ok = bool(record.get("ok", record.get("error") is None))
        error = record.get("error")
        started = self._tool_started_at.pop(call_id, None)
        latency = round(time.monotonic() - started, 3) if started is not None else None
        if latency is not None:
            self.tool_duration.add(latency)
        if ok:
            self.tool_success += 1
        else:
            self.tool_failed += 1
        self._open_tools.pop(call_id, None)
        if self._pending_turn is not None:
            tool = self._pending_turn.setdefault("tool_calls_by_id", {}).setdefault(
                call_id,
                {"name": name, "tool_call_id": call_id, "latency_seconds": None, "ok": None, "error": None},
            )
            tool["name"] = name
            tool["tool_call_id"] = call_id
            tool["ok"] = ok
            tool["error"] = error
            if latency is not None:
                tool["latency_seconds"] = latency

    def note_tool_cancelled(self, record: Mapping[str, Any]) -> None:
        """Record ``tool_cancelled_to_dict``."""
        call_id = str(record.get("tool_call_id") or "")
        name = str(
            record.get("function_name")
            or self._open_tools.get(call_id, {}).get("name")
            or "unknown"
        )
        had_start = call_id in self._tool_started_at or call_id in self._open_tools
        self._tool_started_at.pop(call_id, None)
        self._open_tools.pop(call_id, None)
        self.tool_cancelled += 1
        self.tool_failed += 1
        if not had_start:
            # Cancel without a prior started note still counts as a call.
            self.tool_calls += 1
            self.tool_by_name[name] += 1
        if self._pending_turn is not None:
            tool = self._pending_turn.setdefault("tool_calls_by_id", {}).setdefault(
                call_id or name,
                {"name": name, "tool_call_id": call_id, "latency_seconds": None, "ok": None, "error": None},
            )
            tool["name"] = name
            tool["tool_call_id"] = call_id
            tool["ok"] = False
            tool["error"] = "cancelled"

    def note_startup(self, record: Mapping[str, Any]) -> None:
        """Record ``startup_report_to_dict``."""
        self.startup.total_duration_secs = _as_optional_float(record.get("total_duration_secs"))
        self.startup.setup_phase_secs = _as_optional_float(record.get("setup_phase_secs"))
        self.startup.start_phase_secs = _as_optional_float(record.get("start_phase_secs"))
        timings = []
        for item in record.get("processor_timings") or []:
            if not isinstance(item, Mapping):
                continue
            timings.append(
                ProcessorStartupTiming(
                    processor_name=str(item.get("processor_name") or "unknown"),
                    start_offset_secs=float(item.get("start_offset_secs") or 0.0),
                    duration_secs=float(item.get("duration_secs") or 0.0),
                    setup_duration_secs=float(item.get("setup_duration_secs") or 0.0),
                    start_duration_secs=float(item.get("start_duration_secs") or 0.0),
                )
            )
        self.startup.processor_timings = timings
        warmup = record.get("warmup")
        if isinstance(warmup, Mapping):
            self.startup.warmup_duration_secs = _as_optional_float(warmup.get("duration_secs"))
            self.startup.warmup_blocking_duration_secs = _as_optional_float(
                warmup.get("blocking_duration_secs")
            )

    def note_transport(self, record: Mapping[str, Any]) -> None:
        """Record ``transport_report_to_dict``."""
        self.client_connected_secs = _as_optional_float(record.get("client_connected_secs"))
        self.bot_connected_secs = _as_optional_float(record.get("bot_connected_secs"))

    def note_interrupted_cycle(self) -> None:
        """Approximate cancelled in-flight LLM/TTS work via interruption frames."""
        self.interrupted_cycle_count += 1

    # --- summary ------------------------------------------------------------

    def summary(self) -> SessionSummary:
        """Build a typed session summary from current counters."""
        ended_at = datetime.now(timezone.utc)
        uncached = max(self.llm_prompt_tokens - self.llm_cached_prompt_tokens, 0)
        total_tokens = self.llm_total_tokens or (self.llm_prompt_tokens + self.llm_completion_tokens)

        base = empty_session_summary(self.session_id)
        base.session = SessionMeta(
            session_id=self.session_id,
            started_at=self.started_at.isoformat(),
            ended_at=ended_at.isoformat(),
            duration_seconds=round(time.monotonic() - self._started_monotonic, 3),
        )
        base.events = EventsSection(metric_event_count=self.metric_event_count)
        base.cost_breakdown = CostBreakdown()
        base.credit_simulation = CreditSimulation()
        base.llm = LlmSection(
            request_count=self.llm_requests,
            prompt_tokens=self.llm_prompt_tokens,
            cached_prompt_tokens=self.llm_cached_prompt_tokens,
            uncached_prompt_tokens=uncached,
            completion_tokens=self.llm_completion_tokens,
            total_tokens=total_tokens,
            cache_creation_input_tokens=self.llm_cache_creation_input_tokens,
            reasoning_tokens=self.llm_reasoning_tokens,
            input_audio_tokens=self.llm_input_audio_tokens,
            output_audio_tokens=self.llm_output_audio_tokens,
            cache_read_input_audio_tokens=self.llm_cache_read_input_audio_tokens,
            ttfb_seconds=self.llm_ttfb.summary(),
            ttfat_seconds=self.llm_ttfat.summary(),
            models=dict(self.llm_models),
        )
        base.tts = TtsSection(
            request_count=self.tts_requests,
            characters=self.tts_characters,
            ttfb_seconds=self.tts_ttfb.summary(),
            ttfa_seconds=self.tts_ttfa.summary(),
            leading_silence_seconds=self.tts_leading_silence.summary(),
            models=dict(self.tts_models),
        )
        base.stt = SttSection(
            metric_event_count=self.stt_metric_events,
            utterance_count=self.user_utterance_count,
            audio_duration_seconds=self.stt_audio_duration.summary(),
            models=dict(self.stt_models),
        )
        base.endpointing = EndpointingSection(
            user_turn_seconds=self.user_turn_seconds.summary(),
            turn_prediction_event_count=self.turn_prediction_event_count,
            turn_prediction_duration_seconds=self.turn_prediction_duration.summary(),
        )
        base.turns = TurnsSection(
            count=self.turn_count,
            completed_count=self.completed_turns,
            interrupted_count=self.interrupted_turns,
            interruption_rate_percentage=_percentage(self.interrupted_turns, self.turn_count),
            user_bot_latency_seconds=self.user_bot_latency.summary(),
            first_bot_speech_latency_seconds=self.first_bot_speech_latency_seconds,
            llm_ttfb_seconds=self.turn_llm_ttfb.summary(),
            llm_ttfat_seconds=self.turn_llm_ttfat.summary(),
            tts_ttfb_seconds=self.turn_tts_ttfb.summary(),
            tts_ttfa_seconds=self.turn_tts_ttfa.summary(),
            records=list(self.turns),
        )
        base.interruptions = InterruptionsSection(
            interrupted_turn_count=self.interrupted_turns,
        )
        base.tools = ToolsSection(
            count=self.tool_calls,
            successful_count=self.tool_success,
            failed_count=self.tool_failed,
            cancelled_count=self.tool_cancelled,
            by_name=dict(self.tool_by_name),
            duration_seconds=self.tool_duration.summary(),
        )
        base.runtime = RuntimeSection(
            client_connected_secs=self.client_connected_secs,
            bot_connected_secs=self.bot_connected_secs,
            startup=self.startup,
            interrupted_cycle_count=self.interrupted_cycle_count,
        )
        return base

    def summary_dict(self) -> dict[str, Any]:
        """JSON-serializable summary document."""
        return session_summary_to_dict(self.summary())

    # --- internals ----------------------------------------------------------

    def _new_pending_turn(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_id": None,
            "text": None,
            "interrupted": False,
            "turn_duration_seconds": None,
            "user_bot_latency_seconds": None,
            "llm_model": self.llm_model,
            "llm_requests": 0,
            "prompt_tokens": 0,
            "cached_prompt_tokens": 0,
            "completion_tokens": 0,
            "ttfb_seconds": None,
            "ttfat_seconds": None,
            "tts_model": self.tts_model,
            "tts_requests": 0,
            "tts_characters": 0,
            "tts_ttfb_seconds": None,
            "tts_ttfa_seconds": None,
            "stt_model": self.stt_model,
            "stt_requests": 0,
            "stt_audio_seconds": 0.0,
            "tool_calls_by_id": {},
        }

    def _pending_turn_for_metrics(self) -> dict[str, Any]:
        if self._pending_turn is None:
            self._pending_turn = self._new_pending_turn()
        return self._pending_turn

    def _finalize_turn(self, turn: dict[str, Any]) -> TurnRecord:
        tool_calls_raw = list(turn.pop("tool_calls_by_id", {}).values())
        tool_calls = [
            ToolCallRecord(
                name=item.get("name"),
                tool_call_id=item.get("tool_call_id"),
                latency_seconds=item.get("latency_seconds"),
                ok=item.get("ok"),
                error=item.get("error"),
            )
            for item in tool_calls_raw
        ]
        prompt = int(turn.get("prompt_tokens") or 0)
        cached = int(turn.get("cached_prompt_tokens") or 0)
        llm_requests = int(turn.get("llm_requests") or 0)
        tts_requests = int(turn.get("tts_requests") or 0)
        stt_requests = int(turn.get("stt_requests") or 0)
        stt_audio = float(turn.get("stt_audio_seconds") or 0.0)

        return TurnRecord(
            session_id=turn.get("session_id") or self.session_id,
            turn_id=turn.get("turn_id"),
            text=turn.get("text"),
            interrupted=bool(turn.get("interrupted")),
            turn_duration_seconds=turn.get("turn_duration_seconds"),
            user_bot_latency_seconds=turn.get("user_bot_latency_seconds"),
            llm_model=turn.get("llm_model"),
            prompt_tokens=prompt if llm_requests else None,
            cached_prompt_tokens=cached if llm_requests else None,
            uncached_prompt_tokens=max(prompt - cached, 0) if llm_requests else None,
            completion_tokens=int(turn.get("completion_tokens") or 0) if llm_requests else None,
            ttfb_seconds=turn.get("ttfb_seconds"),
            ttfat_seconds=turn.get("ttfat_seconds"),
            tts_model=turn.get("tts_model"),
            tts_characters=int(turn.get("tts_characters") or 0) if tts_requests else None,
            tts_ttfb_seconds=turn.get("tts_ttfb_seconds"),
            tts_ttfa_seconds=turn.get("tts_ttfa_seconds"),
            stt_model=turn.get("stt_model"),
            stt_audio_seconds=round(stt_audio, 3) if stt_requests else None,
            tool_names=[call.name for call in tool_calls if call.name],
            tool_latency_seconds=[call.latency_seconds for call in tool_calls],
            tool_calls=tool_calls,
        )
