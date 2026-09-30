"""Framework-neutral session summary schema for Pipecat observability.

No Pipecat imports. Shapes mirror useful LiveKit summary outcomes while marking
fields Pipecat cannot populate as explicit ``unavailable`` / ``n_a`` rather than
inventing proxies.

Serialize with ``session_summary_to_dict`` / ``empty_session_summary``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Literal


class Availability(StrEnum):
    """Whether a summary field can be populated for this agent."""

    AVAILABLE = "available"
    APPROXIMATE = "approximate"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "n_a"


class CostLineStatus(StrEnum):
    """Cost line status (filled by costs module in a later step)."""

    MEASURED = "measured"
    MISSING_RATE = "missing_rate"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class StatSummary:
    """count / average / min / max / total — local parity with LiveKit; no percentiles."""

    count: int = 0
    average: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    total: float = 0.0


@dataclass(frozen=True)
class UnavailableMetric:
    """Explicit stand-in when a LiveKit-like metric has no Pipecat source."""

    status: Literal["unavailable"] = "unavailable"
    reason: str = ""


@dataclass(frozen=True)
class NotApplicableMetric:
    """Explicit stand-in when a LiveKit concept does not apply to this agent."""

    status: Literal["n_a"] = "n_a"
    reason: str = ""


def empty_stats() -> StatSummary:
    return StatSummary()


def unavailable(reason: str) -> UnavailableMetric:
    return UnavailableMetric(reason=reason)


def not_applicable(reason: str) -> NotApplicableMetric:
    return NotApplicableMetric(reason=reason)


# ---------------------------------------------------------------------------
# Section shapes
# ---------------------------------------------------------------------------


@dataclass
class SessionMeta:
    session_id: str | None = None
    started_at: str | None = None  # UTC ISO-8601
    ended_at: str | None = None
    duration_seconds: float | None = None


@dataclass
class EventsSection:
    metric_event_count: int = 0


@dataclass
class CostLine:
    status: CostLineStatus = CostLineStatus.NOT_APPLICABLE
    cost_usd: float | None = None


@dataclass
class CostBreakdown:
    currency: str = "USD"
    total_cost_usd: float | None = None
    lines: dict[str, CostLine] = field(
        default_factory=lambda: {
            "llm": CostLine(),
            "stt": CostLine(),
            "tts": CostLine(),
        }
    )
    turns_with_missing_rates: int = 0


@dataclass
class CreditSimulation:
    """Placeholder until branch-2 cost/credit wiring. Not a billing ledger."""

    plan_name: str | None = None
    customer_rate_inr_per_second: float | None = None
    credit_unit: str = "1 connected second"
    connected_seconds_source: str = "unset"
    credits_used: float | None = None
    credits_remaining: float | None = None
    projected_seconds_left: float | None = None
    customer_revenue_inr: float | None = None


@dataclass
class LlmSection:
    request_count: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    uncached_prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: int = 0
    reasoning_tokens: int = 0
    # Approximate: Pipecat LLM ttfb / ttfat, not LiveKit LLMMetrics.ttft
    ttfb_seconds: StatSummary = field(default_factory=empty_stats)
    ttfat_seconds: StatSummary = field(default_factory=empty_stats)
    tokens_per_second: UnavailableMetric = field(
        default_factory=lambda: unavailable(
            "Pipecat metrics types do not emit tokens_per_second"
        )
    )
    models: dict[str, int] = field(default_factory=dict)


@dataclass
class TtsSection:
    request_count: int = 0
    characters: int = 0
    ttfb_seconds: StatSummary = field(default_factory=empty_stats)
    ttfa_seconds: StatSummary = field(default_factory=empty_stats)
    leading_silence_seconds: StatSummary = field(default_factory=empty_stats)
    audio_duration_seconds: UnavailableMetric = field(
        default_factory=lambda: unavailable(
            "Pipecat TTS usage reports characters only; no audio_duration"
        )
    )
    streamed_count: NotApplicableMetric = field(
        default_factory=lambda: not_applicable(
            "Streaming is normal for this TTS path; no streamed flag"
        )
    )
    models: dict[str, int] = field(default_factory=dict)


@dataclass
class SttSection:
    metric_event_count: int = 0
    utterance_count: int = 0
    audio_duration_seconds: StatSummary = field(default_factory=empty_stats)
    models: dict[str, int] = field(default_factory=dict)


@dataclass
class EndpointingSection:
    """Pipecat turn-release / smart-turn approximations — not LiveKit EOUMetrics."""

    availability: Availability = Availability.APPROXIMATE
    user_turn_seconds: StatSummary = field(default_factory=empty_stats)
    turn_prediction_event_count: int = 0
    turn_prediction_duration_seconds: StatSummary = field(default_factory=empty_stats)
    livekit_eou_fields: UnavailableMetric = field(
        default_factory=lambda: unavailable(
            "No LiveKit EOUMetrics equivalent (end_of_utterance_delay, "
            "transcription_delay, callback_delay)"
        )
    )


@dataclass
class ToolCallRecord:
    name: str | None = None
    tool_call_id: str | None = None
    latency_seconds: float | None = None
    ok: bool | None = None
    error: str | None = None


@dataclass
class TurnRecord:
    session_id: str | None = None
    turn_id: str | None = None
    text: str | None = None  # assistant response text; retained deliberately
    interrupted: bool = False
    turn_duration_seconds: float | None = None
    user_bot_latency_seconds: float | None = None
    llm_model: str | None = None
    prompt_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    uncached_prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ttfb_seconds: float | None = None
    ttfat_seconds: float | None = None
    tts_model: str | None = None
    tts_characters: int | None = None
    tts_ttfb_seconds: float | None = None
    tts_ttfa_seconds: float | None = None
    stt_model: str | None = None
    stt_audio_seconds: float | None = None
    tool_names: list[str] = field(default_factory=list)
    tool_latency_seconds: list[float | None] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    cost_breakdown: dict[str, Any] | None = None
    credit_simulation: dict[str, Any] | None = None


@dataclass
class TurnsSection:
    count: int = 0
    completed_count: int = 0
    interrupted_count: int = 0
    interruption_rate_percentage: float | None = None
    # Approximate vs LiveKit turn e2e: user silence → bot speech
    user_bot_latency_seconds: StatSummary = field(default_factory=empty_stats)
    first_bot_speech_latency_seconds: float | None = None
    llm_ttfb_seconds: StatSummary = field(default_factory=empty_stats)
    llm_ttfat_seconds: StatSummary = field(default_factory=empty_stats)
    tts_ttfb_seconds: StatSummary = field(default_factory=empty_stats)
    tts_ttfa_seconds: StatSummary = field(default_factory=empty_stats)
    records: list[TurnRecord] = field(default_factory=list)


@dataclass
class InterruptionsSection:
    """Turn-level interruption only; LiveKit InterruptionMetrics are unavailable."""

    interrupted_turn_count: int = 0
    provider_interruption_metrics: UnavailableMetric = field(
        default_factory=lambda: unavailable(
            "No InterruptionMetrics (probability, detection/prediction delays)"
        )
    )
    backchannel_count: UnavailableMetric = field(
        default_factory=lambda: unavailable("No backchannel signal in this Pipecat path")
    )


@dataclass
class ToolsSection:
    count: int = 0
    successful_count: int = 0
    failed_count: int = 0
    cancelled_count: int = 0
    by_name: dict[str, int] = field(default_factory=dict)
    duration_seconds: StatSummary = field(default_factory=empty_stats)


@dataclass
class ProcessorStartupTiming:
    processor_name: str
    start_offset_secs: float = 0.0
    duration_secs: float = 0.0
    setup_duration_secs: float = 0.0
    start_duration_secs: float = 0.0


@dataclass
class StartupSection:
    total_duration_secs: float | None = None
    setup_phase_secs: float | None = None
    start_phase_secs: float | None = None
    processor_timings: list[ProcessorStartupTiming] = field(default_factory=list)
    warmup_duration_secs: float | None = None
    warmup_blocking_duration_secs: float | None = None


@dataclass
class RuntimeSection:
    client_connected_secs: float | None = None
    bot_connected_secs: float | None = None
    startup: StartupSection = field(default_factory=StartupSection)
    interrupted_cycle_count: int = 0  # approximate cancel signal via interruptions
    vad_event_count: UnavailableMetric = field(
        default_factory=lambda: unavailable(
            "Silero VAD is configured but no LiveKit-like VADMetrics are emitted"
        )
    )
    preemptive_generation: NotApplicableMetric = field(
        default_factory=lambda: not_applicable(
            "Preemptive generation is not a Pipecat concept in this agent"
        )
    )


@dataclass
class SessionSummary:
    """Top-level session summary document produced by the future accumulator."""

    schema_version: str = "pipecat-observability-1"
    session: SessionMeta = field(default_factory=SessionMeta)
    events: EventsSection = field(default_factory=EventsSection)
    cost_breakdown: CostBreakdown = field(default_factory=CostBreakdown)
    credit_simulation: CreditSimulation = field(default_factory=CreditSimulation)
    llm: LlmSection = field(default_factory=LlmSection)
    tts: TtsSection = field(default_factory=TtsSection)
    stt: SttSection = field(default_factory=SttSection)
    endpointing: EndpointingSection = field(default_factory=EndpointingSection)
    turns: TurnsSection = field(default_factory=TurnsSection)
    interruptions: InterruptionsSection = field(default_factory=InterruptionsSection)
    tools: ToolsSection = field(default_factory=ToolsSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)


SESSION_SUMMARY_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "schema_version",
    "session",
    "events",
    "cost_breakdown",
    "credit_simulation",
    "llm",
    "tts",
    "stt",
    "endpointing",
    "turns",
    "interruptions",
    "tools",
    "runtime",
)


def empty_session_summary(session_id: str | None = None) -> SessionSummary:
    """Return a fully keyed empty summary with explicit unavailable markers."""
    return SessionSummary(session=SessionMeta(session_id=session_id))


def _value_to_jsonable(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, (UnavailableMetric, NotApplicableMetric, StatSummary)):
        return asdict(value)
    if isinstance(value, list):
        return [_value_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _value_to_jsonable(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {key: _value_to_jsonable(item) for key, item in asdict(value).items()}
    return value


def session_summary_to_dict(summary: SessionSummary) -> dict[str, Any]:
    """Convert a SessionSummary into a JSON-serializable dict."""
    return _value_to_jsonable(summary)
