# LiveKit → Pipecat observability parity

Target: signals wired through `livekit_agent/simple_agent_new.py` and its metrics helpers. Equivalent **data/outcomes**, not identical APIs.

**Statuses:** `available` | `approximate` | `unavailable` | `N/A`

**Pipecat sources** (this agent: `agent_local_webrtc.py`, Pipecat 1.10):

- `ServiceMetricsObserver`
- `TurnTrackingObserver`
- `UserBotLatencyObserver`
- `StartupTimingObserver`
- `MetricsFrame` (`enable_metrics` / `enable_usage_metrics`)
- transport connect/disconnect
- tool frames (`FunctionCallInProgressFrame` / `FunctionCallResultFrame` / cancel)

---

## Session / identity

| LiveKit signal | Status | Pipecat source |
| --- | --- | --- |
| session_id (room name) | approximate | UUID or transport session id (no room) |
| session start/end + duration | available | worker/transport lifecycle + monotonic clock |
| metric_event_count | available | count observer / `MetricsFrame` records |

## LLM

| LiveKit signal | Status | Pipecat source |
| --- | --- | --- |
| request count | available | `on_service_usage` kind=llm |
| prompt / completion / total tokens | available | `ServiceUsageRecord` token fields |
| cached prompt tokens | available | `cache_read_input_tokens` (map carefully; Pipecat prompt may be net or gross of cache) |
| cache write / reasoning / audio tokens | available | Pipecat-only extras on same record; keep if useful |
| request duration | approximate | `ProcessingMetricsData` on `MetricsFrame` (`ServiceMetricsObserver` skips it) |
| TTFT | approximate | LLM `ttfb` / `ttfat` via `on_service_latency` (not LiveKit `LLMMetrics.ttft`) |
| tokens_per_second | unavailable | not emitted by Pipecat metrics types |
| cancelled LLM requests | approximate | interruptions / cancelled cycles, not a per-request flag |
| model counts | available | `record.model` on usage/latency |
| RealtimeModelMetrics | N/A | cascaded STT/LLM/TTS agent, not realtime S2S |

## TTS

| LiveKit signal | Status | Pipecat source |
| --- | --- | --- |
| request count / characters | available | `on_service_usage` kind=tts |
| TTFB | available | `on_service_latency` kind=ttfb (TTS processor) |
| TTFA (+ leading silence) | available | `on_service_latency` kind=ttfa (Pipecat-native) |
| audio_duration | unavailable | TTS usage is characters only |
| request duration | approximate | `ProcessingMetricsData` if collected manually |
| cancelled / streamed counts | approximate / N/A | interrupt ≈ cancel; streaming is normal, no streamed flag |

## STT

| LiveKit signal | Status | Pipecat source |
| --- | --- | --- |
| usage audio_seconds | available | `on_service_usage` kind=stt (deltas; sum them) |
| metric / request count | available | count of STT usage records |
| processing duration | approximate | `ProcessingMetricsData` if collected manually |
| model counts | available | `record.model` |
| final user utterance count | available | `TranscriptionFrame` (final) or equivalent |

## EOU / turn detection

| LiveKit signal | Status | Pipecat source |
| --- | --- | --- |
| EOU delay / transcription delay / callback delay | approximate | `LatencyBreakdown.user_turn_secs` + contributions; not LiveKit `EOUMetrics` fields |
| EOT inference count / duration | approximate | `TurnMetricsData` / smart-turn on `MetricsFrame` (`ServiceMetricsObserver` skips these) |

## Turns

| LiveKit signal | Status | Pipecat source |
| --- | --- | --- |
| turn start/end, duration | available | `TurnTrackingObserver` `on_turn_*` |
| completed vs interrupted | available | `on_turn_ended(..., was_interrupted)` |
| interruption % | available | derived from turn counts |
| end-to-end / user→bot latency | approximate | `on_latency_measured` (user stop → bot start); different definition than LiveKit turn e2e |
| turn-level LLM TTFT / TTS TTFB | available | attribute service latency records to active turn |
| EOU→speaking TTFA (custom) | approximate | prefer Pipecat TTFA + `on_latency_measured`; do not invent LiveKit EOU timestamp math |
| per-turn assistant text | available | collect LLM/assistant text frames or context (retained on purpose, same as LiveKit) |
| latency breakdown / tool spans | available | `on_latency_breakdown` (Pipecat-native richer detail) |

## Interruptions / backchannel / VAD

| LiveKit signal | Status | Pipecat source |
| --- | --- | --- |
| interrupted turn flag | available | `was_interrupted` on turn end |
| InterruptionMetrics (prob, delays, counts) | unavailable | no equivalent provider metrics |
| backchannel_count | unavailable | — |
| VAD event count / inference ms | unavailable | Silero is configured; no LiveKit-like `VADMetrics` |

## Tools

| LiveKit signal | Status | Pipecat source |
| --- | --- | --- |
| call count / name / duration | available | `FunctionCallInProgressFrame` / `FunctionCallResultFrame`; also `LatencyBreakdown.function_calls` |
| success / failure | approximate | infer from result/cancel frames; no LiveKit `function_tools_executed` batch event |
| per-turn tool lists | available | correlate tool frames to active turn |

## Runtime / preemptive

| LiveKit signal | Status | Pipecat source |
| --- | --- | --- |
| startup timings | available | `StartupTimingObserver` (Pipecat-native; LiveKit layer has no equivalent section) |
| client connected timing | available | `on_transport_timing_report` / `on_client_connected` |
| preemptive_generation counters | N/A | LiveKit dormant; not a Pipecat concept here |

## Costs / credits / persistence

| LiveKit signal | Status | Pipecat source |
| --- | --- | --- |
| rate card + LLM/STT/TTS costs | available | port `costs.py`; keys must match Pipecat model labels |
| missing_rate / measured statuses | available | same semantics |
| credit simulation | approximate | same simulation idea; connected-time source must be Pipecat transport/session, not assumed turn duration |
| atomic JSON summary + checkpoint | available | port persistence; checkpoint on `on_turn_ended` |
| format_summary + shutdown once-guard | available | same pattern |

## Langfuse / tracing / offline report

| LiveKit signal | Status | Pipecat source |
| --- | --- | --- |
| trace export + session metadata | available | Pipecat `enable_tracing` + OTel/Langfuse setup (different hookup than LiveKit `set_tracer_provider`) |
| force_flush on shutdown | available | — |
| langfuse_report CLI | available | port offline; verify observation names after traces exist |

## Out of scope (same as LiveKit layer)

Prometheus/OTLP metrics backends, dashboards, alerts, SLOs, retention/ACL policy.

---

## Verified payloads (Pipecat 1.10.0)

Inspected against installed Pipecat and this agent's observer set. Locked by
`tests/test_observers.py` and `metrics/observers.py` mapping helpers.

### `ServiceMetricsObserver`

Emits from `MetricsFrame` only. **Does not** emit for `ProcessingMetricsData`,
`TextAggregationMetricsData`, or `TurnMetricsData`.

**`on_service_latency` → `ServiceLatencyRecord`**

| Field | Notes |
| --- | --- |
| `kind` | `ttfb` \| `ttfa` \| `ttfat` |
| `processor`, `model`, `timestamp` | model may be null |
| `seconds` | primary latency value |
| `ttfb_secs` | set for ttfa/ttfat |
| `leading_silence_secs` | ttfa only |
| `thinking_time_secs` | ttfat only |

**`on_service_usage` → `ServiceUsageRecord`**

| Field | When set |
| --- | --- |
| `kind` | `llm` \| `stt` \| `tts` |
| `processor`, `model`, `timestamp` | always |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | llm |
| `cache_read_input_tokens`, `cache_creation_input_tokens`, `reasoning_tokens`, audio token fields | llm optional |
| `audio_seconds` | stt (incremental deltas; **not** named `seconds`) |
| `characters` | tts (no audio duration) |

Duplicate `MetricsFrame.id` values are ignored.

### `TurnTrackingObserver`

- `on_turn_started(observer, turn_count: int)`
- `on_turn_ended(observer, turn_count: int, duration: float, was_interrupted: bool)`
- Duration uses `FramePushed.timestamp` in **nanoseconds** → seconds.
- First turn starts on `StartFrame`; interruption = user speaks while bot speaking.

### `UserBotLatencyObserver`

- `on_latency_measured(observer, latency_seconds: float)` — user silence → bot speech.
  User silence anchor is `VADUserStoppedSpeakingFrame.timestamp - stop_secs`
  (frame wall clock, **not** observer `time_source`).
- `on_first_bot_speech_latency(observer, latency_seconds: float)` — client connect → first bot speech.
- `on_latency_breakdown(observer, LatencyBreakdown)` fields: `total_secs`,
  `measured_from`, `user_turn_start_time`, `user_turn_secs`, `ttfb[]`,
  `text_aggregation`, `function_calls[]` (`function_name`, `start_time`,
  `duration_secs`), `contributions[]` (`key`, `label`, `owner`, `owner_kind`,
  `start_time`, `duration_secs`).

### `StartupTimingObserver`

- `on_startup_timing_report` → `StartupTimingReport`: `start_time`,
  `total_duration_secs`, `setup_phase_secs`, `start_phase_secs`,
  `processor_timings[]`, `warmup`.
- `on_transport_timing_report` → `TransportTimingReport`: `start_time`,
  `bot_connected_secs`, `client_connected_secs`.

### Tools / transcripts (frames; not wired in `setup_observers` yet)

- `FunctionCallInProgressFrame`: `function_name`, `tool_call_id`, `arguments`,
  `cancel_on_interruption`, `group_id`
- `FunctionCallResultFrame`: same ids + `result`, `error` (error ⇒ failure)
- `FunctionCallCancelFrame`: `function_name`, `tool_call_id`, `run_llm`
- `TranscriptionFrame`: `text`, `user_id`, `timestamp`, `finalized`, `language`

### Agent wiring note

`setup_observers()` currently returns
`[StartupTimingObserver, UserBotLatencyObserver, TurnTrackingObserver, ServiceMetricsObserver]`
and only logs. Tool/transcription frames still need a dedicated observer or
pipeline hook in a later step.
