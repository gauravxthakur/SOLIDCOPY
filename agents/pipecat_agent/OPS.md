# Pipecat Agent Observability & Operations Guide

This document covers configuration, persistence, cost calculation, Langfuse integration, privacy considerations, and observability limits for the Pipecat WebRTC Voice Agent.

---

## 1. Environment Variables

### Core & Session Identity
| Variable | Description | Default / Fallback |
| :--- | :--- | :--- |
| `SESSION_ID` | Explicit session ID for metrics, checkpoints, and tracing | Auto-generated (`webrtc_YYYYMMDD_HHMMSS_<hex>`) |
| `PIPECAT_SUMMARY_DIR` / `METRICS_SUMMARY_DIR` | Directory where session summaries and turn checkpoints are persisted | `agents/pipecat_agent/metrics/sessions` |

### Rate Cards & Cost Accounting
| Variable | Description | Default / Fallback |
| :--- | :--- | :--- |
| `PIPECAT_RATE_CARD_PATH` / `RATE_CARD_PATH` | Path to rate card JSON file containing model pricing | Built-in default rate card |
| `LLM_RATE_JSON` | Inline JSON string configuring LLM rates | Rate card fallback |
| `STT_RATE_JSON` | Inline JSON string configuring STT rates | Rate card fallback |
| `TTS_RATE_JSON` | Inline JSON string configuring TTS rates | Rate card fallback |

### Credit Simulation
| Variable | Description | Default / Fallback |
| :--- | :--- | :--- |
| `PLAN_NAME` | Simulated plan tier name | `"standard"` |
| `CUSTOMER_RATE_INR_PER_SECOND` | Billing rate in INR per connected second | `0.10` |
| `INITIAL_BALANCE_INR` | Initial credit balance in INR | `1000.0` |

### Langfuse & OpenTelemetry Tracing
| Variable | Description | Default / Fallback |
| :--- | :--- | :--- |
| `LANGFUSE_PUBLIC_KEY` / `PIPECAT_LANGFUSE_PUBLIC_KEY` | Langfuse public API key | Optional (tracing disabled if missing) |
| `LANGFUSE_SECRET_KEY` / `PIPECAT_LANGFUSE_SECRET_KEY` | Langfuse secret API key | Optional |
| `LANGFUSE_BASE_URL` / `LANGFUSE_HOST` / `PIPECAT_LANGFUSE_BASE_URL` | Langfuse backend host / base URL | Optional |
| `LANGFUSE_ENVIRONMENT` / `ENVIRONMENT` | Environment tag (e.g. `production`, `staging`, `local`) | `"local"` |
| `LANGFUSE_RELEASE` / `RELEASE` | Application release tag (e.g. `v1.0.0`) | `None` |

---

## 2. Persistence & Checkpoint Paths

- **Atomic File Writes**:
  All session summaries are written to a temporary file (`{safe_session_id}.json.tmp`) within the target directory and atomically renamed to `{safe_session_id}.json`. This guarantees no partial reads or corrupt state on crash.
- **Turn Checkpoints**:
  On every `on_turn_ended` event from Pipecat's `TurnTrackingObserver`, the accumulator's latest snapshot is checkpointed to `agents/pipecat_agent/metrics/sessions/{session_id}.json`.
- **Final Shutdown Flush**:
  When the client disconnects or during runner shutdown, `SummaryLogger` logs the human-readable summary, writes the final session document, and flushes traces to Langfuse once (guarded against re-entrant calls).

---

## 3. Cost & Credit Simulation Semantics

- **Strict Status Tracking**:
  - `measured`: Accurate cost computed using matched model rate and usage metric.
  - `missing_rate`: Model name was observed, but no rate was configured in the rate card. **Never produces a silent zero USD cost**.
  - `not_applicable`: Service kind was not invoked in this turn/session.
- **Cached Token Accounting**:
  - LLM costs separate cached prompt tokens (`cached_input_per_token`) from uncached prompt tokens (`uncached_input_per_token`).
- **TTS Units**:
  - Evaluated on character count (`per_character`) or audio duration (`per_audio_second`), depending on rate card definition.
- **Credit Simulation Basis**:
  - Uses `transport_client_connected_seconds` when transport timing reports client connection time; otherwise simulates based on completed turn durations (`completed_turn_duration_simulation`).

---

## 4. Privacy & Data Handling

- **Assistant Text Retention**:
  - Assistant utterances are captured on active turns and stored in `TurnRecord.text` inside the session JSON for quality auditing and offline analysis.
- **User Transcripts**:
  - Utterance counts are aggregated by default; full user transcript frames can be stripped or masked before persistence if operating under strict PII / HIPAA / GDPR requirements.
- **Langfuse Observation Hygiene**:
  - Langfuse reporting queries metadata and observation fields (`core,basic,usage,metrics,model`) without exporting conversational payload text unless explicitly enabled.

---

## 5. Metrics Pipecat Cannot Provide (vs. LiveKit)

| Metric | Status | Reason & Pipecat Equivalent |
| :--- | :--- | :--- |
| `tokens_per_second` | **Unavailable** | Pipecat metrics frames provide total/cached token counts and TTFB/TTFAT latency, but do not emit per-token streaming throughput timestamps. |
| `RealtimeModelMetrics` | **N/A** | This agent runs a cascaded STT → LLM → TTS pipeline, not an OpenAI Realtime WebSocket session. |
| `InterruptionMetrics` | **Unavailable** | LiveKit provider-level interruption probabilities and speech durations before interruption are not emitted by Pipecat; Pipecat provides `was_interrupted: bool` on `on_turn_ended`. |
| `VADMetrics` | **Unavailable** | Silero VAD runs internally in the pipeline; per-inference latency and raw probability frames are not published to `ServiceMetricsObserver`. |
| `TTS Audio Duration` | **Unavailable** | Pipecat `CartesiaTTSService` / `ServiceUsageRecord` emits character usage (`characters`), not synthesized audio seconds. |

---

## 6. How to Run & Verify

### Run WebRTC Agent Locally
```powershell
cd agents/pipecat_agent
uv run python agent_local_webrtc.py
```

### Run Offline Langfuse Report & Comparison
```powershell
# Historical 24h summary
uv run python -m metrics.langfuse_report --hours 24

# Compare specific local session with Langfuse traces
uv run python -m metrics.langfuse_report --session-json metrics/sessions/webrtc_20261001_120000_abcd1234.json
```

### Run Test Suite
```powershell
uv run python -m unittest discover -s tests -t .
```
