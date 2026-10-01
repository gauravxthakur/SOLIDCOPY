# Pipecat Local WebRTC Agent

Voice assistant built with Pipecat (Deepgram STT, Google Gemini LLM, Cartesia TTS, Silero VAD) featuring full observability parity with LiveKit.

## Architecture & Observability

- **Local Metrics Accumulation**: Framework-neutral tracking of LLM tokens (prompt, cached, completion), TTFT/TTFB/TTFAT latencies, TTS character counts, STT audio durations, and turn latencies.
- **Cost Accounting & Rate Cards**: Strict model-based cost calculation with cached token accounting and explicit `missing_rate` tracking (never silent zero).
- **Atomic Persistence & Checkpointing**: Writes session summaries and per-turn checkpoints atomically to `metrics/sessions/{session_id}.json`.
- **Langfuse & OpenTelemetry Tracing**: Optional by default tracing integration with unified session IDs, native `TurnTraceObserver` attachment, and graceful shutdown flushing.
- **Offline Analytics**: `metrics.langfuse_report` CLI for cross-session aggregates and local-vs-Langfuse validation.

For detailed operational documentation, see [OPS.md](OPS.md) and [metrics/PARITY.md](metrics/PARITY.md).

## Quick Start

### 1. Configure Credentials (.env)
```env
GOOGLE_API_KEY=your_google_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
CARTESIA_API_KEY=your_cartesia_api_key

# Optional Langfuse tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### 2. Run WebRTC Agent
```powershell
cd agents/pipecat_agent
uv run python agent_local_webrtc.py
```

### 3. Run Offline Langfuse Report
```powershell
uv run python -m metrics.langfuse_report --hours 24
uv run python -m metrics.langfuse_report --session-json metrics/sessions/<session_id>.json
```

### 4. Run Test Suite
```powershell
uv run python -m unittest discover -s tests -t .
```

