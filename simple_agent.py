import os
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.util.types import AttributeValue
from livekit.agents.telemetry import set_tracer_provider


import logging

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, room_io
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import stt, tts, llm, inference
from livekit.agents import AgentStateChangedEvent, MetricsCollectedEvent, metrics
from livekit.agents import function_tool, RunContext, ToolError
from livekit.agents import mcp
from metrics.analyzer import SessionMetricsAccumulator, format_summary
import json
import time
import httpx


logger = logging.getLogger(__name__)

load_dotenv()



def setup_langfuse(
    metadata: dict[str, AttributeValue] | None = None,
    *,
    base_url: str | None = None,
    public_key: str | None = None,
    secret_key: str | None = None,
) -> TracerProvider:
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    base_url = base_url or os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    if not public_key or not secret_key or not base_url:
        raise ValueError(
            "LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL (or LANGFUSE_HOST) must be set"
        )
    trace_provider = TracerProvider()
    set_tracer_provider(trace_provider, metadata=metadata)
    Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        tracer_provider=trace_provider,
        should_export_span=lambda span: True,
    )
    return trace_provider


    


# Define your agent's behavior by extending the Agent class
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an upbeat, slightly sarcastic voice AI for tech support. "
                "Help the caller fix issues without rambling, and keep replies under 3 sentences. "
                "You can answer questions about "
                "LiveKit by searching the documentation. When users ask about LiveKit "
                "features, APIs, or how to build something, use the docs search tools "
                "to find accurate information."
            ),  # System prompt for the LLM
        )


server = AgentServer()


# The entrypoint function runs when a participant joins the room
@server.rtc_session()
async def entrypoint(ctx: JobContext):

    session_metrics = SessionMetricsAccumulator()

    trace_provider = setup_langfuse(
        metadata={
            "langfuse.session.id": ctx.room.name,
        }
    )
    
    async def flush_trace():
        trace_provider.force_flush()
    
    ctx.add_shutdown_callback(flush_trace)


    # Configure the voice pipeline with STT, LLM, TTS, and VAD providers
    session = AgentSession(

        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        # preemptive_generation=True,
        #mcp_servers=[mcp.MCPServerHTTP(url="http://docs.livekit.io/mcp"),],
        tools=[
            mcp.MCPToolset(id="livekit-docs", mcp_server=mcp.MCPServerHTTP(url="http://docs.livekit.io/mcp"),),
            #get_airtable_toolset(),
        ],
    )


    # Aggregate data across all conversation turns
    usage_collector = metrics.UsageCollector()

    # Track End of Utterance timing (when turn detector decides user finished speaking)
    last_eou_metrics: metrics.EOUMetrics | None = None
    summary_logged = False

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics

        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics

        usage_collector.collect(ev.metrics)
        session_metrics.collect(ev.metrics)


    async def log_usage(reason: str = "shutdown") -> None:
        # Exactly once per entrypoint: duplicate prints were from dual log handlers,
        # not double calculation — still guard against re-entrant shutdown.
        nonlocal summary_logged
        if summary_logged:
            logger.debug("Skipping duplicate session metrics summary (reason=%s)", reason)
            return
        summary_logged = True

        session_summary = session_metrics.summary()
        logger.info("\n%s", format_summary(session_summary))
        logger.info("Session metrics JSON: %s", json.dumps(session_summary))


    # Fire log_usage when worker shuts down
    ctx.add_shutdown_callback(log_usage)


    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "speaking":
            if last_eou_metrics:
                # Calculate time since user finished speaking
                elapsed = time.time() - last_eou_metrics.timestamp
                logger.info(f"Time to first audio: {elapsed:.3f}s")


    # Start the session with noise cancellation enabled
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),  # Background voice cancellation
            ),
        ),
        # record=False to disable tracking of this agent
    )


if __name__ == "__main__":
    # Do not call logging.basicConfig here: LiveKit's CLI attaches its own root
    # handler. basicConfig would add a second StreamHandler and print every
    # shutdown log twice in two different formats.
    agents.cli.run_app(server)