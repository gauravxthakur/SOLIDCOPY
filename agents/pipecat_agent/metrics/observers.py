from loguru import logger
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.observers.startup_timing_observer import StartupTimingObserver
from pipecat.observers.service_metrics_observer import ServiceMetricsObserver


def setup_observers():
    """Sets up observers for startup timing, user-bot latency, and turn tracking metrics. 
    Returns a list of observers to be used in the PipelineWorker.
    """
    
    # Measures the time taken by each processor to start up.
    startup_observer = StartupTimingObserver()

    @startup_observer.event_handler("on_startup_timing_report")
    async def on_startup_timing_report(observer, report):
        logger.info(f"Total startup duration: {report.total_duration_secs:.3f}s")
        for timing in report.processor_timings:
            logger.info(f"  {timing.processor_name}: {timing.duration_secs:.3f}s")

    @startup_observer.event_handler("on_transport_timing_report")
    async def on_transport_timing_report(observer, report):
        if report.client_connected_secs is not None:
            logger.info(f"Client connection time: {report.client_connected_secs:.3f}s")
            
            
    # Reports each service metric as a structured record rather than a log line. Emits separate events for latency measurements (TTFB, TTFA, TTFAT) and usage reports (STT audio seconds, TTS characters, LLM token counts).
    service_observer = ServiceMetricsObserver()

    @service_observer.event_handler("on_service_latency")
    async def on_service_latency(observer, record):
        logger.info(f"Service Latency [{record.processor}]: {record.kind} = {record.seconds:.3f}s")

    @service_observer.event_handler("on_service_usage")
    async def on_service_usage(observer, record):
        if record.kind == "llm":
            logger.info(f"LLM Token Usage [{record.processor}]: {record.total_tokens} tokens")
        elif record.kind == "tts":
            logger.info(f"TTS Usage [{record.processor}]: {record.characters} characters")
        elif record.kind == "stt":
            logger.info(f"STT Usage [{record.processor}]: {record.seconds:.2f}s audio")
        
        
    # Measures the time between when a user stops speaking and when the bot starts speaking.
    latency_observer = UserBotLatencyObserver()
    
    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency_seconds):
        logger.info(f"User-to-bot latency: {latency_seconds:.3f}s")
        
    # Tracks conversation turns, emitting events when turns start and end. Handles interruptions and configurable timeouts.
        
    turn_observer = TurnTrackingObserver(turn_end_timeout_secs=2.5)
    
    @turn_observer.event_handler("on_turn_started")
    async def on_turn_started(observer, turn_count):
        logger.info(f"Turn {turn_count} started")

    @turn_observer.event_handler("on_turn_ended")
    async def on_turn_ended(observer, turn_count, duration, was_interrupted):
        status = "interrupted" if was_interrupted else "completed"
        logger.info(f"Turn {turn_count} {status} after {duration:.2f}s")
        
    return [startup_observer, latency_observer, turn_observer, service_observer]