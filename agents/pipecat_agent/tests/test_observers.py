import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ClientConnectedFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    ProcessingMetricsData,
    STTUsage,
    STTUsageMetricsData,
    TextAggregationMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
    TurnMetricsData,
)
from pipecat.observers.service_metrics_observer import ServiceMetricsObserver
from pipecat.observers.startup_timing_observer import (
    StartupTimingObserver,
    StartupTimingReport,
    TransportTimingReport,
)
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver

from metrics.accumulator import SessionMetricsAccumulator
from metrics.observers import (
    latency_breakdown_to_dict,
    latency_measured_to_dict,
    latency_record_to_dict,
    setup_observers,
    startup_report_to_dict,
    tool_cancelled_to_dict,
    tool_result_to_dict,
    tool_started_to_dict,
    transcription_to_dict,
    transport_report_to_dict,
    turn_ended_to_dict,
    usage_record_to_dict,
)
from tests.helpers import ControllableClock, push_frame


class SetupObserversTests(unittest.IsolatedAsyncioTestCase):
    def test_returns_expected_observer_types_in_order(self):
        observers = setup_observers()
        self.assertEqual(len(observers), 4)
        self.assertIsInstance(observers[0], StartupTimingObserver)
        self.assertIsInstance(observers[1], UserBotLatencyObserver)
        self.assertIsInstance(observers[2], TurnTrackingObserver)
        self.assertIsInstance(observers[3], ServiceMetricsObserver)

    async def test_wires_accumulator_and_dispatches_events(self):
        acc = SessionMetricsAccumulator(session_id="wired-session")
        observers = setup_observers(acc)
        service_obs = observers[3]
        turn_obs = observers[2]

        # Push metrics frame to service_obs
        frame = MetricsFrame(
            data=[
                TTSUsageMetricsData(processor="CartesiaTTSService", model="cartesia", value=25),
                TTFBMetricsData(processor="CartesiaTTSService", model="cartesia", value=0.15),
            ]
        )
        await push_frame(service_obs, frame)

        # Push turn start and end to turn_obs
        await push_frame(turn_obs, StartFrame())
        await push_frame(turn_obs, BotStartedSpeakingFrame())
        await push_frame(turn_obs, BotStoppedSpeakingFrame())

        summary = acc.summary()
        self.assertEqual(summary.tts.characters, 25)
        self.assertEqual(summary.tts.ttfb_seconds.count, 1)


class ServiceMetricsObserverTests(unittest.IsolatedAsyncioTestCase):
    async def test_emits_latency_and_usage_records_and_skips_other_metrics(self):
        observer = ServiceMetricsObserver(time_source=lambda: 100.0)
        latency_payloads = []
        usage_payloads = []

        @observer.event_handler("on_service_latency")
        async def on_latency(obs, record):
            latency_payloads.append(latency_record_to_dict(record))

        @observer.event_handler("on_service_usage")
        async def on_usage(obs, record):
            usage_payloads.append(usage_record_to_dict(record))

        frame = MetricsFrame(
            data=[
                TTFBMetricsData(processor="TTS", model="cartesia", value=0.12),
                TTFAMetricsData(
                    processor="TTS",
                    model="cartesia",
                    ttfa=0.15,
                    ttfb=0.12,
                    leading_silence=0.03,
                ),
                TTFATMetricsData(
                    processor="LLM",
                    model="gemini",
                    ttfat=0.4,
                    ttfb=0.2,
                    thinking_time=0.2,
                ),
                LLMUsageMetricsData(
                    processor="LLM",
                    model="gemini",
                    value=LLMTokenUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        cache_read_input_tokens=3,
                    ),
                ),
                STTUsageMetricsData(
                    processor="STT",
                    model="nova",
                    value=STTUsage(audio_seconds=1.25),
                ),
                TTSUsageMetricsData(processor="TTS", model="cartesia", value=42),
                ProcessingMetricsData(processor="LLM", value=0.9),
                TextAggregationMetricsData(processor="TTS", value=0.05),
                TurnMetricsData(
                    processor="smart-turn",
                    is_complete=True,
                    probability=0.9,
                    e2e_processing_time_ms=12.0,
                ),
            ]
        )

        await push_frame(observer, frame)
        await push_frame(observer, frame)  # duplicate frame id ignored

        self.assertEqual(
            [item["kind"] for item in latency_payloads],
            ["ttfb", "ttfa", "ttfat"],
        )
        self.assertEqual(
            [item["kind"] for item in usage_payloads],
            ["llm", "stt", "tts"],
        )

        ttfa = latency_payloads[1]
        self.assertEqual(ttfa["seconds"], 0.15)
        self.assertEqual(ttfa["ttfb_secs"], 0.12)
        self.assertEqual(ttfa["leading_silence_secs"], 0.03)

        llm = usage_payloads[0]
        self.assertEqual(llm["prompt_tokens"], 10)
        self.assertEqual(llm["completion_tokens"], 5)
        self.assertEqual(llm["total_tokens"], 15)
        self.assertEqual(llm["cache_read_input_tokens"], 3)
        self.assertIsNone(llm["audio_seconds"])
        self.assertIsNone(llm["characters"])

        stt = usage_payloads[1]
        self.assertEqual(stt["audio_seconds"], 1.25)
        self.assertIsNone(stt["characters"])
        self.assertNotIn("seconds", stt)

        tts = usage_payloads[2]
        self.assertEqual(tts["characters"], 42)
        self.assertIsNone(tts["audio_seconds"])


class TurnTrackingObserverTests(unittest.IsolatedAsyncioTestCase):
    async def test_turn_start_end_and_interruption_payloads(self):
        observer = TurnTrackingObserver(turn_end_timeout_secs=60.0)
        started = []
        ended = []

        @observer.event_handler("on_turn_started")
        async def on_started(obs, turn_count):
            started.append(turn_count)

        @observer.event_handler("on_turn_ended")
        async def on_ended(obs, turn_count, duration, was_interrupted):
            ended.append(turn_ended_to_dict(turn_count, duration, was_interrupted))

        await push_frame(observer, StartFrame(), timestamp=1_000_000_000)
        await push_frame(observer, BotStartedSpeakingFrame(), timestamp=1_100_000_000)
        await push_frame(observer, BotStoppedSpeakingFrame(), timestamp=2_000_000_000)
        await push_frame(observer, UserStartedSpeakingFrame(), timestamp=2_500_000_000)
        await push_frame(observer, BotStartedSpeakingFrame(), timestamp=3_000_000_000)
        await push_frame(observer, UserStartedSpeakingFrame(), timestamp=3_200_000_000)

        self.assertEqual(started, [1, 2, 3])
        self.assertEqual(len(ended), 2)
        self.assertEqual(ended[0]["turn_count"], 1)
        self.assertEqual(ended[0]["duration_secs"], 1.5)
        self.assertEqual(ended[0]["was_interrupted"], False)
        self.assertEqual(ended[0]["status"], "completed")
        self.assertEqual(ended[1]["turn_count"], 2)
        self.assertAlmostEqual(ended[1]["duration_secs"], 0.7)
        self.assertEqual(ended[1]["was_interrupted"], True)
        self.assertEqual(ended[1]["status"], "interrupted")


class UserBotLatencyObserverTests(unittest.IsolatedAsyncioTestCase):
    async def test_latency_breakdown_includes_tools_and_ttfb(self):
        clock = ControllableClock(0.0)
        observer = UserBotLatencyObserver(time_source=clock)
        measured = []
        breakdowns = []
        first_speech = []

        @observer.event_handler("on_latency_measured")
        async def on_measured(obs, latency_seconds):
            measured.append(latency_measured_to_dict(latency_seconds))

        @observer.event_handler("on_latency_breakdown")
        async def on_breakdown(obs, breakdown):
            breakdowns.append(latency_breakdown_to_dict(breakdown))

        @observer.event_handler("on_first_bot_speech_latency")
        async def on_first(obs, latency_seconds):
            first_speech.append(latency_seconds)

        await push_frame(observer, ClientConnectedFrame(), clock=clock, now=1.0)
        await push_frame(
            observer,
            VADUserStoppedSpeakingFrame(timestamp=2.0, stop_secs=0.0),
            clock=clock,
            now=2.0,
        )
        await push_frame(observer, UserStoppedSpeakingFrame(), clock=clock, now=2.2)
        await push_frame(
            observer,
            FunctionCallInProgressFrame(
                function_name="get_current_weather",
                tool_call_id="call-1",
                arguments={"location": "SF"},
            ),
            clock=clock,
            now=2.3,
        )
        await push_frame(
            observer,
            FunctionCallResultFrame(
                function_name="get_current_weather",
                tool_call_id="call-1",
                arguments={},
                result={"conditions": "sunny"},
            ),
            clock=clock,
            now=2.5,
        )
        await push_frame(
            observer,
            MetricsFrame(
                data=[TTFBMetricsData(processor="LLM", model="gemini", value=0.11)]
            ),
            clock=clock,
            now=2.7,
        )
        await push_frame(observer, BotStartedSpeakingFrame(), clock=clock, now=3.5)

        self.assertEqual(first_speech, [2.5])
        self.assertEqual(len(measured), 1)
        self.assertAlmostEqual(measured[0]["latency_seconds"], 1.5)

        self.assertEqual(len(breakdowns), 1)
        breakdown = breakdowns[0]
        self.assertIn("total_secs", breakdown)
        self.assertIn("contributions", breakdown)
        self.assertAlmostEqual(breakdown["user_turn_secs"], 0.2)
        self.assertEqual(len(breakdown["function_calls"]), 1)
        self.assertEqual(breakdown["function_calls"][0]["function_name"], "get_current_weather")
        self.assertAlmostEqual(breakdown["function_calls"][0]["duration_secs"], 0.2)
        self.assertEqual(breakdown["ttfb"][0]["processor"], "LLM")
        self.assertAlmostEqual(breakdown["ttfb"][0]["duration_secs"], 0.11)


class MappingHelperUnitTests(unittest.TestCase):
    def test_startup_and_transport_report_helpers(self):
        startup = startup_report_to_dict(
            StartupTimingReport(
                start_time=1.0,
                total_duration_secs=0.5,
                setup_phase_secs=0.2,
                start_phase_secs=0.3,
                processor_timings=[],
                warmup=None,
            )
        )
        self.assertEqual(
            startup,
            {
                "start_time": 1.0,
                "total_duration_secs": 0.5,
                "setup_phase_secs": 0.2,
                "start_phase_secs": 0.3,
                "processor_timings": [],
                "warmup": None,
            },
        )

        transport = transport_report_to_dict(
            TransportTimingReport(
                start_time=1.0,
                bot_connected_secs=None,
                client_connected_secs=0.8,
            )
        )
        self.assertEqual(
            transport,
            {
                "start_time": 1.0,
                "bot_connected_secs": None,
                "client_connected_secs": 0.8,
            },
        )

    def test_tool_and_transcription_helpers(self):
        started = tool_started_to_dict(
            FunctionCallInProgressFrame(
                function_name="get_current_weather",
                tool_call_id="call-1",
                arguments={"location": "SF"},
                cancel_on_interruption=True,
                group_id="g1",
            )
        )
        self.assertEqual(started["event"], "started")
        self.assertEqual(started["tool_call_id"], "call-1")
        self.assertTrue(started["cancel_on_interruption"])

        ok = tool_result_to_dict(
            FunctionCallResultFrame(
                function_name="get_current_weather",
                tool_call_id="call-1",
                arguments={},
                result={"ok": True},
            )
        )
        self.assertTrue(ok["ok"])
        self.assertIsNone(ok["error"])

        failed = tool_result_to_dict(
            FunctionCallResultFrame(
                function_name="get_current_weather",
                tool_call_id="call-1",
                arguments={},
                result="tool failed",
                error="boom",
            )
        )
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["error"], "boom")

        cancelled = tool_cancelled_to_dict(
            FunctionCallCancelFrame(
                function_name="get_current_weather",
                tool_call_id="call-1",
                run_llm=False,
            )
        )
        self.assertEqual(cancelled["event"], "cancelled")
        self.assertFalse(cancelled["ok"])

        transcript = transcription_to_dict(
            TranscriptionFrame(
                text="hello",
                user_id="user-1",
                timestamp="2026-01-01T00:00:00Z",
                finalized=True,
            )
        )
        self.assertEqual(
            transcript,
            {
                "text": "hello",
                "user_id": "user-1",
                "timestamp": "2026-01-01T00:00:00Z",
                "finalized": True,
                "language": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
