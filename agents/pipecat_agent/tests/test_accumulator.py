import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.accumulator import RunningStats, SessionMetricsAccumulator
from metrics.types import SESSION_SUMMARY_TOP_LEVEL_KEYS, session_summary_to_dict


class RunningStatsTests(unittest.TestCase):
    def test_ignores_none_and_malformed_values(self):
        stats = RunningStats()
        stats.add(None)
        stats.add("bad")
        stats.add(1.5)
        stats.add(0.5)
        summary = stats.summary()
        self.assertEqual(summary.count, 2)
        self.assertEqual(summary.total, 2.0)
        self.assertEqual(summary.average, 1.0)
        self.assertEqual(summary.minimum, 0.5)
        self.assertEqual(summary.maximum, 1.5)


class SessionMetricsAccumulatorTests(unittest.TestCase):
    def setUp(self):
        self.acc = SessionMetricsAccumulator(
            session_id="sess-1",
            llm_model="gemini-2.5-flash",
            stt_model="nova",
            tts_model="cartesia",
        )

    def test_collect_routes_usage_and_latency(self):
        self.acc.collect(
            {
                "kind": "llm",
                "processor": "GoogleLLMService",
                "model": "gemini-2.5-flash",
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cache_read_input_tokens": 40,
                "cache_creation_input_tokens": 5,
                "reasoning_tokens": 2,
            }
        )
        self.acc.collect(
            {
                "kind": "stt",
                "processor": "DeepgramSTTService",
                "model": "nova",
                "audio_seconds": 1.25,
            }
        )
        self.acc.collect(
            {
                "kind": "tts",
                "processor": "CartesiaTTSService",
                "model": "cartesia",
                "characters": 42,
            }
        )
        self.acc.collect(
            {
                "kind": "ttfat",
                "processor": "GoogleLLMService",
                "model": "gemini-2.5-flash",
                "seconds": 0.4,
                "ttfb_secs": 0.2,
                "thinking_time_secs": 0.2,
            }
        )
        self.acc.collect(
            {
                "kind": "ttfa",
                "processor": "CartesiaTTSService",
                "model": "cartesia",
                "seconds": 0.15,
                "ttfb_secs": 0.12,
                "leading_silence_secs": 0.03,
            }
        )
        self.acc.collect(
            {
                "kind": "ttfb",
                "processor": "CartesiaTTSService",
                "model": "cartesia",
                "seconds": 0.11,
            }
        )

        summary = self.acc.summary()
        self.assertEqual(summary.events.metric_event_count, 6)
        self.assertEqual(summary.llm.request_count, 1)
        self.assertEqual(summary.llm.prompt_tokens, 100)
        self.assertEqual(summary.llm.cached_prompt_tokens, 40)
        self.assertEqual(summary.llm.uncached_prompt_tokens, 60)
        self.assertEqual(summary.llm.completion_tokens, 20)
        self.assertEqual(summary.llm.total_tokens, 120)
        self.assertEqual(summary.llm.cache_creation_input_tokens, 5)
        self.assertEqual(summary.llm.reasoning_tokens, 2)
        self.assertEqual(summary.llm.ttfat_seconds.count, 1)
        self.assertEqual(summary.llm.ttfb_seconds.count, 1)
        self.assertEqual(summary.llm.tokens_per_second.status, "unavailable")

        self.assertEqual(summary.stt.metric_event_count, 1)
        self.assertEqual(summary.stt.audio_duration_seconds.total, 1.25)
        self.assertEqual(summary.tts.request_count, 1)
        self.assertEqual(summary.tts.characters, 42)
        self.assertEqual(summary.tts.ttfa_seconds.count, 1)
        self.assertGreaterEqual(summary.tts.ttfb_seconds.count, 1)
        self.assertEqual(summary.tts.audio_duration_seconds.status, "unavailable")

    def test_collect_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            self.acc.collect({"kind": "nope", "seconds": 1})

    def test_turn_lifecycle_and_assistant_text(self):
        self.acc.note_turn_started(1)
        self.acc.collect(
            {
                "kind": "llm",
                "processor": "LLM",
                "model": "gemini-2.5-flash",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cache_read_input_tokens": 0,
            }
        )
        self.acc.note_assistant_text("Hello")
        self.acc.note_user_bot_latency({"latency_seconds": 1.5})
        self.acc.note_turn_ended(
            {
                "turn_count": 1,
                "duration_secs": 2.0,
                "was_interrupted": False,
                "status": "completed",
            }
        )
        self.acc.note_turn_started(2)
        self.acc.note_turn_ended(
            {
                "turn_count": 2,
                "duration_secs": 0.7,
                "was_interrupted": True,
                "status": "interrupted",
            }
        )

        summary = self.acc.summary()
        self.assertEqual(summary.turns.count, 2)
        self.assertEqual(summary.turns.completed_count, 1)
        self.assertEqual(summary.turns.interrupted_count, 1)
        self.assertEqual(summary.turns.interruption_rate_percentage, 50.0)
        self.assertEqual(summary.interruptions.interrupted_turn_count, 1)
        self.assertEqual(len(summary.turns.records), 2)

        first = summary.turns.records[0]
        self.assertEqual(first.turn_id, "turn-0001")
        self.assertEqual(first.text, "Hello")
        self.assertEqual(first.prompt_tokens, 10)
        self.assertEqual(first.completion_tokens, 5)
        self.assertEqual(first.user_bot_latency_seconds, 1.5)
        self.assertFalse(first.interrupted)

        second = summary.turns.records[1]
        self.assertEqual(second.turn_id, "turn-0002")
        self.assertTrue(second.interrupted)

    def test_tools_transcript_startup_transport(self):
        self.acc.note_final_transcript({"finalized": True, "text": "hi"})
        self.acc.note_final_transcript({"finalized": False, "text": "partial"})
        self.acc.note_tool_started(
            {
                "event": "started",
                "function_name": "get_current_weather",
                "tool_call_id": "c1",
            }
        )
        self.acc.note_tool_result(
            {
                "event": "result",
                "function_name": "get_current_weather",
                "tool_call_id": "c1",
                "ok": True,
                "error": None,
            }
        )
        self.acc.note_tool_cancelled(
            {
                "event": "cancelled",
                "function_name": "get_current_weather",
                "tool_call_id": "c2",
                "ok": False,
            }
        )
        self.acc.note_startup(
            {
                "total_duration_secs": 0.5,
                "setup_phase_secs": 0.2,
                "start_phase_secs": 0.3,
                "processor_timings": [
                    {
                        "processor_name": "DeepgramSTTService",
                        "start_offset_secs": 0.0,
                        "duration_secs": 0.1,
                        "setup_duration_secs": 0.05,
                        "start_duration_secs": 0.05,
                    }
                ],
                "warmup": {"duration_secs": 0.01, "blocking_duration_secs": 0.0},
            }
        )
        self.acc.note_transport(
            {"client_connected_secs": 0.8, "bot_connected_secs": None}
        )
        self.acc.note_first_bot_speech_latency(2.5)
        self.acc.note_latency_breakdown(
            {
                "user_turn_secs": 0.2,
                "function_calls": [
                    {
                        "function_name": "get_current_weather",
                        "duration_secs": 0.2,
                    }
                ],
            }
        )

        summary = self.acc.summary()
        self.assertEqual(summary.stt.utterance_count, 1)
        self.assertEqual(summary.tools.count, 2)
        self.assertEqual(summary.tools.successful_count, 1)
        self.assertEqual(summary.tools.failed_count, 1)
        self.assertEqual(summary.tools.cancelled_count, 1)
        self.assertEqual(summary.tools.by_name["get_current_weather"], 2)
        self.assertEqual(summary.runtime.client_connected_secs, 0.8)
        self.assertEqual(summary.runtime.startup.total_duration_secs, 0.5)
        self.assertEqual(len(summary.runtime.startup.processor_timings), 1)
        self.assertEqual(summary.turns.first_bot_speech_latency_seconds, 2.5)
        self.assertEqual(summary.endpointing.user_turn_seconds.count, 1)

    def test_summary_dict_matches_schema_keys(self):
        typed = self.acc.summary()
        payload = session_summary_to_dict(typed)
        self.assertEqual(tuple(payload.keys()), SESSION_SUMMARY_TOP_LEVEL_KEYS)
        self.assertEqual(payload["session"]["session_id"], "sess-1")
        self.assertEqual(self.acc.summary_dict()["schema_version"], "pipecat-observability-1")

    def test_no_pipecat_import_in_accumulator_class_module_contract(self):
        # Core API must accept plain dicts; this is the observer contract.
        acc = SessionMetricsAccumulator()
        acc.note_service_usage(
            {
                "kind": "tts",
                "processor": "TTS",
                "model": None,
                "characters": 3,
                "audio_seconds": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "cache_read_input_tokens": None,
                "cache_creation_input_tokens": None,
                "reasoning_tokens": None,
                "input_audio_tokens": None,
                "output_audio_tokens": None,
                "cache_read_input_audio_tokens": None,
                "timestamp": 1.0,
            }
        )
        self.assertEqual(acc.summary().tts.characters, 3)

    def test_service_usage_audio_and_reasoning_tokens(self):
        acc = SessionMetricsAccumulator(session_id="sess-audio")
        acc.collect(
            {
                "kind": "llm",
                "processor": "OpenAILLMService",
                "model": "gpt-4o-audio-preview",
                "prompt_tokens": 150,
                "completion_tokens": 80,
                "total_tokens": 230,
                "cache_read_input_tokens": 50,
                "cache_creation_input_tokens": 20,
                "reasoning_tokens": 15,
                "input_audio_tokens": 100,
                "output_audio_tokens": 60,
                "cache_read_input_audio_tokens": 30,
            }
        )
        summary = acc.summary()
        self.assertEqual(summary.llm.prompt_tokens, 150)
        self.assertEqual(summary.llm.cached_prompt_tokens, 50)
        self.assertEqual(summary.llm.uncached_prompt_tokens, 100)
        self.assertEqual(summary.llm.completion_tokens, 80)
        self.assertEqual(summary.llm.total_tokens, 230)
        self.assertEqual(summary.llm.cache_creation_input_tokens, 20)
        self.assertEqual(summary.llm.reasoning_tokens, 15)
        self.assertEqual(summary.llm.input_audio_tokens, 100)
        self.assertEqual(summary.llm.output_audio_tokens, 60)
        self.assertEqual(summary.llm.cache_read_input_audio_tokens, 30)
        self.assertEqual(summary.llm.models.get("gpt-4o-audio-preview"), 1)

    def test_service_latency_ttfb_routing_for_various_processors(self):
        acc = SessionMetricsAccumulator()
        # TTS TTFB routing via processor name
        acc.collect({"kind": "ttfb", "processor": "AzureTTSService", "seconds": 0.18})
        acc.collect({"kind": "ttfb", "processor": "ElevenLabsTTSService", "seconds": 0.22})
        # LLM TTFB routing via processor name
        acc.collect({"kind": "ttfb", "processor": "GroqLLMService", "seconds": 0.09})
        acc.collect({"kind": "ttfb", "processor": "AnthropicLLMService", "seconds": 0.35})

        summary = acc.summary()
        self.assertEqual(summary.tts.ttfb_seconds.count, 2)
        self.assertEqual(summary.tts.ttfb_seconds.minimum, 0.18)
        self.assertEqual(summary.tts.ttfb_seconds.maximum, 0.22)
        self.assertEqual(summary.llm.ttfb_seconds.count, 2)
        self.assertEqual(summary.llm.ttfb_seconds.minimum, 0.09)
        self.assertEqual(summary.llm.ttfb_seconds.maximum, 0.35)

    def test_stt_audio_stats_and_multiple_chunks(self):
        acc = SessionMetricsAccumulator(stt_model="deepgram-nova-2")
        acc.collect({"kind": "stt", "processor": "DeepgramSTTService", "audio_seconds": 1.0})
        acc.collect({"kind": "stt", "processor": "DeepgramSTTService", "audio_seconds": 2.5})
        acc.collect({"kind": "stt", "processor": "DeepgramSTTService", "audio_seconds": 0.5})

        summary = acc.summary()
        self.assertEqual(summary.stt.metric_event_count, 3)
        self.assertEqual(summary.stt.audio_duration_seconds.count, 3)
        self.assertEqual(summary.stt.audio_duration_seconds.total, 4.0)
        self.assertEqual(summary.stt.audio_duration_seconds.minimum, 0.5)
        self.assertEqual(summary.stt.audio_duration_seconds.maximum, 2.5)
        self.assertAlmostEqual(summary.stt.audio_duration_seconds.average, 1.333, places=3)
        self.assertEqual(summary.stt.models.get("deepgram-nova-2"), 3)

    def test_malformed_and_missing_values_handled_gracefully(self):
        acc = SessionMetricsAccumulator()
        # Non-numeric or missing values should not crash the accumulator
        acc.note_service_usage(
            {
                "kind": "llm",
                "processor": "LLM",
                "prompt_tokens": "not-an-int",
                "completion_tokens": None,
                "total_tokens": "invalid",
            }
        )
        acc.note_service_usage(
            {
                "kind": "tts",
                "processor": "TTS",
                "characters": "abc",
            }
        )
        acc.note_service_usage(
            {
                "kind": "stt",
                "processor": "STT",
                "audio_seconds": "not-a-float",
            }
        )
        acc.note_service_latency(
            {
                "kind": "ttfb",
                "processor": "LLM",
                "seconds": "bad-float",
            }
        )
        summary = acc.summary()
        self.assertEqual(summary.llm.request_count, 1)
        self.assertEqual(summary.llm.prompt_tokens, 0)
        self.assertEqual(summary.llm.completion_tokens, 0)
        self.assertEqual(summary.tts.characters, 0)
        self.assertEqual(summary.stt.audio_duration_seconds.count, 0)
        self.assertEqual(summary.llm.ttfb_seconds.count, 0)


if __name__ == "__main__":
    unittest.main()
