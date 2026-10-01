import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.accumulator import (
    RunningStats,
    SessionMetricsAccumulator,
    SummaryLogger,
    format_summary,
    generate_session_id,
    get_default_summary_dir,
    persist_summary,
)
from metrics.types import SESSION_SUMMARY_TOP_LEVEL_KEYS, session_summary_to_dict


class SessionIdGenerationTests(unittest.TestCase):
    def test_default_session_id_format(self):
        sess_id = generate_session_id()
        self.assertTrue(sess_id.startswith("sess_"))
        parts = sess_id.split("_")
        self.assertGreaterEqual(len(parts), 3)

    def test_custom_prefix(self):
        sess_id = generate_session_id(prefix="webrtc")
        self.assertTrue(sess_id.startswith("webrtc_"))

    def test_base_name_sanitization(self):
        sess_id = generate_session_id(base_name="Room-123/Special:Name!@#")
        self.assertTrue(sess_id.startswith("Room-123_Special_Name_"))

    def test_empty_base_name_falls_back(self):
        sess_id = generate_session_id(prefix="custom", base_name="!@#$%^")
        self.assertTrue(sess_id.startswith("custom_"))

    def test_accumulator_defaults_session_id(self):
        acc = SessionMetricsAccumulator()
        self.assertIsNotNone(acc.session_id)
        self.assertTrue(acc.session_id.startswith("sess_"))


class RunningStatsTests(unittest.TestCase):
    def test_empty_running_stats(self):
        stats = RunningStats()
        summary = stats.summary()
        self.assertEqual(summary.count, 0)
        self.assertIsNone(summary.average)
        self.assertIsNone(summary.minimum)
        self.assertIsNone(summary.maximum)
        self.assertEqual(summary.total, 0.0)

    def test_single_value(self):
        stats = RunningStats()
        stats.add(3.14159)
        summary = stats.summary(digits=2)
        self.assertEqual(summary.count, 1)
        self.assertEqual(summary.average, 3.14)
        self.assertEqual(summary.minimum, 3.14)
        self.assertEqual(summary.maximum, 3.14)
        self.assertEqual(summary.total, 3.14)

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

    def test_rejects_booleans_nan_inf(self):
        stats = RunningStats()
        stats.add(True)
        stats.add(False)
        stats.add(float("nan"))
        stats.add(float("inf"))
        stats.add(float("-inf"))
        stats.add("nan")
        stats.add("inf")
        stats.add(10.0)
        summary = stats.summary()
        self.assertEqual(summary.count, 1)
        self.assertEqual(summary.total, 10.0)
        self.assertEqual(summary.average, 10.0)

    def test_negative_values_and_zeros(self):
        stats = RunningStats()
        stats.add(-5.0)
        stats.add(0.0)
        stats.add(5.0)
        summary = stats.summary()
        self.assertEqual(summary.count, 3)
        self.assertEqual(summary.total, 0.0)
        self.assertEqual(summary.average, 0.0)
        self.assertEqual(summary.minimum, -5.0)
        self.assertEqual(summary.maximum, 5.0)

    def test_no_p50_or_p95_in_stat_summary(self):
        stats = RunningStats()
        stats.add(1.0)
        summary = stats.summary()
        # Verify StatSummary fields only match count/average/minimum/maximum/total
        fields = set(summary.__dataclass_fields__.keys())
        self.assertEqual(fields, {"count", "average", "minimum", "maximum", "total"})
        self.assertNotIn("p50", fields)
        self.assertNotIn("p95", fields)


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

    def test_turn_interruption_and_interruption_rate(self):
        acc = SessionMetricsAccumulator(session_id="sess-interrupt")
        acc.note_turn_started(1)
        acc.note_turn_ended({"turn_count": 1, "duration_secs": 3.0, "was_interrupted": False})
        acc.note_turn_started(2)
        acc.note_turn_ended({"turn_count": 2, "duration_secs": 1.2, "was_interrupted": True})
        acc.note_turn_started(3)
        acc.note_turn_ended({"turn_count": 3, "duration_secs": 4.5, "was_interrupted": False})
        acc.note_turn_started(4)
        acc.note_turn_ended({"turn_count": 4, "duration_secs": 0.8, "was_interrupted": True})

        summary = acc.summary()
        self.assertEqual(summary.turns.count, 4)
        self.assertEqual(summary.turns.completed_count, 2)
        self.assertEqual(summary.turns.interrupted_count, 2)
        self.assertEqual(summary.turns.interruption_rate_percentage, 50.0)
        self.assertEqual(summary.interruptions.interrupted_turn_count, 2)
        self.assertEqual(summary.runtime.interrupted_cycle_count, 2)

    def test_vad_eou_preemptive_gaps_honestly_documented(self):
        acc = SessionMetricsAccumulator()
        summary = acc.summary()
        self.assertEqual(summary.runtime.vad_event_count.status, "unavailable")
        self.assertIn("Silero VAD", summary.runtime.vad_event_count.reason)
        self.assertEqual(summary.runtime.preemptive_generation.status, "n_a")
        self.assertEqual(summary.endpointing.livekit_eou_fields.status, "unavailable")
        self.assertEqual(summary.interruptions.provider_interruption_metrics.status, "unavailable")
        self.assertEqual(summary.interruptions.backchannel_count.status, "unavailable")

    def test_turn_level_tool_calls_attribution(self):
        acc = SessionMetricsAccumulator(session_id="sess-tools")
        acc.note_turn_started(1)
        acc.note_tool_started({"function_name": "fetch_user", "tool_call_id": "call_1"})
        acc.note_tool_result({"function_name": "fetch_user", "tool_call_id": "call_1", "ok": True})
        acc.note_tool_started({"function_name": "query_db", "tool_call_id": "call_2"})
        acc.note_tool_result(
            {"function_name": "query_db", "tool_call_id": "call_2", "ok": False, "error": "timeout"}
        )
        acc.note_turn_ended({"turn_count": 1, "duration_secs": 2.5, "was_interrupted": False})

        acc.note_turn_started(2)
        acc.note_tool_started({"function_name": "send_email", "tool_call_id": "call_3"})
        acc.note_tool_cancelled({"function_name": "send_email", "tool_call_id": "call_3"})
        acc.note_turn_ended({"turn_count": 2, "duration_secs": 1.0, "was_interrupted": True})

        summary = acc.summary()
        self.assertEqual(summary.tools.count, 3)
        self.assertEqual(summary.tools.successful_count, 1)
        self.assertEqual(summary.tools.failed_count, 2)
        self.assertEqual(summary.tools.cancelled_count, 1)

        turn_1 = summary.turns.records[0]
        self.assertEqual(turn_1.tool_names, ["fetch_user", "query_db"])
        self.assertEqual(len(turn_1.tool_calls), 2)
        self.assertEqual(turn_1.tool_calls[0].tool_call_id, "call_1")
        self.assertTrue(turn_1.tool_calls[0].ok)
        self.assertIsNone(turn_1.tool_calls[0].error)
        self.assertEqual(turn_1.tool_calls[1].tool_call_id, "call_2")
        self.assertFalse(turn_1.tool_calls[1].ok)
        self.assertEqual(turn_1.tool_calls[1].error, "timeout")

        turn_2 = summary.turns.records[1]
        self.assertEqual(turn_2.tool_names, ["send_email"])
        self.assertEqual(len(turn_2.tool_calls), 1)
        self.assertEqual(turn_2.tool_calls[0].tool_call_id, "call_3")
        self.assertFalse(turn_2.tool_calls[0].ok)
        self.assertEqual(turn_2.tool_calls[0].error, "cancelled")

    def test_empty_session_summary_and_serialization(self):
        acc = SessionMetricsAccumulator(session_id="empty-sess")
        summary = acc.summary()
        self.assertEqual(summary.session.session_id, "empty-sess")
        self.assertEqual(summary.events.metric_event_count, 0)
        self.assertEqual(summary.llm.request_count, 0)
        self.assertEqual(summary.llm.total_tokens, 0)
        self.assertEqual(summary.llm.ttfb_seconds.count, 0)
        self.assertIsNone(summary.llm.ttfb_seconds.average)
        self.assertEqual(summary.tts.characters, 0)
        self.assertEqual(summary.tts.ttfb_seconds.count, 0)
        self.assertEqual(summary.stt.metric_event_count, 0)
        self.assertEqual(summary.stt.audio_duration_seconds.count, 0)
        self.assertEqual(summary.turns.count, 0)
        self.assertEqual(len(summary.turns.records), 0)
        self.assertEqual(summary.tools.count, 0)

        # JSON dictionary serializability check
        as_dict = acc.summary_dict()
        self.assertEqual(tuple(as_dict.keys()), SESSION_SUMMARY_TOP_LEVEL_KEYS)
        self.assertEqual(as_dict["llm"]["ttfb_seconds"]["count"], 0)
        self.assertIsNone(as_dict["llm"]["ttfb_seconds"]["average"])


class FormatSummaryTests(unittest.TestCase):
    def _empty_acc(self) -> SessionMetricsAccumulator:
        return SessionMetricsAccumulator(session_id="test-empty-session")

    def test_format_empty_session_has_header(self):
        acc = self._empty_acc()
        out = format_summary(acc.summary())
        self.assertIn("SESSION METRICS SUMMARY", out)

    def test_format_empty_session_contains_required_sections(self):
        acc = self._empty_acc()
        out = format_summary(acc.summary())
        for section in ("LLM", "TTS", "STT", "TURNS", "TOOLS"):
            self.assertIn(section, out, msg=f"Missing section: {section}")

    def test_format_empty_session_shows_na_for_numeric_fields(self):
        acc = self._empty_acc()
        out = format_summary(acc.summary())
        # With no data, average stats should render as "n/a"
        self.assertIn("n/a", out)

    def test_format_empty_session_shows_session_id(self):
        acc = self._empty_acc()
        out = format_summary(acc.summary())
        self.assertIn("test-empty-session", out)

    def test_format_accepts_dataclass(self):
        acc = self._empty_acc()
        out = format_summary(acc.summary())  # summary() returns SessionSummary dataclass
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_format_accepts_dict(self):
        acc = self._empty_acc()
        out = format_summary(acc.summary_dict())  # summary_dict() returns plain dict
        self.assertIsInstance(out, str)
        self.assertIn("SESSION METRICS SUMMARY", out)

    def test_format_populated_session_shows_llm_values(self):
        acc = SessionMetricsAccumulator(session_id="test-populated")
        acc.collect({
            "kind": "llm",
            "processor": "OpenAILLMService",
            "model": "gpt-4o",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cache_read_input_tokens": 0,
        })
        out = format_summary(acc.summary())
        # prompt tokens, completion tokens, and total tokens should appear
        self.assertIn("150", out)  # total tokens
        self.assertIn("100", out)  # prompt tokens
        self.assertIn("50", out)   # completion tokens

    def test_format_populated_session_shows_tts_values(self):
        acc = SessionMetricsAccumulator(session_id="test-tts")
        acc.collect({
            "kind": "tts",
            "processor": "CartesiaTTSService",
            "model": "sonic",
            "characters": 300,
        })
        out = format_summary(acc.summary())
        self.assertIn("300", out)

    def test_format_populated_session_shows_stt_event_count(self):
        acc = SessionMetricsAccumulator(session_id="test-stt")
        acc.collect({
            "kind": "stt",
            "processor": "DeepgramSTTService",
            "model": "nova",
            "audio_seconds": 5.0,
        })
        out = format_summary(acc.summary())
        self.assertIn("STT", out)

    def test_format_populated_session_shows_turn_counts(self):
        acc = SessionMetricsAccumulator(session_id="test-turns")
        acc.note_turn_started(1)
        acc.note_turn_ended({
            "turn_count": 1,
            "duration_secs": 2.0,
            "was_interrupted": False,
            "status": "completed",
        })
        acc.note_turn_started(2)
        acc.note_turn_ended({
            "turn_count": 2,
            "duration_secs": 0.5,
            "was_interrupted": True,
            "status": "interrupted",
        })
        out = format_summary(acc.summary())
        self.assertIn("TURNS", out)
        # 2 total turns should appear somewhere in the output
        self.assertIn("2", out)

    def test_format_no_cost_or_credit_sections(self):
        """Branch 1: format_summary must NOT include cost or credit sections."""
        acc = self._empty_acc()
        out = format_summary(acc.summary())
        self.assertNotIn("cost", out.lower())
        self.assertNotIn("credit", out.lower())
        self.assertNotIn("revenue", out.lower())


class SummaryLoggerTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.persist_dir = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _make_logger(self, log_fn=None):
        acc = SessionMetricsAccumulator(session_id="logger-test")
        return SummaryLogger(acc, log_fn=log_fn, persist_dir=self.persist_dir)

    def test_emitted_property_starts_false(self):
        logger = self._make_logger()
        self.assertFalse(logger.emitted)

    def test_emit_returns_true_on_first_call(self):
        logger = self._make_logger()
        result = logger.emit(reason="test")
        self.assertTrue(result)

    def test_emitted_property_true_after_emit(self):
        logger = self._make_logger()
        logger.emit(reason="test")
        self.assertTrue(logger.emitted)

    def test_emit_returns_false_on_second_call(self):
        logger = self._make_logger()
        logger.emit(reason="first")
        result = logger.emit(reason="second")
        self.assertFalse(result)

    def test_emit_does_not_call_log_fn_twice(self):
        calls = []
        logger = self._make_logger(log_fn=lambda msg: calls.append(msg))
        logger.emit(reason="first")
        count_after_first = len(calls)
        logger.emit(reason="second")
        # log_fn is not called on second emit attempt (once-only guard)
        self.assertEqual(len(calls), count_after_first)

    def test_emit_with_no_log_fn_does_not_raise(self):
        logger = self._make_logger(log_fn=None)
        try:
            result = logger.emit(reason="test")
            self.assertTrue(result)
        except Exception as exc:
            self.fail(f"emit() raised with log_fn=None: {exc}")

    def test_emit_log_fn_receives_formatted_summary(self):
        messages = []
        logger = self._make_logger(log_fn=lambda msg: messages.append(msg))
        logger.emit(reason="test")
        combined = "\n".join(messages)
        self.assertIn("SESSION METRICS SUMMARY", combined)

    def test_emit_log_fn_receives_json(self):
        import json as _json
        messages = []
        logger = self._make_logger(log_fn=lambda msg: messages.append(msg))
        logger.emit(reason="test")
        json_msg = next((m for m in messages if "Session metrics JSON:" in m), None)
        self.assertIsNotNone(json_msg, "No JSON log message found")
        json_str = json_msg.split("Session metrics JSON:", 1)[1].strip()
        parsed = _json.loads(json_str)
        self.assertIn("session", parsed)
        self.assertIn("llm", parsed)

    def test_third_emit_still_returns_false(self):
        logger = self._make_logger()
        logger.emit()
        logger.emit()
        self.assertFalse(logger.emit())


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_persist_summary_creates_json_file(self):
        import json as _json
        acc = SessionMetricsAccumulator(session_id="test-persist-1")
        acc.collect({
            "kind": "llm",
            "processor": "OpenAILLMService",
            "model": "gpt-4o",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cache_read_input_tokens": 0,
        })
        path = persist_summary(acc.summary(), directory=self.test_dir)
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "test-persist-1.json")

        content = _json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(content["session"]["session_id"], "test-persist-1")
        self.assertEqual(content["llm"]["prompt_tokens"], 10)

    def test_atomic_temp_and_replace(self):
        acc = SessionMetricsAccumulator(session_id="atomic-test")
        path = persist_summary(acc.summary_dict(), directory=self.test_dir)
        # Check target file exists and temp file does not remain
        temp_file = self.test_dir / "atomic-test.json.tmp"
        self.assertTrue(path.exists())
        self.assertFalse(temp_file.exists())

    def test_safe_session_id_filenames(self):
        unsafe_ids = [
            ("Room-123/Special:Name!@#", "Room-123_Special_Name.json"),
            ("path\\with\\backslashes", "path_with_backslashes.json"),
            ("   ...   ", "unknown.json"),
            ("../../../etc/passwd", "etc_passwd.json"),
        ]
        for raw_id, expected_filename in unsafe_ids:
            acc = SessionMetricsAccumulator(session_id=raw_id)
            path = persist_summary(acc.summary(), directory=self.test_dir)
            self.assertEqual(path.name, expected_filename, msg=f"Failed for raw_id: {raw_id}")

    def test_configurable_output_dir_via_argument(self):
        acc = SessionMetricsAccumulator(session_id="custom-dir-test")
        custom_sub_dir = self.test_dir / "nested" / "output"
        path = persist_summary(acc.summary(), directory=custom_sub_dir)
        self.assertTrue(path.exists())
        self.assertEqual(path.parent, custom_sub_dir)

    def test_configurable_output_dir_via_env_var(self):
        import os
        custom_env_dir = self.test_dir / "env_output"
        old_env = os.environ.get("PIPECAT_SUMMARY_DIR")
        try:
            os.environ["PIPECAT_SUMMARY_DIR"] = str(custom_env_dir)
            acc = SessionMetricsAccumulator(session_id="env-var-test")
            path = acc.persist()
            self.assertTrue(path.exists())
            self.assertEqual(path.parent, custom_env_dir)
        finally:
            if old_env is not None:
                os.environ["PIPECAT_SUMMARY_DIR"] = old_env
            else:
                os.environ.pop("PIPECAT_SUMMARY_DIR", None)

    def test_failed_write_raises_and_cleans_temp_file(self):
        # Point to a file as directory to force write failure
        blocker = self.test_dir / "blocker_file"
        blocker.write_text("i am a file", encoding="utf-8")

        acc = SessionMetricsAccumulator(session_id="failed-write")
        with self.assertRaises((OSError, FileExistsError, NotADirectoryError)):
            persist_summary(acc.summary(), directory=blocker)

    def test_accumulator_checkpoint_after_turn(self):
        acc = SessionMetricsAccumulator(session_id="turn-checkpoint")
        acc.note_turn_started(1)
        acc.note_turn_ended({
            "turn_count": 1,
            "duration_secs": 1.2,
            "was_interrupted": False,
            "status": "completed",
        })
        path = acc.checkpoint_after_turn(directory=self.test_dir)
        self.assertTrue(path.exists())
        import json as _json
        content = _json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(content["turns"]["count"], 1)

    def test_subsequent_checkpoints_overwrite_same_file(self):
        acc = SessionMetricsAccumulator(session_id="overwrite-test")
        path1 = acc.checkpoint_after_turn(directory=self.test_dir)
        self.assertTrue(path1.exists())

        acc.note_turn_started(1)
        acc.note_turn_ended({
            "turn_count": 1,
            "duration_secs": 2.5,
            "was_interrupted": False,
            "status": "completed",
        })
        path2 = acc.checkpoint_after_turn(directory=self.test_dir)
        self.assertEqual(path1, path2)

        import json as _json
        content = _json.loads(path2.read_text(encoding="utf-8"))
        self.assertEqual(content["turns"]["count"], 1)

    def test_summary_logger_logs_saved_path(self):
        acc = SessionMetricsAccumulator(session_id="logger-saved-test")
        messages = []
        logger = SummaryLogger(acc, log_fn=lambda msg: messages.append(msg), persist_dir=self.test_dir)
        logger.emit(reason="shutdown")
        saved_msg = next((m for m in messages if "Session metrics saved:" in m), None)
        self.assertIsNotNone(saved_msg, "Session metrics saved message not found")
        self.assertIn("logger-saved-test.json", saved_msg)


if __name__ == "__main__":
    unittest.main()

