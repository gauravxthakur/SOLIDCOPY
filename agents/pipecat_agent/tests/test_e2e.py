import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.accumulator import (
    SessionMetricsAccumulator,
    SummaryLogger,
    format_summary,
    generate_session_id,
)
from metrics.costs import CostCalculator, RateCard
from metrics.langfuse import LangfuseConfig, LangfuseTracer, setup_langfuse
from metrics.observers import setup_observers
from metrics.types import CostLineStatus


class PipecatAgentE2EValidationTests(unittest.TestCase):
    """End-to-end lifecycle and validation tests for Pipecat observability."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.sessions_dir = Path(self.tmp_dir.name)

        # Rate card with gemini-2.5-flash and deepgram, but MISSING cartesia TTS rate
        self.rate_card = RateCard.from_dict({
            "llm": {
                "gemini-2.5-flash": {
                    "cached_input_per_token": "0.00000025",
                    "uncached_input_per_token": "0.000001",
                    "completion_per_token": "0.000004",
                }
            },
            "stt": {
                "deepgram": {
                    "per_audio_second": "0.000072",
                }
            },
            "tts": {},  # intentionally empty to validate missing rate behavior
        })

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_full_e2e_session_lifecycle(self):
        # 1. Generate unified session ID
        session_id = generate_session_id(prefix="webrtc_e2e")
        self.assertTrue(session_id.startswith("webrtc_e2e_"))

        # 2. Setup mock Langfuse tracer
        mock_client = MagicMock()
        mock_client.flush.return_value = True
        langfuse_cfg = LangfuseConfig(
            public_key="pk-test",
            secret_key="sk-test",
            base_url="https://cloud.langfuse.com",
            session_id=session_id,
        )
        langfuse_tracer = LangfuseTracer(config=langfuse_cfg, client=mock_client)
        self.assertEqual(langfuse_tracer.session_id, session_id)

        # 3. Setup accumulator & summary logger with custom rate card & persist dir
        accumulator = SessionMetricsAccumulator(
            session_id=session_id,
            llm_model="gemini-2.5-flash",
            stt_model="deepgram",
            tts_model="cartesia",
            rate_card=self.rate_card,
        )
        logged_messages = []
        summary_logger = SummaryLogger(
            accumulator=accumulator,
            log_fn=lambda msg: logged_messages.append(msg),
            persist_dir=self.sessions_dir,
            langfuse_tracer=langfuse_tracer,
        )

        # 4. Setup observers
        observers = setup_observers(accumulator, session_id=session_id)
        self.assertGreaterEqual(len(observers), 4)

        # 5. Simulate Startup & Transport events
        accumulator.note_startup({
            "start_time": 1000.0,
            "total_duration_secs": 0.450,
            "setup_phase_secs": 0.200,
            "start_phase_secs": 0.250,
            "processor_timings": [
                {
                    "processor_name": "DeepgramSTTService",
                    "start_offset_secs": 0.05,
                    "duration_secs": 0.15,
                    "setup_duration_secs": 0.05,
                    "start_duration_secs": 0.10,
                }
            ],
        })
        accumulator.note_transport({
            "start_time": 1000.0,
            "bot_connected_secs": 0.120,
            "client_connected_secs": 0.350,
        })

        # 6. Turn 1: Normal conversation turn with tool call and assistant response text
        accumulator.note_turn_started(1)
        accumulator.note_final_transcript({"text": "What is the weather in SF?", "finalized": True})
        accumulator.note_user_bot_latency(0.420)
        accumulator.note_first_bot_speech_latency(0.420)

        # Tool execution simulation
        accumulator.note_tool_started({"function_name": "get_current_weather", "tool_call_id": "call_001"})
        accumulator.note_tool_result({"function_name": "get_current_weather", "tool_call_id": "call_001", "ok": True})

        # LLM inference simulation
        accumulator.note_service_latency({
            "kind": "ttfat",
            "processor": "GoogleLLMService",
            "model": "gemini-2.5-flash",
            "seconds": 0.310,
            "ttfb_secs": 0.180,
        })
        accumulator.note_service_usage({
            "kind": "llm",
            "processor": "GoogleLLMService",
            "model": "gemini-2.5-flash",
            "prompt_tokens": 150,
            "cache_read_input_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 180,
        })
        accumulator.note_assistant_text("The weather in San Francisco is sunny and 75 degrees.")

        # TTS inference simulation
        accumulator.note_service_latency({
            "kind": "ttfa",
            "processor": "CartesiaTTSService",
            "model": "cartesia",
            "seconds": 0.190,
            "ttfb_secs": 0.110,
            "leading_silence_secs": 0.020,
        })
        accumulator.note_service_usage({
            "kind": "tts",
            "processor": "CartesiaTTSService",
            "model": "cartesia",
            "characters": 53,
        })

        # STT audio duration simulation
        accumulator.note_service_usage({
            "kind": "stt",
            "processor": "DeepgramSTTService",
            "model": "deepgram",
            "audio_seconds": 2.4,
        })

        # End Turn 1 (triggers turn checkpoint)
        accumulator.note_turn_ended({
            "turn_count": 1,
            "duration_secs": 3.8,
            "was_interrupted": False,
        })
        checkpoint_path = accumulator.checkpoint_after_turn(directory=self.sessions_dir)
        self.assertTrue(checkpoint_path.exists())

        # Verify Turn 1 Checkpoint Content on disk
        saved_data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        self.assertEqual(saved_data["session"]["session_id"], session_id)
        self.assertEqual(saved_data["turns"]["count"], 1)
        self.assertEqual(saved_data["turns"]["completed_count"], 1)
        self.assertEqual(saved_data["turns"]["interrupted_count"], 0)
        self.assertEqual(saved_data["tools"]["count"], 1)
        self.assertEqual(saved_data["tools"]["successful_count"], 1)

        # Verify privacy: assistant response text is retained on turn record
        turn_rec = saved_data["turns"]["records"][0]
        self.assertEqual(turn_rec["text"], "The weather in San Francisco is sunny and 75 degrees.")

        # Verify missing rate card behavior (never silent zero):
        # LLM has measured cost: (100 uncached * 1e-6) + (50 cached * 0.25e-6) + (30 * 4e-6) = 0.000100 + 0.0000125 + 0.000120 = 0.0002325 -> 0.000233
        # STT has measured cost: 2.4 * 0.000072 = 0.0001728
        # TTS has MISSING_RATE status because cartesia is not in rate card!
        turn_cost = turn_rec["cost_breakdown"]
        self.assertEqual(turn_cost["status"], "missing_rate")
        self.assertEqual(turn_cost["lines"]["llm"]["status"], "measured")
        self.assertEqual(turn_cost["lines"]["stt"]["status"], "measured")
        self.assertEqual(turn_cost["lines"]["tts"]["status"], "missing_rate")
        self.assertIsNone(turn_cost["lines"]["tts"]["cost_usd"])

        # 7. Turn 2: Interrupted turn simulation
        accumulator.note_turn_started(2)
        accumulator.note_service_usage({
            "kind": "llm",
            "processor": "GoogleLLMService",
            "model": "gemini-2.5-flash",
            "prompt_tokens": 100,
            "cache_read_input_tokens": 0,
            "completion_tokens": 15,
            "total_tokens": 115,
        })
        accumulator.note_assistant_text("Sure, let me help you with...")
        accumulator.note_turn_ended({
            "turn_count": 2,
            "duration_secs": 1.2,
            "was_interrupted": True,
        })

        # 8. Shutdown & SummaryLogger emission
        emitted = summary_logger.emit(reason="shutdown")
        self.assertTrue(emitted)
        mock_client.flush.assert_called_once()

        # Check that once-only guard prevents double emission
        self.assertFalse(summary_logger.emit(reason="shutdown"))

        # Verify final session JSON on disk
        final_file = self.sessions_dir / f"{session_id}.json"
        self.assertTrue(final_file.exists())
        final_summary = json.loads(final_file.read_text(encoding="utf-8"))

        self.assertEqual(final_summary["session"]["session_id"], session_id)
        self.assertEqual(final_summary["turns"]["count"], 2)
        self.assertEqual(final_summary["turns"]["completed_count"], 1)
        self.assertEqual(final_summary["turns"]["interrupted_count"], 1)
        self.assertEqual(final_summary["turns"]["interruption_rate_percentage"], 50.0)

        # Verify human-readable summary output was logged
        summary_log = next((m for m in logged_messages if "SESSION METRICS SUMMARY" in m), None)
        self.assertIsNotNone(summary_log)
        self.assertIn(f"Session ID: {session_id}", summary_log)
        self.assertIn("Langfuse traces flushed successfully.", logged_messages)


if __name__ == "__main__":
    unittest.main()

