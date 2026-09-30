import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.types import (
    SESSION_SUMMARY_TOP_LEVEL_KEYS,
    SessionSummary,
    empty_session_summary,
    session_summary_to_dict,
)


class SessionSummarySchemaTests(unittest.TestCase):
    def test_empty_summary_has_stable_top_level_keys(self):
        payload = session_summary_to_dict(empty_session_summary("sess-1"))
        self.assertEqual(tuple(payload.keys()), SESSION_SUMMARY_TOP_LEVEL_KEYS)
        self.assertEqual(payload["schema_version"], "pipecat-observability-1")
        self.assertEqual(payload["session"]["session_id"], "sess-1")

    def test_unavailable_and_n_a_fields_are_explicit(self):
        payload = session_summary_to_dict(empty_session_summary())

        self.assertEqual(payload["llm"]["tokens_per_second"]["status"], "unavailable")
        self.assertIn("tokens_per_second", payload["llm"]["tokens_per_second"]["reason"])

        self.assertEqual(payload["tts"]["audio_duration_seconds"]["status"], "unavailable")
        self.assertEqual(payload["tts"]["streamed_count"]["status"], "n_a")

        self.assertEqual(
            payload["endpointing"]["livekit_eou_fields"]["status"], "unavailable"
        )
        self.assertEqual(payload["endpointing"]["availability"], "approximate")

        self.assertEqual(
            payload["interruptions"]["provider_interruption_metrics"]["status"],
            "unavailable",
        )
        self.assertEqual(payload["interruptions"]["backchannel_count"]["status"], "unavailable")

        self.assertEqual(payload["runtime"]["vad_event_count"]["status"], "unavailable")
        self.assertEqual(payload["runtime"]["preemptive_generation"]["status"], "n_a")

    def test_stats_and_cost_placeholders_have_expected_shape(self):
        payload = session_summary_to_dict(empty_session_summary())

        stats = payload["llm"]["ttfb_seconds"]
        self.assertEqual(
            set(stats),
            {"count", "average", "minimum", "maximum", "total"},
        )
        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["average"])

        self.assertEqual(
            payload["cost_breakdown"]["lines"]["llm"]["status"],
            "not_applicable",
        )
        self.assertIsNone(payload["cost_breakdown"]["lines"]["llm"]["cost_usd"])
        self.assertEqual(payload["credit_simulation"]["connected_seconds_source"], "unset")

    def test_turn_record_and_tools_sections_exist(self):
        payload = session_summary_to_dict(empty_session_summary())
        turns = payload["turns"]
        self.assertEqual(turns["count"], 0)
        self.assertEqual(turns["records"], [])
        self.assertIn("user_bot_latency_seconds", turns)
        self.assertIn("llm_ttfat_seconds", turns)
        self.assertIn("tts_ttfa_seconds", turns)

        tools = payload["tools"]
        self.assertEqual(tools["count"], 0)
        self.assertEqual(tools["cancelled_count"], 0)
        self.assertEqual(tools["by_name"], {})

    def test_summary_dict_is_json_serializable(self):
        payload = session_summary_to_dict(empty_session_summary("json-check"))
        encoded = json.dumps(payload)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["session"]["session_id"], "json-check")
        self.assertIsInstance(decoded, dict)

    def test_session_summary_type_defaults(self):
        summary = SessionSummary()
        self.assertIsInstance(summary, SessionSummary)
        self.assertEqual(summary.events.metric_event_count, 0)
        self.assertEqual(summary.stt.utterance_count, 0)


if __name__ == "__main__":
    unittest.main()
