import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.costs import RateCard
from metrics.langfuse_report import (
    CANONICAL_NAMES,
    _float,
    _latency_stats,
    _local_cost_line,
    _local_llm_summary,
    _obs_get,
    _percentile,
    _sum_usage,
    _usage_value,
    aggregate_observations,
    build_report,
    compare_session,
    fetch_observations,
    main,
)


class LangfuseReportHelperTests(unittest.TestCase):
    def test_float_helper(self):
        self.assertIsNone(_float(None))
        self.assertIsNone(_float("invalid"))
        self.assertEqual(_float(12), 12.0)
        self.assertEqual(_float("3.14"), 3.14)

    def test_percentile_helper(self):
        self.assertIsNone(_percentile([], 50))
        self.assertEqual(_percentile([10.0], 90), 10.0)
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(_percentile(vals, 50), 3.0)
        self.assertAlmostEqual(_percentile(vals, 95), 4.8, places=2)

    def test_latency_stats(self):
        stats = _latency_stats([])
        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["average"])
        self.assertIsNone(stats["p50"])
        self.assertIsNone(stats["p95"])
        self.assertIsNone(stats["maximum"])

        stats = _latency_stats([1.0, 2.0, 3.0])
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["average"], 2.0)
        self.assertEqual(stats["p50"], 2.0)
        self.assertEqual(stats["maximum"], 3.0)

    def test_obs_get(self):
        d = {"name": "test", "val": 123}
        self.assertEqual(_obs_get(d, "name"), "test")
        self.assertEqual(_obs_get(d, "missing", "def"), "def")

        class DummyObj:
            name = "obj_test"
        obj = DummyObj()
        self.assertEqual(_obs_get(obj, "name"), "obj_test")
        self.assertEqual(_obs_get(obj, "missing", 456), 456)

    def test_usage_value_and_sum_usage(self):
        obs1 = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "input_token_details": {"cached_tokens": 20},
            }
        }
        obs2 = {
            "input_tokens": 200,
            "output_tokens": 80,
            "cached_input_tokens": 40,
        }

        self.assertEqual(_usage_value(obs1, "prompt_tokens"), 100.0)
        self.assertEqual(_usage_value(obs1, "completion_tokens"), 50.0)
        self.assertEqual(_usage_value(obs1, "cached_input_tokens"), 20.0)

        self.assertEqual(_usage_value(obs2, "input_tokens"), 200.0)
        self.assertEqual(_usage_value(obs2, "completion_tokens"), 80.0)
        self.assertEqual(_usage_value(obs2, "cached_input_tokens"), 40.0)

        total_input = _sum_usage([obs1, obs2], "input_tokens", "prompt_tokens")
        self.assertEqual(total_input, 300.0)


class AggregateObservationsTests(unittest.TestCase):
    def test_empty_observations(self):
        report = aggregate_observations([])
        self.assertEqual(report["observation_count"], 0)
        self.assertEqual(report["session_count"], 0)
        self.assertEqual(report["total_cost_usd"], 0.0)
        self.assertEqual(report["tools"]["count"], 0)

    def test_canonical_aggregations(self):
        observations = [
            {
                "name": "conversation",
                "type": "span",
                "session_id": "sess_1",
                "latency": 15.5,
            },
            {
                "name": "turn",
                "type": "span",
                "session_id": "sess_1",
                "latency": 2.5,
                "user_bot_latency_seconds": 0.8,
            },
            {
                "name": "llm_request",
                "type": "generation",
                "session_id": "sess_1",
                "latency": 0.9,
                "time_to_first_token": 0.25,
                "total_cost": 0.0015,
            },
            {
                "name": "tts_request",
                "type": "span",
                "session_id": "sess_1",
                "latency": 0.4,
                "time_to_first_byte": 0.12,
                "time_to_first_audio": 0.18,
            },
            {
                "name": "stt_request",
                "type": "span",
                "session_id": "sess_1",
                "latency": 0.3,
                "audio_duration_seconds": 3.2,
            },
            {
                "name": "tool_execution",
                "type": "tool",
                "session_id": "sess_1",
                "latency": 0.5,
                "level": "DEFAULT",
            },
            {
                "name": "tool_execution",
                "type": "tool",
                "session_id": "sess_1",
                "latency": 0.2,
                "level": "ERROR",
            },
        ]

        report = aggregate_observations(observations)
        self.assertEqual(report["observation_count"], 7)
        self.assertEqual(report["session_count"], 1)
        self.assertEqual(report["total_cost_usd"], 0.0015)

        self.assertIn("conversation", report["canonical"])
        self.assertIn("turn", report["canonical"])
        self.assertIn("llm_request", report["canonical"])
        self.assertIn("tts_request", report["canonical"])
        self.assertIn("stt_request", report["canonical"])

        self.assertEqual(report["tools"]["count"], 2)
        self.assertEqual(report["tools"]["error_count"], 1)
        self.assertEqual(report["tools"]["failure_rate_percentage"], 50.0)

        self.assertEqual(report["ttft"]["llm_request"]["average"], 0.25)
        self.assertEqual(report["ttfb"]["tts_request"]["average"], 0.12)
        self.assertEqual(report["audio_duration"]["stt_request"]["average"], 3.2)


class CompareSessionTests(unittest.TestCase):
    def setUp(self):
        self.rate_card = RateCard.from_dict({
            "llm": {
                "gemini-2.5-flash": {
                    "cached_input_per_token": "0.00000025",
                    "uncached_input_per_token": "0.000001",
                    "completion_per_token": "0.000004",
                }
            },
            "stt": {"deepgram": {"per_audio_second": "0.000072"}},
            "tts": {"cartesia": {"per_character": "0.000038"}},
        })
        self.local_session = {
            "session": {"session_id": "sess_compare_123"},
            "llm": {
                "request_count": 2,
                "prompt_tokens": 100,
                "cached_prompt_tokens": 20,
                "completion_tokens": 50,
                "models": {"gemini-2.5-flash": 2},
                "ttfb_seconds": {"average": 0.22, "count": 2},
                "ttfat_seconds": {"average": 0.45, "count": 2},
            },
            "cost_breakdown": {
                "lines": {
                    "llm": {"status": "measured", "cost_usd": 0.000028},
                    "stt": {"status": "measured", "cost_usd": 0.00036},
                    "tts": {"status": "measured", "cost_usd": 0.00095},
                }
            },
            "turns": {
                "records": [
                    {
                        "turn_id": "turn-0001",
                        "llm_model": "gemini-2.5-flash",
                        "prompt_tokens": 50,
                        "cached_prompt_tokens": 10,
                        "completion_tokens": 25,
                        "stt_model": "deepgram",
                        "stt_audio_seconds": 2.5,
                        "tts_model": "cartesia",
                        "tts_characters": 12,
                    },
                    {
                        "turn_id": "turn-0002",
                        "llm_model": "gemini-2.5-flash",
                        "prompt_tokens": 50,
                        "cached_prompt_tokens": 10,
                        "completion_tokens": 25,
                        "stt_model": "deepgram",
                        "stt_audio_seconds": 2.5,
                        "tts_model": "cartesia",
                        "tts_characters": 13,
                    },
                ]
            },
        }

    def test_compare_matching_session(self):
        observations = [
            {
                "name": "llm_request",
                "type": "generation",
                "session_id": "sess_compare_123",
                "model": "gemini-2.5-flash",
                "prompt_tokens": 50,
                "cached_input_tokens": 10,
                "completion_tokens": 25,
                "time_to_first_token": 0.2,
                "total_cost": 0.000014,
            },
            {
                "name": "llm_request",
                "type": "generation",
                "session_id": "sess_compare_123",
                "model": "gemini-2.5-flash",
                "prompt_tokens": 50,
                "cached_input_tokens": 10,
                "completion_tokens": 25,
                "time_to_first_token": 0.24,
                "total_cost": 0.000014,
            },
        ]

        comp = compare_session(self.local_session, observations, rate_card=self.rate_card)
        self.assertEqual(comp["session_id"], "sess_compare_123")
        self.assertEqual(comp["request_count_check"]["status"], "match")
        self.assertEqual(comp["request_count_check"]["local"], 2)
        self.assertEqual(comp["request_count_check"]["langfuse"], 2)
        self.assertEqual(comp["llm"]["local"]["input_tokens"], 100)
        self.assertEqual(comp["llm"]["langfuse"]["input_tokens"], 100)
        self.assertEqual(comp["llm"]["local"]["completion_tokens"], 50)
        self.assertEqual(comp["llm"]["langfuse"]["completion_tokens"], 50)
        self.assertIn("notes", comp)

    def test_compare_mismatched_session(self):
        observations = [
            {
                "name": "llm_request",
                "type": "generation",
                "session_id": "sess_compare_123",
                "model": "gemini-2.5-flash",
                "prompt_tokens": 50,
                "completion_tokens": 25,
            }
        ]

        comp = compare_session(self.local_session, observations, rate_card=self.rate_card)
        self.assertEqual(comp["request_count_check"]["status"], "mismatch")
        self.assertEqual(comp["request_count_check"]["difference"], -1)


class FetchAndBuildReportTests(unittest.TestCase):
    def test_fetch_observations_with_pagination(self):
        mock_client = MagicMock()
        page1 = MagicMock()
        page1.data = [{"name": "obs1"}, {"name": "obs2"}]
        page1.meta.cursor = "cursor-2"

        page2 = MagicMock()
        page2.data = [{"name": "obs3"}]
        page2.meta.cursor = None

        mock_client.api.observations.get_many.side_effect = [page1, page2]

        rows = fetch_observations(mock_client, session_id="sess_page")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["name"], "obs1")
        self.assertEqual(rows[2]["name"], "obs3")
        self.assertEqual(mock_client.api.observations.get_many.call_count, 2)

    def test_build_report_with_mock_client(self):
        mock_client = MagicMock()
        page = MagicMock()
        page.data = [
            {"name": "turn", "session_id": "s1", "latency": 1.5},
            {"name": "llm", "session_id": "s1", "latency": 0.8},
        ]
        page.meta.cursor = None
        mock_client.api.observations.get_many.return_value = page

        report = build_report(session_id="s1", hours=12, client=mock_client)
        self.assertEqual(report["observation_count"], 2)
        self.assertEqual(report["query"]["session_id"], "s1")
        self.assertEqual(report["query"]["hours"], 12)

    def test_build_report_missing_config_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                build_report(session_id="s1")
            self.assertIn("LANGFUSE_PUBLIC_KEY", str(ctx.exception))


class MainCLITests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp_dir.name)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_cli_with_session_json(self):
        session_file = self.path / "test_session.json"
        out_file = self.path / "output.json"
        session_data = {
            "session": {"session_id": "sess_cli_test"},
            "llm": {"request_count": 1, "prompt_tokens": 20, "completion_tokens": 10},
            "turns": {"records": []},
        }
        session_file.write_text(json.dumps(session_data), encoding="utf-8")

        mock_client = MagicMock()
        page = MagicMock()
        page.data = [{"name": "llm_request", "session_id": "sess_cli_test", "prompt_tokens": 20}]
        page.meta.cursor = None
        mock_client.api.observations.get_many.return_value = page

        with patch("sys.argv", ["langfuse_report.py", "--session-json", str(session_file), "--output", str(out_file)]):
            with patch("metrics.langfuse_report.build_report") as mock_build:
                mock_build.return_value = {
                    "observation_count": 1,
                    "session_count": 1,
                    "query": {"session_id": "sess_cli_test"},
                }
                with patch("metrics.langfuse_report.fetch_observations", return_value=page.data):
                    main()

        self.assertTrue(out_file.exists())
        saved = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertIn("session_comparison", saved)
        self.assertEqual(saved["session_comparison"]["session_id"], "sess_cli_test")


if __name__ == "__main__":
    unittest.main()
