import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekit.agents import metrics

from metrics.analyzer import SessionMetricsAccumulator


class SessionMetricsAccumulatorTests(unittest.TestCase):
    def llm(self, prompt, completion, cached, ttft=0.5, speed=10):
        return metrics.LLMMetrics(
            label="openai/gpt-4.1-mini",
            request_id=str(prompt),
            timestamp=0,
            duration=1,
            ttft=ttft,
            cancelled=False,
            completion_tokens=completion,
            prompt_tokens=prompt,
            prompt_cached_tokens=cached,
            total_tokens=prompt + completion,
            tokens_per_second=speed,
        )

    def tts(self, duration, audio, characters=10):
        return metrics.TTSMetrics(
            label="cartesia/sonic-3",
            request_id=str(duration),
            timestamp=0,
            ttfb=0.1,
            duration=duration,
            audio_duration=audio,
            cancelled=False,
            characters_count=characters,
            streamed=True,
        )

    def stt(self, audio):
        return metrics.STTMetrics(
            label="assemblyai/universal-streaming",
            request_id=str(audio),
            timestamp=0,
            duration=0.2,
            audio_duration=audio,
            streamed=True,
        )

    def test_events_are_counted_once_and_legitimate_llm_requests_both_count(self):
        accumulator = SessionMetricsAccumulator()
        accumulator.collect(self.llm(100, 10, 60))
        accumulator.collect(self.llm(200, 20, 100))

        summary = accumulator.summary()
        self.assertEqual(summary["events"]["metric_event_count"], 2)
        self.assertEqual(summary["llm"]["request_count"], 2)
        self.assertEqual(summary["llm"]["prompt_tokens"], 300)
        self.assertEqual(summary["llm"]["completion_tokens"], 30)

    def test_cached_tokens_are_not_added_to_prompt_tokens(self):
        accumulator = SessionMetricsAccumulator()
        accumulator.collect(self.llm(100, 10, 60))

        llm = accumulator.summary()["llm"]
        self.assertEqual(llm["prompt_tokens"], 100)
        self.assertEqual(llm["cached_prompt_tokens"], 60)
        self.assertEqual(llm["uncached_prompt_tokens"], 40)
        self.assertEqual(llm["total_tokens"], 110)

    def test_missing_values_are_omitted_and_zero_is_preserved(self):
        from metrics.analyzer import _Statistics

        missing = _Statistics()
        missing.add(None)
        self.assertEqual(missing.summary()["count"], 0)
        self.assertIsNone(missing.summary()["average"])

        zero = _Statistics()
        zero.add(0)
        result = zero.summary()
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["average"], 0)
        self.assertEqual(result["minimum"], 0)
        self.assertEqual(result["maximum"], 0)

    def test_tts_processing_duration_and_audio_duration_are_separate(self):
        accumulator = SessionMetricsAccumulator()
        accumulator.collect(self.tts(duration=2, audio=8))

        tts = accumulator.summary()["tts"]
        self.assertEqual(tts["duration_seconds"]["total"], 2)
        self.assertEqual(tts["audio_duration_seconds"]["total"], 8)

    def test_stt_event_count_is_not_presented_as_utterance_count(self):
        accumulator = SessionMetricsAccumulator()
        accumulator.collect(self.stt(3))
        accumulator.collect(self.stt(4))

        stt = accumulator.summary()["stt"]
        self.assertEqual(stt["metric_event_count"], 2)
        self.assertEqual(stt["utterance_count"], 0)

        accumulator.note_final_transcript()
        accumulator.note_final_transcript()
        self.assertEqual(accumulator.summary()["stt"]["utterance_count"], 2)
        self.assertEqual(accumulator.summary()["stt"]["metric_event_count"], 2)

    def test_turns_tools_and_ttfa_from_session_events(self):
        accumulator = SessionMetricsAccumulator()

        class Msg:
            role = "assistant"
            interrupted = False
            metrics = {
                "e2e_latency": 2.0,
                "end_of_turn_delay": 0.5,
                "llm_node_ttft": 0.8,
                "tts_node_ttfb": 0.1,
            }

        class Call:
            name = "docs_search"
            call_id = "c1"

        class Out:
            is_error = False

        accumulator.note_assistant_message(Msg())
        accumulator.note_ttfa(1.25)
        accumulator.note_function_tools_executed([Call()], [Out()])
        accumulator.note_tool_started("c1", "docs_search")
        accumulator.note_tool_ended("c1", "done")

        summary = accumulator.summary()
        self.assertEqual(summary["turns"]["count"], 1)
        self.assertEqual(summary["turns"]["completed_count"], 1)
        self.assertEqual(summary["turns"]["end_to_end_latency_seconds"]["average"], 2.0)
        self.assertEqual(summary["turns"]["ttfa_seconds"]["average"], 1.25)
        self.assertEqual(summary["tools"]["count"], 1)
        self.assertEqual(summary["tools"]["by_name"]["docs_search"], 1)
        self.assertEqual(summary["tools"]["successful_count"], 1)
        self.assertGreaterEqual(summary["tools"]["duration_seconds"]["count"], 1)

    def test_turn_records_usage_text_and_tool_latency(self):
        accumulator = SessionMetricsAccumulator(session_id="session-1")

        class Msg:
            role = "assistant"
            text = "Here is the answer."
            interrupted = True
            metrics = {"e2e_latency": 3.25}

        accumulator.collect(self.llm(100, 20, 60, ttft=1.2, speed=16))
        accumulator.collect(self.tts(duration=1.0, audio=2.5, characters=24))
        accumulator.collect(self.stt(4.0))
        accumulator.note_tool_started("tool-1", "docs_search")
        accumulator.note_tool_ended("tool-1", "done")
        accumulator.note_assistant_message(Msg())

        record = accumulator.summary()["turns"]["records"][0]
        self.assertEqual(record["session_id"], "session-1")
        self.assertEqual(record["turn_id"], "turn-0001")
        self.assertEqual(record["text"], "Here is the answer.")
        self.assertEqual(record["llm_model"], "openai/gpt-4.1-mini")
        self.assertEqual(record["prompt_tokens"], 100)
        self.assertEqual(record["cached_prompt_tokens"], 60)
        self.assertEqual(record["uncached_prompt_tokens"], 40)
        self.assertEqual(record["completion_tokens"], 20)
        self.assertEqual(record["ttft_seconds"], 1.2)
        self.assertEqual(record["tokens_per_second"], 16.0)
        self.assertEqual(record["tts_characters"], 24)
        self.assertEqual(record["tts_audio_seconds"], 2.5)
        self.assertEqual(record["stt_audio_seconds"], 4.0)
        self.assertEqual(record["tool_names"], ["docs_search"])
        self.assertEqual(len(record["tool_latency_seconds"]), 1)
        self.assertGreaterEqual(record["tool_latency_seconds"][0], 0)
        self.assertEqual(record["turn_duration_seconds"], 3.25)
        self.assertTrue(record["interrupted"])

    def test_turn_and_session_costs_use_configured_rate_card(self):
        from metrics.analyzer import SessionMetricsAccumulator
        from metrics.costs import RateCard

        rate_card = RateCard.from_dict({
            "llm": {
                "openai/gpt-4.1-mini": {
                    "cached_input_per_token": "0.000001",
                    "uncached_input_per_token": "0.000002",
                    "completion_per_token": "0.000003",
                }
            },
            "stt": {
                "assemblyai/universal-streaming": {"per_audio_second": "0.0001"}
            },
            "tts": {
                "cartesia/sonic-3": {
                    "per_character": "0.00001",
                    "billing_basis": "characters",
                }
            },
        })
        accumulator = SessionMetricsAccumulator(session_id="session-cost", rate_card=rate_card)

        class Msg:
            role = "assistant"
            text = "Done"
            interrupted = False
            metrics = {"e2e_latency": 1.0}

        accumulator.collect(self.llm(100, 20, 60))
        accumulator.collect(self.tts(duration=1.0, audio=2.5, characters=24))
        accumulator.collect(self.stt(4.0))
        accumulator.note_assistant_message(Msg())

        summary = accumulator.summary()
        breakdown = summary["turns"]["records"][0]["cost_breakdown"]
        self.assertEqual(breakdown["status"], "measured")
        self.assertEqual(breakdown["total_cost_usd"], 0.00084)
        self.assertEqual(summary["cost_breakdown"]["total_cost_usd"], 0.00084)
        self.assertEqual(summary["cost_breakdown"]["lines"]["llm"]["status"], "measured")

    def test_missing_rate_is_not_reported_as_zero_cost(self):
        accumulator = SessionMetricsAccumulator()

        class Msg:
            role = "assistant"
            interrupted = False
            metrics = {}

        accumulator.collect(self.llm(10, 2, 5))
        accumulator.note_assistant_message(Msg())

        summary = accumulator.summary()
        self.assertEqual(summary["cost_breakdown"]["lines"]["llm"]["status"], "missing_rate")
        self.assertIsNone(summary["cost_breakdown"]["lines"]["llm"]["cost_usd"])
        self.assertEqual(summary["cost_breakdown"]["total_cost_usd"], 0)

    def test_credit_simulation_tracks_plan_balance_usage_and_revenue(self):
        from decimal import Decimal

        from metrics.analyzer import SessionMetricsAccumulator
        from metrics.costs import CreditAccount

        account = CreditAccount(
            plan_name="standard",
            customer_rate_per_second=Decimal("0.10"),
            credit_balance=Decimal("100"),
        )
        accumulator = SessionMetricsAccumulator(
            session_id="credits-1",
            credit_account=account,
        )

        class Msg:
            role = "assistant"
            interrupted = False
            metrics = {"e2e_latency": 12.5}

        accumulator.note_assistant_message(Msg())
        summary = accumulator.summary()
        credit = summary["credit_simulation"]
        self.assertEqual(credit["plan_name"], "standard")
        self.assertEqual(credit["credits_used"], 12.5)
        self.assertEqual(credit["credits_remaining"], 87.5)
        self.assertEqual(credit["projected_seconds_left"], 87.5)
        self.assertEqual(credit["customer_revenue_inr"], 1.25)
        self.assertEqual(
            summary["turns"]["records"][0]["credit_simulation"]["credits_used"],
            12.5,
        )

    def test_checkpoint_writes_after_turn_and_deduplicates_turn_key(self):
        import tempfile

        accumulator = SessionMetricsAccumulator(session_id="checkpoint-1")

        class Msg:
            role = "assistant"
            interrupted = False
            metrics = {"e2e_latency": 2.0}

        accumulator.note_assistant_message(Msg())
        with tempfile.TemporaryDirectory() as tmp:
            first = accumulator.checkpoint_after_turn(tmp)
            second = accumulator.checkpoint_after_turn(tmp)
            self.assertEqual(first, second)
            loaded = json.loads(first.read_text(encoding="utf-8"))
            records = loaded["turns"]["records"]
            self.assertEqual(len(records), 1)
            self.assertEqual(
                f"{loaded['session']['session_id']}:{records[0]['turn_id']}",
                "checkpoint-1:turn-0001",
            )

    def test_interruption_durations_remain_separate(self):
        accumulator = SessionMetricsAccumulator()
        accumulator.collect(metrics.InterruptionMetrics(
            timestamp=0,
            total_duration=3,
            prediction_duration=1,
            detection_delay=0.2,
            num_interruptions=1,
            num_backchannels=0,
            num_requests=1,
        ))

        interruptions = accumulator.summary()["interruptions"]
        self.assertEqual(interruptions["prediction_duration_seconds"]["total"], 1)
        self.assertEqual(interruptions["total_duration_seconds"]["total"], 3)

    def test_averages_use_only_values_that_exist(self):
        from metrics.analyzer import _Statistics

        values = _Statistics()
        values.add(1)
        values.add(None)
        values.add(3)

        result = values.summary()
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["average"], 2)

    def test_empty_summary_is_valid_and_json_serializable(self):
        summary = SessionMetricsAccumulator().summary()
        self.assertIsNone(summary["llm"]["ttft_seconds"]["average"])
        self.assertIsNone(summary["tts"]["streamed_percentage"])
        self.assertIsNone(summary["interruptions"]["event_rate_percentage"])
        json.dumps(summary)

    def test_persist_summary_writes_session_json(self):
        import tempfile
        from pathlib import Path

        from metrics.analyzer import persist_summary

        accumulator = SessionMetricsAccumulator(session_id="console-room-abc123")
        summary = accumulator.summary()
        with tempfile.TemporaryDirectory() as tmp:
            path = persist_summary(summary, directory=tmp)
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "console-room-abc123.json")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["session"]["session_id"], "console-room-abc123")
            self.assertNotIn("messages", loaded)
            self.assertNotIn("system_prompt", loaded)
            self.assertNotIn("prompt_text", loaded)
            self.assertNotIn("transcript_text", loaded)
            self.assertNotIn("user_text", loaded)


class LangfuseReportTests(unittest.TestCase):
    def test_compare_session_reads_object_shaped_langfuse_usage(self):
        from types import SimpleNamespace

        from metrics.costs import RateCard
        from metrics.langfuse_report import compare_session

        class Observation:
            name = "llm_request"
            model = "openai/gpt-4.1-mini"
            total_cost = 0.0001
            time_to_first_token = 0.4
            usage_details = SimpleNamespace(
                input=100,
                output=20,
                input_token_details=SimpleNamespace(cached_tokens=60),
            )

        result = compare_session(
            {
                "session": {"session_id": "object-session"},
                "llm": {},
                "turns": {"records": []},
            },
            [Observation()],
            RateCard(),
        )

        langfuse = result["llm"]["langfuse"]
        self.assertEqual(langfuse["model"], ["openai/gpt-4.1-mini"])
        self.assertEqual(langfuse["input_tokens"], 100)
        self.assertEqual(langfuse["cached_tokens"], 60)
        self.assertEqual(langfuse["completion_tokens"], 20)
        self.assertEqual(langfuse["cost_usd"], 0.0001)
        self.assertEqual(langfuse["ttft_seconds"]["count"], 1)

    def test_compare_session_reports_unavailable_cached_usage_as_none(self):
        from types import SimpleNamespace

        from metrics.costs import RateCard
        from metrics.langfuse_report import compare_session

        observation = SimpleNamespace(
            name="llm_request",
            model="openai/gpt-4.1-mini",
            usage_details=SimpleNamespace(input=100, output=20),
            total_cost=0.0001,
            time_to_first_token=0.4,
        )
        result = compare_session(
            {"session": {"session_id": "no-cache-field"}, "turns": {"records": []}},
            [observation],
            RateCard(),
        )

        self.assertIsNone(result["llm"]["langfuse"]["cached_tokens"])

    def test_compare_session_marks_used_stt_tts_without_rates_as_missing_rate(self):
        from metrics.costs import RateCard
        from metrics.langfuse_report import compare_session

        result = compare_session(
            {
                "session": {"session_id": "used-no-rates"},
                "turns": {
                    "records": [
                        {
                            "stt_model": "deepgram/nova-3:en",
                            "stt_audio_seconds": 4,
                            "tts_model": "cartesia/sonic-3",
                            "tts_characters": 20,
                            "tts_audio_seconds": 2,
                        }
                    ]
                },
            },
            [],
            RateCard(),
        )

        self.assertEqual(result["own_rate_card_costs"]["stt"]["status"], "missing_rate")
        self.assertEqual(result["own_rate_card_costs"]["tts"]["status"], "missing_rate")

    def test_compare_session_marks_unused_stt_tts_not_applicable(self):
        from metrics.costs import RateCard
        from metrics.langfuse_report import compare_session

        result = compare_session(
            {"session": {"session_id": "unused"}, "turns": {"records": [{}]}},
            [],
            RateCard(),
        )

        self.assertEqual(result["own_rate_card_costs"]["stt"]["status"], "not_applicable")
        self.assertEqual(result["own_rate_card_costs"]["tts"]["status"], "not_applicable")

    def test_compare_session_reports_request_count_mismatch(self):
        from metrics.costs import RateCard
        from metrics.langfuse_report import compare_session

        result = compare_session(
            {
                "session": {"session_id": "count-check"},
                "llm": {"request_count": 5},
                "turns": {"records": []},
            },
            [
                {"name": "llm_request", "session_id": "count-check"},
                {"name": "llm_request", "session_id": "count-check"},
                {"name": "llm_request", "session_id": "other-session"},
            ],
            RateCard(),
        )

        check = result["request_count_check"]
        self.assertEqual(check["status"], "mismatch")
        self.assertEqual(check["local"], 5)
        self.assertEqual(check["langfuse"], 2)
        self.assertEqual(check["difference"], -3)

    def test_compare_session_checks_llm_and_calculates_local_stt_tts_costs(self):
        from metrics.costs import RateCard
        from metrics.langfuse_report import compare_session

        local_summary = {
            "session": {"session_id": "s1"},
            "cost_breakdown": {"lines": {}},
            "turns": {
                "records": [
                    {
                        "llm_model": "openai/gpt-4.1-mini",
                        "prompt_tokens": 100,
                        "cached_prompt_tokens": 60,
                        "completion_tokens": 20,
                        "ttft_seconds": 0.4,
                        "stt_model": "deepgram/nova-3:en",
                        "stt_audio_seconds": 4,
                        "tts_model": "cartesia/sonic-3",
                        "tts_characters": 24,
                        "tts_audio_seconds": 2,
                    }
                ]
            },
        }
        rate_card = RateCard.from_dict({
            "llm": {
                "openai/gpt-4.1-mini": {
                    "cached_input_per_token": "0.000001",
                    "uncached_input_per_token": "0.000002",
                    "completion_per_token": "0.000003",
                }
            },
            "stt": {"deepgram/nova-3:en": {"per_audio_second": "0.01"}},
            "tts": {
                "cartesia/sonic-3": {
                    "per_character": "0.001",
                    "billing_basis": "characters",
                }
            },
        })
        result = compare_session(
            local_summary,
            [{
                "name": "llm_request",
                "model": "openai/gpt-4.1-mini",
                "input_tokens": 100,
                "cached_input_tokens": 60,
                "output_tokens": 20,
                "time_to_first_token": 0.4,
                "total_cost": 0.0001,
            }],
            rate_card,
        )

        self.assertEqual(result["llm"]["local"]["request_count"], 1)
        self.assertEqual(result["llm"]["langfuse"]["request_count"], 1)
        self.assertEqual(result["llm"]["differences"]["input_tokens"]["local"], 100)
        self.assertEqual(result["llm"]["langfuse"]["cost_usd"], 0.0001)
        self.assertEqual(result["own_rate_card_costs"]["stt"]["cost_usd"], 0.04)
        self.assertEqual(result["own_rate_card_costs"]["tts"]["cost_usd"], 0.024)

    def test_aggregate_uses_exact_names_and_tool_failure_rate(self):
        from metrics.langfuse_report import aggregate_observations

        observations = [
            {
                "name": "llm_request",
                "type": "GENERATION",
                "session_id": "s1",
                "latency": 1.0,
                "time_to_first_token": 0.4,
                "total_cost": 0.01,
                "level": "DEFAULT",
            },
            {
                "name": "llm_node",
                "type": "SPAN",
                "session_id": "s1",
                "latency": 9.0,
                "total_cost": 0.0,
                "level": "DEFAULT",
            },
            {
                "name": "docs_search",
                "type": "TOOL",
                "session_id": "s1",
                "latency": 0.2,
                "total_cost": 0.0,
                "level": "ERROR",
            },
            {
                "name": "docs_search",
                "type": "TOOL",
                "session_id": "s1",
                "latency": 0.3,
                "total_cost": 0.0,
                "level": "DEFAULT",
            },
        ]
        report = aggregate_observations(observations)
        self.assertEqual(report["canonical"]["llm_request"]["count"], 1)
        self.assertNotIn("llm_node", report["canonical"])
        self.assertEqual(report["other_names"]["llm_node"], 1)
        self.assertEqual(report["total_cost_usd"], 0.01)
        self.assertEqual(report["tools"]["count"], 2)
        self.assertEqual(report["tools"]["error_count"], 1)
        self.assertEqual(report["tools"]["failure_rate_percentage"], 50.0)
        self.assertEqual(report["latency"]["llm_request"]["p50"], 1.0)

    def test_report_only_emits_applicable_latency_metrics(self):
        from metrics.langfuse_report import aggregate_observations

        report = aggregate_observations([
            {"name": "llm_request", "latency": 1.0, "time_to_first_token": 0.0},
            {"name": "tts_request", "latency": 2.0, "ttfb": 0.0},
            {"name": "stt_request", "latency": 0.5, "audio_duration": 3.0},
            {"name": "eou_detection", "latency": 0.0, "time_to_first_token": 1.0},
            {"name": "agent_turn", "latency": 4.0, "time_to_first_token": 1.0},
            {"name": "agent_session", "latency": 8.0, "time_to_first_token": 1.0},
        ])

        self.assertEqual(report["canonical"]["llm_request"]["ttft_seconds"]["count"], 1)
        self.assertEqual(report["canonical"]["llm_request"]["ttft_seconds"]["average"], 0.0)
        self.assertEqual(report["canonical"]["tts_request"]["ttfb_seconds"]["average"], 0.0)
        self.assertEqual(
            report["canonical"]["stt_request"]["audio_duration_seconds"]["average"], 3.0
        )
        for name in ("tts_request", "eou_detection", "agent_turn", "agent_session"):
            self.assertNotIn("ttft_seconds", report["canonical"][name])
        self.assertIn("session_duration_seconds", report["canonical"]["agent_session"])
        self.assertNotIn("latency_seconds", report["canonical"]["agent_session"])

    def test_report_preserves_zero_cost_and_zero_latency(self):
        from metrics.langfuse_report import aggregate_observations

        report = aggregate_observations([
            {
                "name": "llm_request",
                "latency": 0.0,
                "time_to_first_token": 0.0,
                "total_cost": 0.0,
            }
        ])

        llm = report["canonical"]["llm_request"]
        self.assertEqual(llm["cost_usd"], 0.0)
        self.assertEqual(llm["latency_seconds"]["average"], 0.0)
        self.assertEqual(llm["ttft_seconds"]["average"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
