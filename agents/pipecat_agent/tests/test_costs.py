import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from decimal import Decimal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.accumulator import SessionMetricsAccumulator
from metrics.costs import CostCalculator, CreditAccount, LLMRate, RateCard, STTRate, TTSRate


class RateCardValidationTests(unittest.TestCase):
    def setUp(self):
        self.sample_valid_dict = {
            "llm": {
                "gemini-2.5-flash": {
                    "cached_input_per_token": "0.0000001",
                    "uncached_input_per_token": "0.0000003",
                    "completion_per_token": "0.0000010",
                },
                "gpt-4o": {
                    "cached_input_per_token": 0.0000025,
                    "uncached_input_per_token": 0.000005,
                    "completion_per_token": 0.000015,
                },
            },
            "stt": {
                "deepgram": {
                    "per_audio_second": "0.000072",
                }
            },
            "tts": {
                "cartesia": {
                    "per_character": "0.000038",
                    "billing_basis": "characters",
                },
                "elevenlabs": {
                    "per_audio_second": "0.0005",
                    "billing_basis": "audio_seconds",
                },
            },
        }

    def test_valid_rate_card_from_dict(self):
        card = RateCard.from_dict(self.sample_valid_dict)
        self.assertTrue(card.has_rates())

        llm = card.get_llm_rate("gemini-2.5-flash")
        self.assertIsNotNone(llm)
        self.assertEqual(llm.cached_input_per_token, Decimal("0.0000001"))
        self.assertEqual(llm.uncached_input_per_token, Decimal("0.0000003"))
        self.assertEqual(llm.completion_per_token, Decimal("0.0000010"))

        stt = card.get_stt_rate("deepgram")
        self.assertIsNotNone(stt)
        self.assertEqual(stt.per_audio_second, Decimal("0.000072"))

        tts_char = card.get_tts_rate("cartesia")
        self.assertIsNotNone(tts_char)
        self.assertEqual(tts_char.billing_basis, "characters")
        self.assertEqual(tts_char.per_character, Decimal("0.000038"))

        tts_sec = card.get_tts_rate("elevenlabs")
        self.assertIsNotNone(tts_sec)
        self.assertEqual(tts_sec.billing_basis, "audio_seconds")
        self.assertEqual(tts_sec.per_audio_second, Decimal("0.0005"))

    def test_valid_rate_card_from_json_string(self):
        json_str = json.dumps(self.sample_valid_dict)
        card = RateCard.from_json(json_str)
        self.assertTrue(card.has_rates())
        self.assertIn("gpt-4o", card.llm)

    def test_valid_rate_card_from_json_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(self.sample_valid_dict, f)
            temp_path = f.name
        try:
            card = RateCard.from_json(Path(temp_path))
            self.assertTrue(card.has_rates())
            self.assertEqual(card.get_llm_rate("gemini-2.5-flash").completion_per_token, Decimal("0.0000010"))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_missing_model_returns_none_never_silent_zero(self):
        card = RateCard.from_dict(self.sample_valid_dict)
        self.assertIsNone(card.get_llm_rate("claude-3-5-sonnet"))
        self.assertIsNone(card.get_stt_rate("whisper"))
        self.assertIsNone(card.get_tts_rate("piper"))
        self.assertIsNone(card.get_llm_rate(None))
        self.assertIsNone(card.get_llm_rate(""))

    def test_empty_rate_card(self):
        card = RateCard()
        self.assertFalse(card.has_rates())
        self.assertIsNone(card.get_llm_rate("any"))
        self.assertIsNone(card.get_stt_rate("any"))
        self.assertIsNone(card.get_tts_rate("any"))

    def test_from_environment_empty(self):
        for key in (
            "PIPECAT_RATE_CARD_PATH",
            "FONAZO_RATE_CARD_PATH",
            "RATE_CARD_PATH",
            "PIPECAT_RATE_CARD_JSON",
            "FONAZO_RATE_CARD_JSON",
            "RATE_CARD_JSON",
        ):
            os.environ.pop(key, None)

        card = RateCard.from_environment()
        self.assertFalse(card.has_rates())

    def test_from_environment_json(self):
        os.environ["PIPECAT_RATE_CARD_JSON"] = json.dumps(self.sample_valid_dict)
        try:
            card = RateCard.from_environment()
            self.assertTrue(card.has_rates())
            self.assertIn("gemini-2.5-flash", card.llm)
        finally:
            os.environ.pop("PIPECAT_RATE_CARD_JSON", None)

    def test_from_environment_path(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(self.sample_valid_dict, f)
            temp_path = f.name
        os.environ["PIPECAT_RATE_CARD_PATH"] = temp_path
        try:
            card = RateCard.from_environment()
            self.assertTrue(card.has_rates())
            self.assertIn("deepgram", card.stt)
        finally:
            os.environ.pop("PIPECAT_RATE_CARD_PATH", None)
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_from_environment_missing_file_raises_error(self):
        os.environ["PIPECAT_RATE_CARD_PATH"] = "non_existent_rate_card_file_12345.json"
        try:
            with self.assertRaises(FileNotFoundError):
                RateCard.from_environment()
        finally:
            os.environ.pop("PIPECAT_RATE_CARD_PATH", None)

    def test_rejects_corrupted_json_string(self):
        with self.assertRaises(ValueError):
            RateCard.from_json("{invalid: json,")

    def test_rejects_non_dict_payload(self):
        with self.assertRaises(ValueError):
            RateCard.from_dict("not-a-dict")
        with self.assertRaises(ValueError):
            RateCard.from_dict([1, 2, 3])

    def test_rejects_non_dict_sections(self):
        with self.assertRaises(ValueError):
            RateCard.from_dict({"llm": ["gemini"]})
        with self.assertRaises(ValueError):
            RateCard.from_dict({"stt": 123})
        with self.assertRaises(ValueError):
            RateCard.from_dict({"tts": "string"})

    def test_rejects_missing_llm_fields(self):
        bad_llm = {
            "llm": {
                "gemini": {
                    "cached_input_per_token": "0.0001",
                    # missing uncached_input_per_token and completion_per_token
                }
            }
        }
        with self.assertRaises(ValueError):
            RateCard.from_dict(bad_llm)

    def test_rejects_missing_stt_field(self):
        bad_stt = {"stt": {"deepgram": {}}}
        with self.assertRaises(ValueError):
            RateCard.from_dict(bad_stt)

    def test_rejects_invalid_tts_billing_basis(self):
        bad_tts = {
            "tts": {
                "cartesia": {
                    "per_character": "0.000038",
                    "billing_basis": "words",  # invalid
                }
            }
        }
        with self.assertRaises(ValueError):
            RateCard.from_dict(bad_tts)

    def test_rejects_missing_tts_basis_rate(self):
        bad_tts = {
            "tts": {
                "cartesia": {
                    "billing_basis": "characters",
                    # missing per_character
                }
            }
        }
        with self.assertRaises(ValueError):
            RateCard.from_dict(bad_tts)

    def test_rejects_negative_rates(self):
        neg_dict = {
            "llm": {
                "gemini": {
                    "cached_input_per_token": "-0.0001",
                    "uncached_input_per_token": "0.0001",
                    "completion_per_token": "0.0001",
                }
            }
        }
        with self.assertRaises(ValueError):
            RateCard.from_dict(neg_dict)

    def test_rejects_booleans(self):
        bool_dict = {
            "stt": {
                "deepgram": {
                    "per_audio_second": True,
                }
            }
        }
        with self.assertRaises(ValueError):
            RateCard.from_dict(bool_dict)

    def test_rejects_nan_and_inf(self):
        nan_dict = {
            "tts": {
                "cartesia": {
                    "per_character": "nan",
                    "billing_basis": "characters",
                }
            }
        }
        with self.assertRaises(ValueError):
            RateCard.from_dict(nan_dict)

        inf_dict = {
            "tts": {
                "cartesia": {
                    "per_character": "inf",
                    "billing_basis": "characters",
                }
            }
        }
        with self.assertRaises(ValueError):
            RateCard.from_dict(inf_dict)


class CostCalculatorTurnTests(unittest.TestCase):
    def setUp(self):
        self.rate_card = RateCard.from_dict({
            "llm": {
                "gemini-2.5-flash": {
                    "cached_input_per_token": "0.000001",
                    "uncached_input_per_token": "0.000005",
                    "completion_per_token": "0.000015",
                },
            },
            "stt": {
                "deepgram": {
                    "per_audio_second": "0.0001",
                },
            },
            "tts": {
                "cartesia": {
                    "per_character": "0.00003",
                    "billing_basis": "characters",
                },
                "elevenlabs": {
                    "per_audio_second": "0.001",
                    "billing_basis": "audio_seconds",
                },
            },
        })
        self.calc = CostCalculator(self.rate_card)

    def test_llm_cached_and_uncached_token_cost(self):
        # 1000 prompt tokens total, 400 cached => 600 uncached, 200 completion tokens
        # Cost: (400 * 0.000001) + (600 * 0.000005) + (200 * 0.000015)
        #       = 0.0004 + 0.003 + 0.003 = 0.0064
        turn = {
            "llm_model": "gemini-2.5-flash",
            "prompt_tokens": 1000,
            "cached_prompt_tokens": 400,
            "completion_tokens": 200,
        }
        res = self.calc.calculate_turn(turn)
        self.assertEqual(res["status"], "measured")
        self.assertEqual(res["lines"]["llm"]["status"], "measured")
        self.assertEqual(res["lines"]["llm"]["cost_usd"], 0.0064)
        self.assertEqual(res["lines"]["stt"]["status"], "not_applicable")
        self.assertEqual(res["lines"]["tts"]["status"], "not_applicable")
        self.assertEqual(res["total_cost_usd"], 0.0064)

    def test_stt_audio_seconds_cost(self):
        # 12.5 seconds * 0.0001 = 0.00125
        turn = {
            "stt_model": "deepgram",
            "stt_audio_seconds": 12.5,
        }
        res = self.calc.calculate_turn(turn)
        self.assertEqual(res["status"], "measured")
        self.assertEqual(res["lines"]["stt"]["status"], "measured")
        self.assertEqual(res["lines"]["stt"]["cost_usd"], 0.00125)
        self.assertEqual(res["total_cost_usd"], 0.00125)

    def test_tts_characters_basis_cost(self):
        # 300 characters * 0.00003 = 0.009
        turn = {
            "tts_model": "cartesia",
            "tts_characters": 300,
        }
        res = self.calc.calculate_turn(turn)
        self.assertEqual(res["status"], "measured")
        self.assertEqual(res["lines"]["tts"]["status"], "measured")
        self.assertEqual(res["lines"]["tts"]["billing_basis"], "characters")
        self.assertEqual(res["lines"]["tts"]["cost_usd"], 0.009)
        self.assertEqual(res["total_cost_usd"], 0.009)

    def test_tts_audio_seconds_basis_cost(self):
        # 4.5 seconds * 0.001 = 0.0045
        turn = {
            "tts_model": "elevenlabs",
            "tts_audio_seconds": 4.5,
        }
        res = self.calc.calculate_turn(turn)
        self.assertEqual(res["status"], "measured")
        self.assertEqual(res["lines"]["tts"]["status"], "measured")
        self.assertEqual(res["lines"]["tts"]["billing_basis"], "audio_seconds")
        self.assertEqual(res["lines"]["tts"]["cost_usd"], 0.0045)
        self.assertEqual(res["total_cost_usd"], 0.0045)

    def test_turn_all_measured(self):
        turn = {
            "llm_model": "gemini-2.5-flash",
            "prompt_tokens": 100,
            "cached_prompt_tokens": 0,
            "completion_tokens": 50,
            "stt_model": "deepgram",
            "stt_audio_seconds": 2.0,
            "tts_model": "cartesia",
            "tts_characters": 100,
        }
        # LLM: 100 * 0.000005 + 50 * 0.000015 = 0.0005 + 0.00075 = 0.00125
        # STT: 2.0 * 0.0001 = 0.0002
        # TTS: 100 * 0.00003 = 0.003
        # Total = 0.00445
        res = self.calc.calculate_turn(turn)
        self.assertEqual(res["status"], "measured")
        self.assertEqual(res["lines"]["llm"]["cost_usd"], 0.00125)
        self.assertEqual(res["lines"]["stt"]["cost_usd"], 0.0002)
        self.assertEqual(res["lines"]["tts"]["cost_usd"], 0.003)
        self.assertEqual(res["total_cost_usd"], 0.00445)

    def test_turn_missing_rate(self):
        turn = {
            "llm_model": "unconfigured-model",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "stt_model": "deepgram",
            "stt_audio_seconds": 1.0,
        }
        res = self.calc.calculate_turn(turn)
        self.assertEqual(res["status"], "missing_rate")
        self.assertEqual(res["lines"]["llm"]["status"], "missing_rate")
        self.assertIsNone(res["lines"]["llm"]["cost_usd"])
        self.assertEqual(res["lines"]["stt"]["status"], "measured")
        # STT is measured (0.0001), total_cost_usd reflects measured portions
        self.assertEqual(res["total_cost_usd"], 0.0001)

    def test_turn_not_applicable_when_empty(self):
        turn = {}
        res = self.calc.calculate_turn(turn)
        self.assertEqual(res["status"], "not_applicable")
        self.assertIsNone(res["total_cost_usd"])
        for mod in ("llm", "stt", "tts"):
            self.assertEqual(res["lines"][mod]["status"], "not_applicable")
            self.assertIsNone(res["lines"][mod]["cost_usd"])


class AccumulatorCostIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.rate_card = RateCard.from_dict({
            "llm": {
                "gemini-2.5-flash": {
                    "cached_input_per_token": "0.000001",
                    "uncached_input_per_token": "0.000005",
                    "completion_per_token": "0.000015",
                },
            },
            "stt": {
                "deepgram": {
                    "per_audio_second": "0.0001",
                },
            },
            "tts": {
                "cartesia": {
                    "per_character": "0.00003",
                    "billing_basis": "characters",
                },
            },
        })

    def test_accumulator_records_turn_costs_and_session_summary(self):
        acc = SessionMetricsAccumulator(
            session_id="acc-cost-test",
            llm_model="gemini-2.5-flash",
            stt_model="deepgram",
            tts_model="cartesia",
            rate_card=self.rate_card,
        )

        # Turn 1
        acc.note_turn_started(1)
        acc.collect({
            "kind": "llm",
            "processor": "GoogleLLMService",
            "model": "gemini-2.5-flash",
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
            "cache_read_input_tokens": 50,
        })
        acc.collect({
            "kind": "stt",
            "processor": "DeepgramSTTService",
            "model": "deepgram",
            "audio_seconds": 3.0,
        })
        acc.collect({
            "kind": "tts",
            "processor": "CartesiaTTSService",
            "model": "cartesia",
            "characters": 150,
        })
        acc.note_turn_ended({
            "turn_count": 1,
            "duration_secs": 4.0,
            "was_interrupted": False,
            "status": "completed",
        })

        summary = acc.summary()
        self.assertEqual(len(summary.turns.records), 1)
        t1_cost = summary.turns.records[0].cost_breakdown
        self.assertIsNotNone(t1_cost)
        self.assertEqual(t1_cost["status"], "measured")

        # Turn 1:
        # LLM: 50 * 0.000001 + 150 * 0.000005 + 100 * 0.000015 = 0.00005 + 0.00075 + 0.0015 = 0.0023
        # STT: 3.0 * 0.0001 = 0.0003
        # TTS: 150 * 0.00003 = 0.0045
        # Total = 0.0071
        self.assertEqual(t1_cost["lines"]["llm"]["cost_usd"], 0.0023)
        self.assertEqual(t1_cost["lines"]["stt"]["cost_usd"], 0.0003)
        self.assertEqual(t1_cost["lines"]["tts"]["cost_usd"], 0.0045)
        self.assertEqual(t1_cost["total_cost_usd"], 0.0071)

        # Session summary cost breakdown
        self.assertEqual(summary.cost_breakdown.currency, "USD")
        self.assertEqual(summary.cost_breakdown.total_cost_usd, 0.0071)
        self.assertEqual(summary.cost_breakdown.lines["llm"].status, "measured")
        self.assertEqual(summary.cost_breakdown.lines["llm"].cost_usd, 0.0023)
        self.assertEqual(summary.cost_breakdown.lines["stt"].status, "measured")
        self.assertEqual(summary.cost_breakdown.lines["stt"].cost_usd, 0.0003)
        self.assertEqual(summary.cost_breakdown.lines["tts"].status, "measured")
        self.assertEqual(summary.cost_breakdown.lines["tts"].cost_usd, 0.0045)
        self.assertEqual(summary.cost_breakdown.turns_with_missing_rates, 0)

        # JSON dictionary serializability check
        as_dict = acc.summary_dict()
        self.assertEqual(as_dict["cost_breakdown"]["total_cost_usd"], 0.0071)
        self.assertEqual(as_dict["cost_breakdown"]["lines"]["llm"]["status"], "measured")

    def test_accumulator_tracks_missing_rate_turns(self):
        acc = SessionMetricsAccumulator(
            session_id="acc-missing-rate-test",
            llm_model="unknown-llm",
            stt_model="deepgram",
            tts_model="cartesia",
            rate_card=self.rate_card,
        )
        acc.note_turn_started(1)
        acc.collect({
            "kind": "llm",
            "processor": "CustomLLM",
            "model": "unknown-llm",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })
        acc.note_turn_ended({
            "turn_count": 1,
            "duration_secs": 2.0,
            "was_interrupted": False,
            "status": "completed",
        })

        summary = acc.summary()
        self.assertEqual(summary.cost_breakdown.turns_with_missing_rates, 1)
        self.assertEqual(summary.cost_breakdown.lines["llm"].status, "missing_rate")
        self.assertIsNone(summary.cost_breakdown.lines["llm"].cost_usd)


class CreditSimulationTests(unittest.TestCase):
    def setUp(self):
        for key in (
            "PIPECAT_PLAN_NAME",
            "FONAZO_PLAN_NAME",
            "PIPECAT_CUSTOMER_RATE_INR_PER_SECOND",
            "FONAZO_CUSTOMER_RATE_INR_PER_SECOND",
            "PIPECAT_CREDIT_BALANCE",
            "FONAZO_CREDIT_BALANCE",
        ):
            os.environ.pop(key, None)

    def tearDown(self):
        self.setUp()

    def test_credit_account_defaults(self):
        account = CreditAccount()
        self.assertEqual(account.plan_name, "standard")
        self.assertEqual(account.customer_rate_inr_per_second, Decimal("0.10"))
        self.assertIsNone(account.credit_balance)
        self.assertEqual(account.credits_used, Decimal("0"))
        self.assertEqual(account.customer_revenue, Decimal("0"))

        snap = account.snapshot()
        self.assertEqual(snap["plan_name"], "standard")
        self.assertEqual(snap["customer_rate_inr_per_second"], 0.1)
        self.assertEqual(snap["credit_unit"], "1 connected second")
        self.assertEqual(snap["connected_seconds_source"], "completed_turn_duration_simulation")
        self.assertEqual(snap["credits_used"], 0.0)
        self.assertIsNone(snap["credits_remaining"])
        self.assertIsNone(snap["projected_seconds_left"])
        self.assertEqual(snap["customer_revenue_inr"], 0.0)

    def test_credit_account_from_environment(self):
        os.environ["PIPECAT_PLAN_NAME"] = "enterprise"
        os.environ["PIPECAT_CUSTOMER_RATE_INR_PER_SECOND"] = "0.25"
        os.environ["PIPECAT_CREDIT_BALANCE"] = "300"

        account = CreditAccount.from_environment()
        self.assertEqual(account.plan_name, "enterprise")
        self.assertEqual(account.customer_rate_inr_per_second, Decimal("0.25"))
        self.assertEqual(account.credit_balance, Decimal("300"))

        snap = account.snapshot()
        self.assertEqual(snap["credits_remaining"], 300.0)
        self.assertEqual(snap["projected_seconds_left"], 300.0)

    def test_record_connected_seconds_progression(self):
        account = CreditAccount(
            plan_name="pro",
            customer_rate_inr_per_second=Decimal("0.20"),
            credit_balance=Decimal("100"),
        )
        snap1 = account.record_connected_seconds(30.0, source="completed_turn_duration_simulation")
        self.assertEqual(snap1["credits_used"], 30.0)
        self.assertEqual(snap1["credits_remaining"], 70.0)
        self.assertEqual(snap1["projected_seconds_left"], 70.0)
        self.assertEqual(snap1["customer_revenue_inr"], 6.0)

        snap2 = account.record_connected_seconds(50.0, source="completed_turn_duration_simulation")
        self.assertEqual(snap2["credits_used"], 80.0)
        self.assertEqual(snap2["credits_remaining"], 20.0)
        self.assertEqual(snap2["projected_seconds_left"], 20.0)
        self.assertEqual(snap2["customer_revenue_inr"], 16.0)

    def test_explicit_connected_seconds_source_label(self):
        account = CreditAccount(
            plan_name="standard",
            customer_rate_inr_per_second=Decimal("0.10"),
            credit_balance=Decimal("60"),
        )
        snap = account.record_connected_seconds(10.0, source="completed_turn_duration_simulation")
        self.assertEqual(snap["connected_seconds_source"], "completed_turn_duration_simulation")

        session_snap = account.snapshot(source="transport_client_connected_seconds")
        self.assertEqual(session_snap["connected_seconds_source"], "transport_client_connected_seconds")

    def test_accumulator_tracks_turn_and_session_credit_simulation(self):
        account = CreditAccount(
            plan_name="starter",
            customer_rate_inr_per_second=Decimal("0.10"),
            credit_balance=Decimal("120"),
        )
        acc = SessionMetricsAccumulator(
            session_id="credit-test",
            credit_account=account,
        )

        acc.note_turn_started(1)
        acc.note_turn_ended({
            "turn_count": 1,
            "duration_secs": 15.0,
            "was_interrupted": False,
            "status": "completed",
        })

        acc.note_turn_started(2)
        acc.note_turn_ended({
            "turn_count": 2,
            "duration_secs": 25.0,
            "was_interrupted": False,
            "status": "completed",
        })

        summary = acc.summary()
        self.assertEqual(len(summary.turns.records), 2)

        # Turn 1
        t1_credit = summary.turns.records[0].credit_simulation
        self.assertIsNotNone(t1_credit)
        self.assertEqual(t1_credit["credits_used"], 15.0)
        self.assertEqual(t1_credit["connected_seconds_source"], "completed_turn_duration_simulation")

        # Turn 2
        t2_credit = summary.turns.records[1].credit_simulation
        self.assertIsNotNone(t2_credit)
        self.assertEqual(t2_credit["credits_used"], 40.0)
        self.assertEqual(t2_credit["credits_remaining"], 80.0)
        self.assertEqual(t2_credit["customer_revenue_inr"], 4.0)

        # Session summary credit simulation
        self.assertEqual(summary.credit_simulation.plan_name, "starter")
        self.assertEqual(summary.credit_simulation.credits_used, 40.0)
        self.assertEqual(summary.credit_simulation.credits_remaining, 80.0)
        self.assertEqual(summary.credit_simulation.customer_revenue_inr, 4.0)
        self.assertEqual(summary.credit_simulation.connected_seconds_source, "completed_turn_duration_simulation")

        # When transport client connected is present
        acc.note_transport({"client_connected_secs": 42.5})
        summary_with_transport = acc.summary()
        self.assertEqual(summary_with_transport.credit_simulation.connected_seconds_source, "transport_client_connected_seconds")

        # JSON dictionary serializability check
        as_dict = acc.summary_dict()
        self.assertIn("credit_simulation", as_dict)
        self.assertEqual(as_dict["credit_simulation"]["credits_used"], 40.0)


if __name__ == "__main__":
    unittest.main()


