import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from decimal import Decimal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.costs import LLMRate, RateCard, STTRate, TTSRate


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


if __name__ == "__main__":
    unittest.main()
