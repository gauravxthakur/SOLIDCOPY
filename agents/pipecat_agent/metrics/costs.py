"""Configurable vendor rate cards and validation for Pipecat observability.

Prices are expressed in USD per unit. No vendor prices are hardcoded;
load them from a JSON file, environment variable, or pass a RateCard explicitly.
Missing rates are explicitly tracked as missing, never defaulted to silent zeros.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
from typing import Any, Mapping


def _decimal_rate(value: Any, field_name: str = "rate") -> Decimal:
    """Convert a value to a non-negative Decimal rate, rejecting booleans, NaN, Inf, and negatives."""
    if isinstance(value, bool):
        raise ValueError(f"Boolean value not allowed for {field_name}: {value!r}")
    if value is None:
        raise ValueError(f"Missing required numeric value for {field_name}")
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric rate for {field_name}: {value!r}") from exc

    if dec.is_nan() or dec.is_infinite():
        raise ValueError(f"NaN and Inf are not allowed for {field_name}: {value!r}")
    if dec < Decimal("0"):
        raise ValueError(f"Negative rate is not allowed for {field_name}: {value!r}")

    return dec


@dataclass(frozen=True)
class LLMRate:
    """Per-token pricing for an LLM model."""

    cached_input_per_token: Decimal
    uncached_input_per_token: Decimal
    completion_per_token: Decimal

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], model_name: str = "llm") -> LLMRate:
        if not isinstance(data, Mapping):
            raise ValueError(f"LLM rate entry for {model_name!r} must be a mapping, got {type(data).__name__}")
        for req_field in ("cached_input_per_token", "uncached_input_per_token", "completion_per_token"):
            if req_field not in data:
                raise ValueError(f"LLM rate entry for {model_name!r} missing required field: {req_field!r}")

        return cls(
            cached_input_per_token=_decimal_rate(data["cached_input_per_token"], f"{model_name}.cached_input_per_token"),
            uncached_input_per_token=_decimal_rate(data["uncached_input_per_token"], f"{model_name}.uncached_input_per_token"),
            completion_per_token=_decimal_rate(data["completion_per_token"], f"{model_name}.completion_per_token"),
        )


@dataclass(frozen=True)
class STTRate:
    """Per-audio-second pricing for an STT model."""

    per_audio_second: Decimal

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], model_name: str = "stt") -> STTRate:
        if not isinstance(data, Mapping):
            raise ValueError(f"STT rate entry for {model_name!r} must be a mapping, got {type(data).__name__}")
        if "per_audio_second" not in data:
            raise ValueError(f"STT rate entry for {model_name!r} missing required field 'per_audio_second'")

        return cls(
            per_audio_second=_decimal_rate(data["per_audio_second"], f"{model_name}.per_audio_second"),
        )


@dataclass(frozen=True)
class TTSRate:
    """Pricing for a TTS model, either per character or per audio second."""

    per_character: Decimal | None = None
    per_audio_second: Decimal | None = None
    billing_basis: str = "characters"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], model_name: str = "tts") -> TTSRate:
        if not isinstance(data, Mapping):
            raise ValueError(f"TTS rate entry for {model_name!r} must be a mapping, got {type(data).__name__}")

        basis = str(data.get("billing_basis", "characters")).strip().lower()
        if basis not in ("characters", "audio_seconds"):
            raise ValueError(
                f"TTS rate entry for {model_name!r} has invalid billing_basis {basis!r}; "
                "must be 'characters' or 'audio_seconds'"
            )

        per_char_val = data.get("per_character")
        per_sec_val = data.get("per_audio_second")

        if basis == "characters":
            if per_char_val is None:
                raise ValueError(
                    f"TTS rate entry for {model_name!r} with billing_basis='characters' "
                    "requires 'per_character'"
                )
            per_char = _decimal_rate(per_char_val, f"{model_name}.per_character")
            per_sec = _decimal_rate(per_sec_val, f"{model_name}.per_audio_second") if per_sec_val is not None else None
        else:  # audio_seconds
            if per_sec_val is None:
                raise ValueError(
                    f"TTS rate entry for {model_name!r} with billing_basis='audio_seconds' "
                    "requires 'per_audio_second'"
                )
            per_sec = _decimal_rate(per_sec_val, f"{model_name}.per_audio_second")
            per_char = _decimal_rate(per_char_val, f"{model_name}.per_character") if per_char_val is not None else None

        return cls(
            per_character=per_char,
            per_audio_second=per_sec,
            billing_basis=basis,
        )


@dataclass
class RateCard:
    """Configurable rate card keyed by model/provider labels emitted by Pipecat."""

    llm: dict[str, LLMRate] = field(default_factory=dict)
    stt: dict[str, STTRate] = field(default_factory=dict)
    tts: dict[str, TTSRate] = field(default_factory=dict)

    def has_rates(self) -> bool:
        """Return True if at least one model rate is configured."""
        return bool(self.llm or self.stt or self.tts)

    def get_llm_rate(self, model: str | None) -> LLMRate | None:
        """Lookup LLM rate by model name. Returns None if model is unset or missing."""
        if not model:
            return None
        return self.llm.get(model)

    def get_stt_rate(self, model: str | None) -> STTRate | None:
        """Lookup STT rate by model name. Returns None if model is unset or missing."""
        if not model:
            return None
        return self.stt.get(model)

    def get_tts_rate(self, model: str | None) -> TTSRate | None:
        """Lookup TTS rate by model name. Returns None if model is unset or missing."""
        if not model:
            return None
        return self.tts.get(model)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RateCard:
        """Construct and strictly validate a RateCard from a mapping."""
        if not isinstance(payload, Mapping):
            raise ValueError(f"RateCard payload must be a mapping, got {type(payload).__name__}")

        llm_dict = payload.get("llm") or {}
        stt_dict = payload.get("stt") or {}
        tts_dict = payload.get("tts") or {}

        if not isinstance(llm_dict, Mapping):
            raise ValueError(f"RateCard 'llm' section must be a mapping, got {type(llm_dict).__name__}")
        if not isinstance(stt_dict, Mapping):
            raise ValueError(f"RateCard 'stt' section must be a mapping, got {type(stt_dict).__name__}")
        if not isinstance(tts_dict, Mapping):
            raise ValueError(f"RateCard 'tts' section must be a mapping, got {type(tts_dict).__name__}")

        llm_rates = {name: LLMRate.from_dict(val, model_name=name) for name, val in llm_dict.items()}
        stt_rates = {name: STTRate.from_dict(val, model_name=name) for name, val in stt_dict.items()}
        tts_rates = {name: TTSRate.from_dict(val, model_name=name) for name, val in tts_dict.items()}

        return cls(llm=llm_rates, stt=stt_rates, tts=tts_rates)

    @classmethod
    def from_json(cls, value: str | Path) -> RateCard:
        """Construct RateCard from a JSON file path or raw JSON string.

        Fails clearly if the file does not exist, JSON is malformed, or config is invalid.
        """
        raw_str = ""
        path = Path(value) if isinstance(value, (str, Path)) else None
        if path is not None and path.exists() and path.is_file():
            try:
                raw_str = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise OSError(f"Failed to read rate card file at {path}: {exc}") from exc
        else:
            raw_str = str(value)

        try:
            payload = json.loads(raw_str)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Invalid JSON for rate card: {exc}") from exc

        return cls.from_dict(payload)

    @classmethod
    def from_environment(cls) -> RateCard:
        """Load RateCard from environment variables.

        Checked in priority order:
        1. PIPECAT_RATE_CARD_PATH / FONAZO_RATE_CARD_PATH / RATE_CARD_PATH (file path)
        2. PIPECAT_RATE_CARD_JSON / FONAZO_RATE_CARD_JSON / RATE_CARD_JSON (inline JSON string)

        If configured path does not exist or content is invalid, fails clearly.
        If no environment variable is set, returns an empty RateCard().
        """
        env_path = (
            os.getenv("PIPECAT_RATE_CARD_PATH")
            or os.getenv("FONAZO_RATE_CARD_PATH")
            or os.getenv("RATE_CARD_PATH")
        )
        if env_path:
            p = Path(env_path)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"Configured rate card file does not exist: {env_path}")
            return cls.from_json(p)

        env_json = (
            os.getenv("PIPECAT_RATE_CARD_JSON")
            or os.getenv("FONAZO_RATE_CARD_JSON")
            or os.getenv("RATE_CARD_JSON")
        )
        if env_json:
            return cls.from_json(env_json)

        return cls()


def _number(value: Decimal) -> float:
    """Format decimal currency to float with 6 decimal places."""
    return float(value.quantize(Decimal("0.000001")))


def _decimal_non_negative(value: Any, name: str = "value") -> Decimal:
    """Convert usage quantities to non-negative Decimal, defaulting None to 0."""
    if isinstance(value, bool):
        raise ValueError(f"Boolean not allowed for {name}: {value!r}")
    if value is None:
        return Decimal("0")
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {name}: {value!r}") from exc
    if dec.is_nan() or dec.is_infinite():
        raise ValueError(f"NaN and Inf not allowed for {name}: {value!r}")
    if dec < Decimal("0"):
        return Decimal("0")
    return dec


class CostCalculator:
    """Convert one finalized turn's usage into an auditable cost breakdown."""

    def __init__(self, rate_card: RateCard | None = None) -> None:
        self.rate_card = rate_card or RateCard()

    def calculate_turn(self, turn: Mapping[str, Any] | Any) -> dict[str, Any]:
        """Calculate line-item costs and overall status for a single turn."""
        if hasattr(turn, "__dict__") and not isinstance(turn, Mapping):
            turn_dict = turn.__dict__
        else:
            turn_dict = dict(turn)

        lines = {
            "llm": self._llm_line(turn_dict),
            "stt": self._stt_line(turn_dict),
            "tts": self._tts_line(turn_dict),
        }
        measured = [line for line in lines.values() if line.get("status") == "measured"]
        missing = [line for line in lines.values() if line.get("status") == "missing_rate"]

        if measured:
            total = sum(
                (_decimal_rate(line["cost_usd"], "cost_usd") for line in measured if line.get("cost_usd") is not None),
                Decimal("0"),
            )
            total_cost_usd = _number(total)
        else:
            total_cost_usd = None

        if missing:
            status = "missing_rate"
        elif measured:
            status = "measured"
        else:
            status = "not_applicable"

        return {
            "total_cost_usd": total_cost_usd,
            "status": status,
            "lines": lines,
        }

    def _llm_line(self, turn: Mapping[str, Any]) -> dict[str, Any]:
        model = turn.get("llm_model")
        prompt_tokens = turn.get("prompt_tokens")
        cached_prompt_tokens = turn.get("cached_prompt_tokens")
        completion_tokens = turn.get("completion_tokens")

        # If LLM was not used in this turn
        if not model and prompt_tokens is None and completion_tokens is None:
            return {"status": "not_applicable", "cost_usd": None}

        rate = self.rate_card.get_llm_rate(model)
        if rate is None:
            return {"status": "missing_rate", "cost_usd": None, "model": model}

        prompt = _decimal_non_negative(prompt_tokens, "prompt_tokens")
        cached = _decimal_non_negative(cached_prompt_tokens, "cached_prompt_tokens")
        completion = _decimal_non_negative(completion_tokens, "completion_tokens")
        uncached = max(prompt - cached, Decimal("0"))

        cost = (
            cached * rate.cached_input_per_token
            + uncached * rate.uncached_input_per_token
            + completion * rate.completion_per_token
        )
        return {"status": "measured", "cost_usd": _number(cost), "model": model}

    def _stt_line(self, turn: Mapping[str, Any]) -> dict[str, Any]:
        model = turn.get("stt_model")
        audio_seconds = turn.get("stt_audio_seconds")

        if not model and audio_seconds is None:
            return {"status": "not_applicable", "cost_usd": None}

        rate = self.rate_card.get_stt_rate(model)
        if rate is None:
            return {"status": "missing_rate", "cost_usd": None, "model": model}

        if audio_seconds is None:
            return {"status": "not_applicable", "cost_usd": None}

        seconds = _decimal_non_negative(audio_seconds, "stt_audio_seconds")
        cost = seconds * rate.per_audio_second
        return {"status": "measured", "cost_usd": _number(cost), "model": model}

    def _tts_line(self, turn: Mapping[str, Any]) -> dict[str, Any]:
        model = turn.get("tts_model")
        characters = turn.get("tts_characters")
        audio_seconds = turn.get("tts_audio_seconds")

        if not model and characters is None and audio_seconds is None:
            return {"status": "not_applicable", "cost_usd": None}

        rate = self.rate_card.get_tts_rate(model)
        if rate is None:
            return {"status": "missing_rate", "cost_usd": None, "model": model}

        if rate.billing_basis == "characters":
            if characters is None or rate.per_character is None:
                return {
                    "status": "missing_rate" if characters is not None else "not_applicable",
                    "cost_usd": None,
                    "model": model,
                    "billing_basis": rate.billing_basis,
                }
            chars = _decimal_non_negative(characters, "tts_characters")
            cost = chars * rate.per_character
        elif rate.billing_basis == "audio_seconds":
            if audio_seconds is None or rate.per_audio_second is None:
                return {
                    "status": "missing_rate" if audio_seconds is not None else "not_applicable",
                    "cost_usd": None,
                    "model": model,
                    "billing_basis": rate.billing_basis,
                }
            secs = _decimal_non_negative(audio_seconds, "tts_audio_seconds")
            cost = secs * rate.per_audio_second
        else:
            return {"status": "missing_rate", "cost_usd": None, "model": model}

        return {
            "status": "measured",
            "cost_usd": _number(cost),
            "model": model,
            "billing_basis": rate.billing_basis,
        }

