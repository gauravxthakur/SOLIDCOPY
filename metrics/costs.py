"""Configurable vendor rate cards and usage cost calculation.

Prices are expressed in USD per unit. No vendor prices are hardcoded here;
load them from a JSON file or pass a RateCard explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMRate:
    cached_input_per_token: Decimal
    uncached_input_per_token: Decimal
    completion_per_token: Decimal


@dataclass(frozen=True)
class STTRate:
    per_audio_second: Decimal


@dataclass(frozen=True)
class TTSRate:
    per_character: Decimal | None = None
    per_audio_second: Decimal | None = None
    billing_basis: str = "characters"


@dataclass
class RateCard:
    """Rates keyed by the model/provider labels emitted by LiveKit metrics."""

    llm: dict[str, LLMRate] = field(default_factory=dict)
    stt: dict[str, STTRate] = field(default_factory=dict)
    tts: dict[str, TTSRate] = field(default_factory=dict)

    @classmethod
    def from_json(cls, value: str | Path) -> "RateCard":
        path = Path(value)
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(str(value))
        return cls.from_dict(payload)

    @classmethod
    def from_environment(cls) -> "RateCard":
        path = os.getenv("FONAZO_RATE_CARD_PATH")
        if path:
            return cls.from_json(path)
        raw = os.getenv("FONAZO_RATE_CARD_JSON")
        return cls.from_json(raw) if raw else cls()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RateCard":
        return cls(
            llm={
                name: LLMRate(
                    cached_input_per_token=_decimal(values["cached_input_per_token"]),
                    uncached_input_per_token=_decimal(values["uncached_input_per_token"]),
                    completion_per_token=_decimal(values["completion_per_token"]),
                )
                for name, values in (payload.get("llm") or {}).items()
            },
            stt={
                name: STTRate(per_audio_second=_decimal(values["per_audio_second"]))
                for name, values in (payload.get("stt") or {}).items()
            },
            tts={
                name: TTSRate(
                    per_character=(
                        _decimal(values["per_character"])
                        if values.get("per_character") is not None
                        else None
                    ),
                    per_audio_second=(
                        _decimal(values["per_audio_second"])
                        if values.get("per_audio_second") is not None
                        else None
                    ),
                    billing_basis=values.get("billing_basis", "characters"),
                )
                for name, values in (payload.get("tts") or {}).items()
            },
        )


class CostCalculator:
    """Convert one finalized turn's usage into an auditable cost breakdown."""

    def __init__(self, rate_card: RateCard):
        self.rate_card = rate_card

    def calculate_turn(self, turn: dict[str, Any]) -> dict[str, Any]:
        lines = {
            "llm": self._llm_line(turn),
            "stt": self._stt_line(turn),
            "tts": self._tts_line(turn),
        }
        measured = [line for line in lines.values() if line["status"] == "measured"]
        missing = [line for line in lines.values() if line["status"] == "missing_rate"]
        total = sum(
            (_decimal(line["cost_usd"]) for line in measured),
            Decimal("0"),
        )
        return {
            "total_cost_usd": _number(total) if measured else None,
            "status": "missing_rate" if missing else "measured",
            "lines": lines,
        }

    def _llm_line(self, turn: dict[str, Any]) -> dict[str, Any]:
        model = turn.get("llm_model")
        if not turn.get("llm_model") and turn.get("prompt_tokens") is None:
            return _not_applicable()
        rate = self.rate_card.llm.get(model)
        if rate is None:
            return _missing_rate(model)
        cached = _decimal(turn.get("cached_prompt_tokens", 0))
        prompt = _decimal(turn.get("prompt_tokens", 0))
        completion = _decimal(turn.get("completion_tokens", 0))
        uncached = max(prompt - cached, Decimal("0"))
        cost = (
            cached * rate.cached_input_per_token
            + uncached * rate.uncached_input_per_token
            + completion * rate.completion_per_token
        )
        return _measured(cost, {"model": model})

    def _stt_line(self, turn: dict[str, Any]) -> dict[str, Any]:
        seconds = turn.get("stt_audio_seconds")
        model = turn.get("stt_model")
        if seconds is None:
            return _not_applicable()
        rate = self.rate_card.stt.get(model)
        if rate is None:
            return _missing_rate(model)
        return _measured(_decimal(seconds) * rate.per_audio_second, {"model": model})

    def _tts_line(self, turn: dict[str, Any]) -> dict[str, Any]:
        characters = turn.get("tts_characters")
        seconds = turn.get("tts_audio_seconds")
        model = turn.get("tts_model")
        if characters is None and seconds is None:
            return _not_applicable()
        rate = self.rate_card.tts.get(model)
        if rate is None:
            return _missing_rate(model)
        if rate.billing_basis == "characters":
            if rate.per_character is None or characters is None:
                return _missing_rate(model)
            cost = _decimal(characters) * rate.per_character
        elif rate.billing_basis == "audio_seconds":
            if rate.per_audio_second is None or seconds is None:
                return _missing_rate(model)
            cost = _decimal(seconds) * rate.per_audio_second
        else:
            return _missing_rate(model)
        return _measured(cost, {"model": model, "billing_basis": rate.billing_basis})


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid rate or usage value: {value!r}") from exc


def _number(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001")))


def _measured(cost: Decimal, details: dict[str, Any]) -> dict[str, Any]:
    return {"status": "measured", "cost_usd": _number(cost), **details}


def _not_applicable() -> dict[str, Any]:
    return {"status": "not_applicable", "cost_usd": None}


def _missing_rate(model: str | None) -> dict[str, Any]:
    return {"status": "missing_rate", "cost_usd": None, "model": model}
