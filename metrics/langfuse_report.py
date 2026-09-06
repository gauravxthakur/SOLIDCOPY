"""Historical analytics from the Langfuse API (not JSONL exports).

Counts only exact observation names. Does not fetch input/output (prompts/transcripts).

  uv run python -m metrics.langfuse_report --hours 24
  uv run python -m metrics.langfuse_report --session-id console-room-abc
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import argparse
import json
import os
from pathlib import Path
import statistics
from typing import Any

from dotenv import load_dotenv
from langfuse import Langfuse
from metrics.costs import CostCalculator, RateCard

CANONICAL_NAMES = {
    "llm_request",
    "tts_request",
    "stt_request",
    "user_turn",
    "agent_turn",
    "eou_detection",
    "user_speaking",
    "agent_session",
}

OBSERVATION_FIELDS = "core,basic,usage,metrics,model"


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    rank = (len(ordered) - 1) * pct / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 4)


def _latency_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "average": round(statistics.fmean(values), 4) if values else None,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "maximum": round(max(values), 4) if values else None,
    }


def _obs_get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _first_not_none(obs: Any, *keys: str) -> Any:
    for key in keys:
        value = _obs_get(obs, key)
        if value is not None:
            return value
    return None


def _metric_stats(
    name: str,
    values: dict[str, list[float]],
    counts: Counter[str],
    costs: dict[str, float],
    latencies: dict[str, list[float]],
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "count": counts[name],
        "cost_usd": round(costs[name], 6),
    }
    if name == "agent_session":
        stats["session_duration_seconds"] = _latency_stats(latencies.get(name, []))
    else:
        stats["latency_seconds"] = _latency_stats(latencies.get(name, []))
    if name == "llm_request":
        stats["ttft_seconds"] = _latency_stats(values.get("ttft", []))
    elif name == "tts_request":
        stats["ttfb_seconds"] = _latency_stats(values.get("ttfb", []))
    elif name == "stt_request":
        stats["audio_duration_seconds"] = _latency_stats(values.get("audio", []))
    return stats


def aggregate_observations(observations: list[Any]) -> dict[str, Any]:
    """Build a report from Langfuse observation objects or dicts."""
    by_name: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    by_session: Counter[str] = Counter()
    latencies: dict[str, list[float]] = defaultdict(list)
    metric_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    total_cost = 0.0
    cost_by_name: dict[str, float] = defaultdict(float)
    tool_total = 0
    tool_errors = 0
    tool_by_name: Counter[str] = Counter()

    for obs in observations:
        name = str(_obs_get(obs, "name") or "unknown")
        obs_type = str(_obs_get(obs, "type") or "unknown")
        session_id = _obs_get(obs, "session_id") or _obs_get(obs, "sessionId")
        level = str(_obs_get(obs, "level") or "DEFAULT").upper()
        latency = _float(_obs_get(obs, "latency"))
        ttft = _float(_first_not_none(obs, "time_to_first_token", "timeToFirstToken"))
        ttfb = _float(_first_not_none(obs, "time_to_first_byte", "timeToFirstByte", "ttfb"))
        audio_duration = _float(
            _first_not_none(obs, "audio_duration", "audioDuration", "audio_duration_seconds")
        )
        cost_value = _float(_first_not_none(obs, "total_cost", "totalCost"))
        cost = cost_value if cost_value is not None else 0.0

        by_name[name] += 1
        by_type[obs_type] += 1
        if session_id:
            by_session[str(session_id)] += 1
        if latency is not None:
            latencies[name].append(latency)
        if ttft is not None:
            metric_values[name]["ttft"].append(ttft)
        if ttfb is not None:
            metric_values[name]["ttfb"].append(ttfb)
        if audio_duration is not None:
            metric_values[name]["audio"].append(audio_duration)
        total_cost += cost
        cost_by_name[name] += cost

        if obs_type.upper() == "TOOL":
            tool_total += 1
            tool_by_name[name] += 1
            if level == "ERROR":
                tool_errors += 1

    named = {
        name: _metric_stats(name, metric_values[name], by_name, cost_by_name, latencies)
        for name in CANONICAL_NAMES
        if by_name[name]
    }

    return {
        "observation_count": len(observations),
        "session_count": len(by_session),
        "total_cost_usd": round(total_cost, 6),
        "by_type": dict(by_type),
        "canonical": named,
        "other_names": {
            name: count
            for name, count in by_name.items()
            if name not in CANONICAL_NAMES
        },
        "tools": {
            "count": tool_total,
            "error_count": tool_errors,
            "failure_rate_percentage": (
                round(tool_errors * 100 / tool_total, 2) if tool_total else None
            ),
            "by_name": dict(tool_by_name),
        },
        "latency": {
            "llm_request": _latency_stats(latencies.get("llm_request", [])),
            "tts_request": _latency_stats(latencies.get("tts_request", [])),
            "stt_request": _latency_stats(latencies.get("stt_request", [])),
            "eou_detection": _latency_stats(latencies.get("eou_detection", [])),
            "user_turn": _latency_stats(latencies.get("user_turn", [])),
            "agent_turn": _latency_stats(latencies.get("agent_turn", [])),
            "agent_session": _latency_stats(latencies.get("agent_session", [])),
        },
        "ttft": {
            "llm_request": _latency_stats(metric_values["llm_request"].get("ttft", [])),
        },
        "ttfb": {
            "tts_request": _latency_stats(metric_values["tts_request"].get("ttfb", [])),
        },
        "audio_duration": {
            "stt_request": _latency_stats(metric_values["stt_request"].get("audio", [])),
        },
    }


def _usage_value(obs: Any, *keys: str) -> float | None:
    value = _first_not_none(obs, *keys)
    if value is not None:
        return _float(value)
    usage = _first_not_none(obs, "usage", "usage_details", "usageDetails")
    if usage is not None:
        value = _first_not_none(usage, *keys)
        if value is not None:
            return _float(value)
        if "input_tokens" in keys or "prompt_tokens" in keys:
            value = _first_not_none(usage, "input", "prompt", "inputTokens", "promptTokens")
            if value is not None:
                return _float(value)
        if "output_tokens" in keys or "completion_tokens" in keys:
            value = _first_not_none(
                usage,
                "output",
                "completion",
                "outputTokens",
                "completionTokens",
            )
            if value is not None:
                return _float(value)
        if "cached_input_tokens" in keys or "prompt_cached_tokens" in keys:
            details = _first_not_none(
                usage,
                "input_token_details",
                "inputTokenDetails",
                "prompt_token_details",
                "promptTokenDetails",
            )
            if details is not None:
                value = _first_not_none(details, "cached_tokens", "cachedTokens")
                if value is not None:
                    return _float(value)
    return None


def _sum_usage(observations: list[Any], *keys: str) -> float | None:
    values = [_usage_value(observation, *keys) for observation in observations]
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _local_llm_summary(session_summary: dict[str, Any]) -> dict[str, Any]:
    records = (session_summary.get("turns") or {}).get("records") or []
    llm = session_summary.get("llm") or {}
    llm_records = [record for record in records if record.get("prompt_tokens") is not None]
    cost_line = _local_cost_line(session_summary, "llm")
    return {
        "model": sorted({record.get("llm_model") for record in llm_records if record.get("llm_model")}),
        "request_count": llm.get("request_count", len(llm_records)),
        "input_tokens": llm.get(
            "prompt_tokens", sum(record.get("prompt_tokens") or 0 for record in llm_records)
        ),
        "cached_tokens": llm.get(
            "cached_prompt_tokens",
            sum(record.get("cached_prompt_tokens") or 0 for record in llm_records),
        ),
        "completion_tokens": llm.get(
            "completion_tokens", sum(record.get("completion_tokens") or 0 for record in llm_records)
        ),
        "ttft_seconds": llm.get("ttft_seconds") or _latency_stats([
            float(record["ttft_seconds"])
            for record in llm_records
            if record.get("ttft_seconds") is not None
        ]),
        "cost_usd": cost_line.get("cost_usd"),
        "cost_status": cost_line.get("status"),
    }


def _local_cost_line(session_summary: dict[str, Any], name: str) -> dict[str, Any]:
    line = ((session_summary.get("cost_breakdown") or {}).get("lines") or {}).get(name)
    return line or {"status": "not_available", "cost_usd": None}


def compare_session(
    session_summary: dict[str, Any],
    observations: list[Any],
    rate_card: RateCard,
) -> dict[str, Any]:
    """Compare one local session summary with its exact Langfuse observations."""
    session_id = (session_summary.get("session") or {}).get("session_id")
    langfuse_llm = [
        observation
        for observation in observations
        if str(_obs_get(observation, "name") or "") == "llm_request"
    ]
    langfuse_summary = {
        "model": sorted({
            str(model)
            for observation in langfuse_llm
            for model in [_first_not_none(observation, "model", "model_name", "modelName")]
            if model is not None
        }),
        "request_count": len(langfuse_llm),
        "input_tokens": sum(
            _usage_value(observation, "input_tokens", "prompt_tokens", "inputTokens", "promptTokens") or 0
            for observation in langfuse_llm
        ),
        "cached_tokens": _sum_usage(
            langfuse_llm,
            "cached_input_tokens",
            "prompt_cached_tokens",
            "cachedTokens",
        ),
        "completion_tokens": sum(
            _usage_value(
                observation,
                "output_tokens",
                "completion_tokens",
                "outputTokens",
                "completionTokens",
            ) or 0
            for observation in langfuse_llm
        ),
        "ttft_seconds": _latency_stats([
            value
            for observation in langfuse_llm
            for value in [_float(_first_not_none(observation, "time_to_first_token", "timeToFirstToken"))]
            if value is not None
        ]),
        "cost_usd": round(sum(
            _float(_first_not_none(observation, "total_cost", "totalCost")) or 0
            for observation in langfuse_llm
        ), 6),
    }
    local = _local_llm_summary(session_summary)
    local_costs = {"llm": [], "stt": [], "tts": []}
    local_cost_statuses = {"llm": [], "stt": [], "tts": []}
    calculator = CostCalculator(rate_card)
    for record in (session_summary.get("turns") or {}).get("records") or []:
        breakdown = calculator.calculate_turn(record)
        for name, line in breakdown["lines"].items():
            local_cost_statuses[name].append(line["status"])
            if line["status"] == "measured":
                local_costs[name].append(line["cost_usd"])
    own_costs = {
        name: {
            "status": (
                "missing_rate" if "missing_rate" in local_cost_statuses[name]
                else "measured" if values
                else "not_applicable"
            ),
            "cost_usd": round(sum(values), 6) if values else None,
        }
        for name, values in local_costs.items()
    }
    return {
        "session_id": session_id,
        "langfuse_observation_count": len(observations),
        "llm": {
            "local": local,
            "langfuse": langfuse_summary,
            "differences": {
                key: {
                    "local": local[key],
                    "langfuse": langfuse_summary[key],
                }
                for key in (
                    "model",
                    "request_count",
                    "input_tokens",
                    "cached_tokens",
                    "completion_tokens",
                    "cost_usd",
                    "ttft_seconds",
                )
            },
        },
        "own_rate_card_costs": {
            "stt": own_costs["stt"],
            "tts": own_costs["tts"],
        },
        "notes": [
            "Langfuse is compared only for exact llm_request observations.",
            "STT and TTS costs come from the local usage records and supplied rate card.",
            "Langfuse zero STT/TTS cost is not treated as proof of free usage.",
        ],
    }


def fetch_observations(
    client: Langfuse,
    *,
    session_id: str | None = None,
    from_time: datetime | None = None,
    to_time: datetime | None = None,
    limit: int = 100,
    max_pages: int = 20,
) -> list[Any]:
    filters = []
    if session_id:
        filters.append(
            {"type": "string", "column": "sessionId", "operator": "=", "value": session_id}
        )
    filter_json = json.dumps(filters) if filters else None
    rows: list[Any] = []
    cursor = None
    for _ in range(max_pages):
        page = client.api.observations.get_many(
            fields=OBSERVATION_FIELDS,
            limit=limit,
            cursor=cursor,
            from_start_time=from_time,
            to_start_time=to_time,
            filter=filter_json,
        )
        data = getattr(page, "data", None) or []
        rows.extend(data)
        meta = getattr(page, "meta", None)
        cursor = getattr(meta, "cursor", None) or getattr(meta, "next_cursor", None)
        if not cursor or not data:
            break
    return rows


def build_report(
    *,
    session_id: str | None = None,
    hours: float = 24,
) -> dict[str, Any]:
    load_dotenv()
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    if not public_key or not secret_key or not base_url:
        raise ValueError("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL must be set")

    to_time = datetime.now(timezone.utc)
    from_time = to_time - timedelta(hours=hours)
    client = Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url)
    observations = fetch_observations(
        client,
        session_id=session_id,
        from_time=from_time,
        to_time=to_time,
    )
    report = aggregate_observations(observations)
    report["query"] = {
        "session_id": session_id,
        "from": from_time.isoformat(),
        "to": to_time.isoformat(),
        "hours": hours,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Langfuse observations for a time range.")
    parser.add_argument("--session-id", default=None, help="Optional Langfuse session id")
    parser.add_argument(
        "--session-json",
        default=None,
        help="Local session JSON to compare with Langfuse for the same session",
    )
    parser.add_argument(
        "--rate-card",
        default=None,
        help="Rate-card JSON file used for local STT/TTS cost calculation",
    )
    parser.add_argument("--hours", type=float, default=24, help="Lookback window (default 24)")
    args = parser.parse_args()
    report = build_report(session_id=args.session_id, hours=args.hours)
    if args.session_json:
        session_summary = json.loads(Path(args.session_json).read_text(encoding="utf-8"))
        session_id = (session_summary.get("session") or {}).get("session_id")
        if not session_id:
            raise ValueError("Local session JSON does not contain session.session_id")
        if args.session_id and args.session_id != session_id:
            raise ValueError("--session-id does not match session.session_id in --session-json")
        rate_card = RateCard.from_json(args.rate_card) if args.rate_card else RateCard.from_environment()
        report["session_comparison"] = compare_session(
            session_summary,
            fetch_observations(
                Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    base_url=os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST"),
                ),
                session_id=session_id,
            ),
            rate_card,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
