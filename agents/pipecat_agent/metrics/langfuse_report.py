"""Historical analytics from the Langfuse API for Pipecat sessions.

Queries Langfuse observations and generates aggregated analytics over a lookback
window or for a specific session ID, and supports local-vs-Langfuse session comparisons.

Usage:
  python -m metrics.langfuse_report --hours 24
  python -m metrics.langfuse_report --session-id sess_123
  python -m metrics.langfuse_report --session-json metrics/sessions/sess_123.json
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

from metrics.costs import CostCalculator, RateCard

CANONICAL_NAMES = {
    "conversation",
    "turn",
    "llm_request",
    "tts_request",
    "stt_request",
    "llm",
    "tts",
    "stt",
    "user_bot_latency",
    "tool_execution",
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
    if name in ("conversation", "agent_session"):
        stats["session_duration_seconds"] = _latency_stats(latencies.get(name, []))
    else:
        stats["latency_seconds"] = _latency_stats(latencies.get(name, []))

    if name in ("llm_request", "llm"):
        stats["ttft_seconds"] = _latency_stats(values.get("ttft", []))
    elif name in ("tts_request", "tts"):
        stats["ttfb_seconds"] = _latency_stats(values.get("ttfb", []))
        stats["ttfa_seconds"] = _latency_stats(values.get("ttfa", []))
    elif name in ("stt_request", "stt"):
        stats["audio_duration_seconds"] = _latency_stats(values.get("audio", []))
    elif name == "turn":
        stats["user_bot_latency_seconds"] = _latency_stats(values.get("user_bot_latency", []))
    return stats


def _usage_value(obs: Any, *keys: str) -> float | None:
    value = _first_not_none(obs, *keys)
    if value is not None:
        return _float(value)

    # Check root level aliases
    if any(k in ("input_tokens", "prompt_tokens", "input") for k in keys):
        val = _first_not_none(obs, "input_tokens", "prompt_tokens", "input", "prompt", "inputTokens", "promptTokens", "gen_ai.usage.input_tokens")
        if val is not None:
            return _float(val)
    if any(k in ("output_tokens", "completion_tokens", "output") for k in keys):
        val = _first_not_none(obs, "output_tokens", "completion_tokens", "output", "completion", "outputTokens", "completionTokens", "gen_ai.usage.output_tokens")
        if val is not None:
            return _float(val)
    if any(k in ("cached_input_tokens", "prompt_cached_tokens", "cached_tokens", "cache_read_input_tokens") for k in keys):
        val = _first_not_none(obs, "cached_input_tokens", "prompt_cached_tokens", "cached_tokens", "cache_read_input_tokens", "cachedTokens", "gen_ai.usage.cache_read.input_tokens")
        if val is not None:
            return _float(val)

    usage = _first_not_none(obs, "usage", "usage_details", "usageDetails")
    if usage is not None:
        val = _first_not_none(usage, *keys)
        if val is not None:
            return _float(val)
        if any(k in ("input_tokens", "prompt_tokens", "input") for k in keys):
            val = _first_not_none(usage, "input", "prompt", "inputTokens", "promptTokens", "input_tokens", "prompt_tokens", "gen_ai.usage.input_tokens")
            if val is not None:
                return _float(val)
        if any(k in ("output_tokens", "completion_tokens", "output") for k in keys):
            val = _first_not_none(usage, "output", "completion", "outputTokens", "completionTokens", "output_tokens", "completion_tokens", "gen_ai.usage.output_tokens")
            if val is not None:
                return _float(val)
        if any(k in ("cached_input_tokens", "prompt_cached_tokens", "cached_tokens", "cache_read_input_tokens") for k in keys):
            details = _first_not_none(
                usage,
                "input_token_details",
                "inputTokenDetails",
                "prompt_token_details",
                "promptTokenDetails",
            )
            if details is not None:
                val = _first_not_none(details, "cached_tokens", "cachedTokens", "cache_read_tokens")
                if val is not None:
                    return _float(val)
            val = _first_not_none(usage, "cached_tokens", "cachedTokens", "cached_input_tokens", "cache_read_input_tokens", "gen_ai.usage.cache_read.input_tokens")
            if val is not None:
                return _float(val)
    return None


def _sum_usage(observations: list[Any], *keys: str) -> float | None:
    values = [_usage_value(observation, *keys) for observation in observations]
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def aggregate_observations(observations: list[Any]) -> dict[str, Any]:
    """Build a report from Langfuse observation objects or dicts."""
    by_name: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    by_session: Counter[str] = Counter()
    latencies: dict[str, list[float]] = defaultdict(list)
    metric_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
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
        latency = _float(_first_not_none(obs, "latency", "duration", "duration_seconds"))
        ttft = _float(_first_not_none(obs, "time_to_first_token", "timeToFirstToken", "ttft", "ttfb_secs"))
        ttfb = _float(_first_not_none(obs, "time_to_first_byte", "timeToFirstByte", "ttfb"))
        ttfa = _float(_first_not_none(obs, "time_to_first_audio", "timeToFirstAudio", "ttfa"))
        ub_latency = _float(_first_not_none(obs, "user_bot_latency_seconds", "turn.user_bot_latency_seconds"))
        audio_duration = _float(
            _first_not_none(obs, "audio_duration", "audioDuration", "audio_duration_seconds", "audio_seconds")
        )
        cost_value = _float(_first_not_none(obs, "total_cost", "totalCost", "cost_usd"))
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
        if ttfa is not None:
            metric_values[name]["ttfa"].append(ttfa)
        if ub_latency is not None:
            metric_values[name]["user_bot_latency"].append(ub_latency)
        if audio_duration is not None:
            metric_values[name]["audio"].append(audio_duration)
        total_cost += cost
        cost_by_name[name] += cost

        if obs_type.upper() in ("TOOL", "FUNCTION") or name.startswith("tool_") or name == "tool_execution":
            tool_total += 1
            tool_by_name[name] += 1
            if level in ("ERROR", "FAILED") or _obs_get(obs, "status") == "error":
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
            name: _latency_stats(latencies.get(name, []))
            for name in (
                "conversation",
                "turn",
                "llm_request",
                "tts_request",
                "stt_request",
                "llm",
                "tts",
                "stt",
            )
            if latencies.get(name)
        },
        "ttft": {
            name: _latency_stats(metric_values[name].get("ttft", []))
            for name in ("llm_request", "llm")
            if metric_values[name].get("ttft")
        },
        "ttfb": {
            name: _latency_stats(metric_values[name].get("ttfb", []))
            for name in ("tts_request", "tts", "llm_request", "llm")
            if metric_values[name].get("ttfb")
        },
        "audio_duration": {
            name: _latency_stats(metric_values[name].get("audio", []))
            for name in ("stt_request", "stt")
            if metric_values[name].get("audio")
        },
    }


def _local_cost_line(session_summary: dict[str, Any], name: str) -> dict[str, Any]:
    line = ((session_summary.get("cost_breakdown") or {}).get("lines") or {}).get(name)
    return line or {"status": "not_applicable", "cost_usd": None}


def _local_llm_summary(session_summary: dict[str, Any]) -> dict[str, Any]:
    records = (session_summary.get("turns") or {}).get("records") or []
    llm = session_summary.get("llm") or {}
    llm_records = [record for record in records if record.get("prompt_tokens") is not None]
    cost_line = _local_cost_line(session_summary, "llm")
    models = sorted(list(llm.get("models", {}).keys()) or [r.get("llm_model") for r in llm_records if r.get("llm_model")])
    return {
        "model": models,
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
        "ttfb_seconds": llm.get("ttfb_seconds"),
        "ttfat_seconds": llm.get("ttfat_seconds"),
        "cost_usd": cost_line.get("cost_usd"),
        "cost_status": cost_line.get("status"),
    }


def compare_session(
    session_summary: dict[str, Any],
    observations: list[Any],
    rate_card: RateCard | None = None,
) -> dict[str, Any]:
    """Compare one local Pipecat session summary with its Langfuse observations."""
    session_id = (session_summary.get("session") or {}).get("session_id")
    observation_session_ids = {
        str(obs_sid)
        for obs in observations
        for obs_sid in [
            _first_not_none(obs, "session_id", "sessionId")
        ]
        if obs_sid is not None
    }
    if session_id and observation_session_ids:
        observations = [
            obs
            for obs in observations
            if str(_first_not_none(obs, "session_id", "sessionId")) == str(session_id)
        ]

    langfuse_llm = [
        obs
        for obs in observations
        if str(_obs_get(obs, "name") or "") in ("llm_request", "llm", "generation", "chat")
        or str(_obs_get(obs, "type") or "").upper() == "GENERATION"
    ]

    langfuse_models = sorted({
        str(m)
        for obs in langfuse_llm
        for m in [_first_not_none(obs, "model", "model_name", "modelName")]
        if m is not None
    })

    langfuse_input_tokens = sum(
        _usage_value(obs, "input_tokens", "prompt_tokens", "inputTokens", "promptTokens", "gen_ai.usage.input_tokens") or 0
        for obs in langfuse_llm
    )
    langfuse_cached_tokens = _sum_usage(
        langfuse_llm,
        "cached_input_tokens",
        "prompt_cached_tokens",
        "cachedTokens",
        "cache_read_input_tokens",
        "gen_ai.usage.cache_read.input_tokens",
    )
    langfuse_completion_tokens = sum(
        _usage_value(
            obs,
            "output_tokens",
            "completion_tokens",
            "outputTokens",
            "completionTokens",
            "gen_ai.usage.output_tokens",
        ) or 0
        for obs in langfuse_llm
    )

    ttfb_values = [
        val
        for obs in langfuse_llm
        for val in [_float(_first_not_none(obs, "time_to_first_token", "timeToFirstToken", "ttft", "ttfb"))]
        if val is not None
    ]

    langfuse_summary = {
        "model": langfuse_models,
        "request_count": len(langfuse_llm),
        "input_tokens": langfuse_input_tokens,
        "cached_tokens": langfuse_cached_tokens,
        "completion_tokens": langfuse_completion_tokens,
        "ttfb_seconds": _latency_stats(ttfb_values),
        "cost_usd": round(sum(
            _float(_first_not_none(obs, "total_cost", "totalCost", "cost_usd")) or 0
            for obs in langfuse_llm
        ), 6),
    }

    local = _local_llm_summary(session_summary)
    langfuse_request_count = langfuse_summary["request_count"]
    local_request_count = local["request_count"]

    rc = rate_card if rate_card is not None else RateCard.from_environment()
    calculator = CostCalculator(rc)
    local_costs = {"llm": [], "stt": [], "tts": []}
    local_cost_statuses = {"llm": [], "stt": [], "tts": []}

    for record in (session_summary.get("turns") or {}).get("records") or []:
        breakdown = calculator.calculate_turn(record)
        for name, line in breakdown.get("lines", {}).items():
            local_cost_statuses[name].append(line.get("status"))
            if line.get("status") == "measured" and line.get("cost_usd") is not None:
                local_costs[name].append(line["cost_usd"])

    own_costs = {
        name: {
            "status": (
                "missing_rate" if "missing_rate" in local_cost_statuses[name]
                else "measured" if local_costs[name]
                else "not_applicable"
            ),
            "cost_usd": round(sum(local_costs[name]), 6) if local_costs[name] else None,
        }
        for name in ("stt", "tts", "llm")
    }

    return {
        "session_id": session_id,
        "langfuse_observation_count": len(observations),
        "llm": {
            "local": local,
            "langfuse": langfuse_summary,
            "differences": {
                key: {
                    "local": local.get(key),
                    "langfuse": langfuse_summary.get(key),
                }
                for key in (
                    "model",
                    "request_count",
                    "input_tokens",
                    "cached_tokens",
                    "completion_tokens",
                    "cost_usd",
                )
            },
        },
        "request_count_check": {
            "status": "match" if local_request_count == langfuse_request_count else "mismatch",
            "local": local_request_count,
            "langfuse": langfuse_request_count,
            "difference": langfuse_request_count - local_request_count,
        },
        "own_rate_card_costs": {
            "stt": own_costs["stt"],
            "tts": own_costs["tts"],
            "llm": own_costs["llm"],
        },
        "notes": [
            "Langfuse LLM comparison aligns with exact LLM generation spans/records.",
            "STT and TTS costs are priced from local usage metrics and configured rate cards.",
            "Zero or missing STT/TTS cost in Langfuse is not interpreted as free usage.",
        ],
    }


def fetch_observations(
    client: Any,
    *,
    session_id: str | None = None,
    from_time: datetime | None = None,
    to_time: datetime | None = None,
    limit: int = 100,
    max_pages: int = 20,
) -> list[Any]:
    """Fetch observation pages from Langfuse API client."""
    filters = []
    if session_id:
        filters.append(
            {"type": "string", "column": "sessionId", "operator": "=", "value": session_id}
        )
    filter_json = json.dumps(filters) if filters else None
    rows: list[Any] = []
    cursor = None

    for _ in range(max_pages):
        page = None
        if hasattr(client, "api") and hasattr(client.api, "observations"):
            page = client.api.observations.get_many(
                fields=OBSERVATION_FIELDS,
                limit=limit,
                cursor=cursor,
                from_start_time=from_time,
                to_start_time=to_time,
                filter=filter_json,
            )
        elif hasattr(client, "get_observations"):
            page = client.get_observations(
                limit=limit,
                cursor=cursor,
                from_start_time=from_time,
                to_start_time=to_time,
                session_id=session_id,
            )
        elif hasattr(client, "observations"):
            page = client.observations(session_id=session_id)

        if page is None:
            break

        data = getattr(page, "data", None)
        if data is None and isinstance(page, list):
            data = page
        elif data is None and isinstance(page, dict):
            data = page.get("data", [])

        rows.extend(data or [])
        meta = getattr(page, "meta", None) or (page if isinstance(page, dict) else {})
        raw_cursor = getattr(meta, "cursor", None) or getattr(meta, "next_cursor", None)
        if isinstance(meta, dict):
            raw_cursor = meta.get("cursor") or meta.get("next_cursor")
        cursor = str(raw_cursor) if isinstance(raw_cursor, str) and raw_cursor else None
        if not cursor or not data:
            break

    return rows


def build_report(
    *,
    session_id: str | None = None,
    hours: float = 24,
    from_time: datetime | None = None,
    to_time: datetime | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Build Langfuse report over a time window or session."""
    load_dotenv()
    if client is None:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or os.getenv("PIPECAT_LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY") or os.getenv("PIPECAT_LANGFUSE_SECRET_KEY")
        base_url = (
            os.getenv("LANGFUSE_BASE_URL")
            or os.getenv("LANGFUSE_HOST")
            or os.getenv("PIPECAT_LANGFUSE_BASE_URL")
            or os.getenv("PIPECAT_LANGFUSE_HOST")
        )
        if not public_key or not secret_key or not base_url:
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL must be configured"
            )
        try:
            from langfuse import Langfuse
            client = Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url)
        except ImportError as exc:
            raise RuntimeError("The 'langfuse' package is required to fetch reports from Langfuse.") from exc

    end_time = to_time or datetime.now(timezone.utc)
    start_time = from_time or (end_time - timedelta(hours=hours))

    observations = fetch_observations(
        client,
        session_id=session_id,
        from_time=start_time,
        to_time=end_time,
    )

    report = aggregate_observations(observations)
    report["query"] = {
        "session_id": session_id,
        "from": start_time.isoformat(),
        "to": end_time.isoformat(),
        "hours": hours if from_time is None else None,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate historical Langfuse observations and compare with local session summaries."
    )
    parser.add_argument("--session-id", default=None, help="Optional Langfuse session id filter")
    parser.add_argument(
        "--session-json",
        default=None,
        help="Path to local session JSON to compare with Langfuse data",
    )
    parser.add_argument(
        "--rate-card",
        default=None,
        help="Rate-card JSON file used for local STT/TTS/LLM cost calculation",
    )
    parser.add_argument("--hours", type=float, default=24, help="Lookback window in hours (default 24)")
    parser.add_argument("--output", default=None, help="Optional output JSON file path to save report")

    args = parser.parse_args()
    session_summary = None
    session_id = args.session_id

    if args.session_json:
        session_path = Path(args.session_json)
        if not session_path.exists():
            raise FileNotFoundError(f"Session JSON file not found: {session_path}")
        session_summary = json.loads(session_path.read_text(encoding="utf-8"))
        local_sid = (session_summary.get("session") or {}).get("session_id")
        if not local_sid:
            raise ValueError("Local session JSON does not contain session.session_id")
        if session_id and session_id != local_sid:
            raise ValueError("--session-id does not match session.session_id in --session-json")
        session_id = local_sid

    report = build_report(session_id=session_id, hours=args.hours)

    if session_summary is not None:
        rate_card = RateCard.from_json(args.rate_card) if args.rate_card else RateCard.from_environment()
        report["session_comparison"] = compare_session(
            session_summary,
            fetch_observations(
                client=None,
                session_id=session_id,
            ),
            rate_card,
        )

    output_json = json.dumps(report, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json, encoding="utf-8")
        print(f"Report written to {out_path}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
