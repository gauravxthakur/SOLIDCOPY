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
import statistics
from typing import Any

from dotenv import load_dotenv
from langfuse import Langfuse

CANONICAL_NAMES = {
    "llm_request",
    "tts_request",
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


def aggregate_observations(observations: list[Any]) -> dict[str, Any]:
    """Build a report from Langfuse observation objects or dicts."""
    by_name: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    by_session: Counter[str] = Counter()
    latencies: dict[str, list[float]] = defaultdict(list)
    ttfts: dict[str, list[float]] = defaultdict(list)
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
        ttft = _float(_obs_get(obs, "time_to_first_token") or _obs_get(obs, "timeToFirstToken"))
        cost = _float(_obs_get(obs, "total_cost") or _obs_get(obs, "totalCost")) or 0.0

        by_name[name] += 1
        by_type[obs_type] += 1
        if session_id:
            by_session[str(session_id)] += 1
        if latency is not None:
            latencies[name].append(latency)
        if ttft is not None:
            ttfts[name].append(ttft)
        total_cost += cost
        cost_by_name[name] += cost

        if obs_type.upper() == "TOOL":
            tool_total += 1
            tool_by_name[name] += 1
            if level == "ERROR":
                tool_errors += 1

    named = {
        name: {
            "count": by_name[name],
            "cost_usd": round(cost_by_name[name], 6),
            "latency_seconds": _latency_stats(latencies.get(name, [])),
            "ttft_seconds": _latency_stats(ttfts.get(name, [])),
        }
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
            "agent_turn": _latency_stats(latencies.get("agent_turn", [])),
        },
        "ttft": {
            "llm_request": _latency_stats(ttfts.get("llm_request", [])),
            "tts_request": _latency_stats(ttfts.get("tts_request", [])),
        },
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
    parser.add_argument("--hours", type=float, default=24, help="Lookback window (default 24)")
    args = parser.parse_args()
    report = build_report(session_id=args.session_id, hours=args.hours)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
