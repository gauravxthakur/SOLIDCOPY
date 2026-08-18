import json
import os
import sys
import pandas as pd
from typing import Dict, Any, List

def extract_record_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics directly from Langfuse trace export structure."""
    
    # Direct top-level fields in Langfuse export
    metrics = {
        "id": record.get("id"),
        "trace_id": record.get("traceId"),
        "name": record.get("name", "unknown"),
        "type": record.get("type", "unknown"),
        "model": record.get("providedModelName") or record.get("modelId"),
        "latency_ms": record.get("latencyMs"),
        "ttft_ms": record.get("timeToFirstTokenMs"),
        "ttfb_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "total_cost": record.get("totalCost", 0.0),
        "interrupted": None,
        "e2e_latency_s": None
    }

    # Extract usage from top-level usageDetails
    usage = record.get("usageDetails", {})
    if isinstance(usage, dict) and usage:
        metrics["input_tokens"] = usage.get("input") or usage.get("prompt_tokens")
        metrics["output_tokens"] = usage.get("output") or usage.get("completion_tokens")
        metrics["total_tokens"] = usage.get("total") or usage.get("total_tokens")

    # Extract metadata attributes
    meta = record.get("metadata", {})
    if isinstance(meta, dict):
        # Latency extractions
        if "attributes.lk.response.ttft" in meta:
            try:
                metrics["ttft_ms"] = float(meta["attributes.lk.response.ttft"]) * 1000
            except (ValueError, TypeError):
                pass

        if "attributes.lk.response.ttfb" in meta:
            try:
                metrics["ttfb_ms"] = float(meta["attributes.lk.response.ttfb"]) * 1000
            except (ValueError, TypeError):
                pass

        if "attributes.lk.interrupted" in meta:
            metrics["interrupted"] = str(meta["attributes.lk.interrupted"]).lower() == "true"

        if "attributes.lk.e2e_latency" in meta:
            try:
                metrics["e2e_latency_s"] = float(meta["attributes.lk.e2e_latency"])
            except (ValueError, TypeError):
                pass

        # Embedded JSON metrics fallback
        for llm_key in ["attributes.lk.llm_metrics", "lk.llm_metrics"]:
            if llm_key in meta:
                try:
                    val = meta[llm_key]
                    llm_data = json.loads(val) if isinstance(val, str) else val
                    if isinstance(llm_data, dict):
                        if "ttft" in llm_data and llm_data["ttft"] is not None:
                            metrics["ttft_ms"] = float(llm_data["ttft"]) * 1000
                        metrics["input_tokens"] = metrics["input_tokens"] or llm_data.get("prompt_tokens")
                        metrics["output_tokens"] = metrics["output_tokens"] or llm_data.get("completion_tokens")
                        metrics["total_tokens"] = metrics["total_tokens"] or llm_data.get("total_tokens")
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

        for tts_key in ["attributes.lk.tts_metrics", "lk.tts_metrics"]:
            if tts_key in meta:
                try:
                    val = meta[tts_key]
                    tts_data = json.loads(val) if isinstance(val, str) else val
                    if isinstance(tts_data, dict) and "ttfb" in tts_data and tts_data["ttfb"] is not None:
                        metrics["ttfb_ms"] = float(tts_data["ttfb"]) * 1000
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

    return metrics


def analyze_langfuse_jsonl(file_path: str):
    """Parses JSONL file and prints structured metrics summary."""
    if not os.path.exists(file_path):
        print(f"Error: Trace file not found at path '{file_path}'")
        return

    records: List[Dict[str, Any]] = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                records.append(extract_record_metrics(record))
            except json.JSONDecodeError:
                continue

    if not records:
        print(f"No valid records found in {file_path}")
        return

    df = pd.DataFrame(records)

    print("\n" + "="*50)
    print("        LANGFUSE TELEMETRY METRICS SUMMARY        ")
    print("="*50 + "\n")
    print(f"Total Telemetry Spans Analyzed: {len(df)}\n")

    # LLM Metrics
    llm_df = df[
        (df["type"].str.upper() == "GENERATION") | 
        (df["ttft_ms"].notna()) |
        (df["name"].str.contains("llm|generation|openai|groq", case=False, na=False))
    ]
    if not llm_df.empty:
        print("--- LLM Metrics ---")
        print(f"Total LLM Spans:     {len(llm_df)}")
        valid_ttft = llm_df.dropna(subset=["ttft_ms"])
        if not valid_ttft.empty:
            print(f"Avg TTFT:             {valid_ttft['ttft_ms'].mean():.2f} ms")
            print(f"Min TTFT:             {valid_ttft['ttft_ms'].min():.2f} ms")
            print(f"Max TTFT:             {valid_ttft['ttft_ms'].max():.2f} ms")
        if "total_tokens" in llm_df and llm_df["total_tokens"].notna().any():
            print(f"Total Tokens:         {int(llm_df['total_tokens'].dropna().sum())}")
        if "total_cost" in llm_df and llm_df["total_cost"].notna().any():
            print(f"Total Cost ($):       ${llm_df['total_cost'].dropna().sum():.6f}")
        print()
    else:
        print("No LLM/GENERATION spans found in trace data.\n")

    # TTS Metrics
    tts_df = df[
        (df["ttfb_ms"].notna()) |
        (df["name"].str.contains("tts|elevenlabs|cartesia|deepgram", case=False, na=False))
    ]
    if not tts_df.empty:
        print("--- TTS Metrics ---")
        print(f"Total TTS Spans:     {len(tts_df)}")
        valid_ttfb = tts_df.dropna(subset=["ttfb_ms"])
        if not valid_ttfb.empty:
            print(f"Avg TTFB:             {valid_ttfb['ttfb_ms'].mean():.2f} ms")
            print(f"Min TTFB:             {valid_ttfb['ttfb_ms'].min():.2f} ms")
            print(f"Max TTFB:             {valid_ttfb['ttfb_ms'].max():.2f} ms")
        print()

    # Agent Turn Metrics
    turn_df = df[
        (df["e2e_latency_s"].notna()) |
        (df["name"].str.contains("agent_turn|user_turn|session", case=False, na=False))
    ]
    if not turn_df.empty:
        print("--- Agent Turn Metrics ---")
        print(f"Total Turns:          {len(turn_df)}")
        valid_e2e = turn_df.dropna(subset=["e2e_latency_s"])
        if not valid_e2e.empty:
            print(f"Avg E2E Latency:      {valid_e2e['e2e_latency_s'].mean():.3f} s")
        if "interrupted" in turn_df and turn_df["interrupted"].notna().any():
            print(f"Interrupted Turns:    {turn_df['interrupted'].sum()} / {len(turn_df)}")
        print()


if __name__ == "__main__":
    trace_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join("metrics", "traces", "traces.jsonl")
    analyze_langfuse_jsonl(trace_file)