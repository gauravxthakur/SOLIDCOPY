import json
import os
import sys
import pandas as pd
from typing import Dict, Any, List

def extract_record_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comprehensive metrics directly from Langfuse trace export structure."""
    
    # Direct top-level fields in Langfuse export
    metrics = {
        # Span identifiers & metadata
        "id": record.get("id"),
        "trace_id": record.get("traceId"),
        "name": record.get("name", "unknown"),
        "type": record.get("type", "unknown"),
        "model": record.get("providedModelName") or record.get("modelId"),
        
        # LLM Metrics
        "latency_ms": record.get("latencyMs"),
        "ttft_ms": record.get("timeToFirstTokenMs"),
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "total_cost": record.get("totalCost", 0.0),
        
        # TTS Metrics
        "ttfb_ms": None,
        "tts_duration_ms": None,
        "tts_audio_duration_s": None,
        "tts_characters_count": None,
        "tts_cancelled": None,
        "tts_streamed": None,
        
        # STT/Speech Recognition Metrics
        "transcription_delay_s": None,
        "end_of_turn_delay_s": None,
        "transcript_confidence": None,
        
        # Interruption Detection Metrics
        "interruption_detected": None,
        "interruption_detection_delay_s": None,
        "interruption_prediction_duration_s": None,
        "interruption_probability": None,
        "is_interruption": None,
        
        # Agent Turn Metrics
        "e2e_latency_s": None,
        
        # End-of-Utterance (EOU) Detection
        "eou_probability": None,
        "eou_endpointing_delay_s": None,
        "eou_source": None,
        "eou_language": None,
        
        # Request/Retry Tracking
        "provider_request_ids": None,
        "retry_count": None,
        
        # Session/Resource Info
        "agent_name": None,
        "job_id": None,
        "room_name": None,
        "agent_label": None,
        "service_name": None,
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
        # === LLM & Response Latencies ===
        if "attributes.lk.response.ttft" in meta:
            try:
                metrics["ttft_ms"] = float(meta["attributes.lk.response.ttft"]) * 1000
            except (ValueError, TypeError):
                pass

        # === TTS Metrics (Response TTFB) ===
        if "attributes.lk.response.ttfb" in meta:
            try:
                metrics["ttfb_ms"] = float(meta["attributes.lk.response.ttfb"]) * 1000
            except (ValueError, TypeError):
                pass

        # === Interruption Detection ===
        if "attributes.lk.interrupted" in meta:
            metrics["interruption_detected"] = str(meta["attributes.lk.interrupted"]).lower() == "true"

        if "attributes.lk.is_interruption" in meta:
            metrics["is_interruption"] = str(meta["attributes.lk.is_interruption"]).lower() == "true"

        if "attributes.lk.interruption.detection_delay" in meta:
            try:
                metrics["interruption_detection_delay_s"] = float(meta["attributes.lk.interruption.detection_delay"])
            except (ValueError, TypeError):
                pass

        if "attributes.lk.interruption.prediction_duration" in meta:
            try:
                metrics["interruption_prediction_duration_s"] = float(meta["attributes.lk.interruption.prediction_duration"])
            except (ValueError, TypeError):
                pass

        if "attributes.lk.interruption.total_duration" in meta:
            try:
                metrics["interruption_prediction_duration_s"] = float(meta["attributes.lk.interruption.total_duration"])
            except (ValueError, TypeError):
                pass

        if "attributes.lk.interruption.probability" in meta:
            try:
                metrics["interruption_probability"] = float(meta["attributes.lk.interruption.probability"])
            except (ValueError, TypeError):
                pass

        # === Agent Turn Metrics ===
        if "attributes.lk.e2e_latency" in meta:
            try:
                metrics["e2e_latency_s"] = float(meta["attributes.lk.e2e_latency"])
            except (ValueError, TypeError):
                pass

        # === STT/Speech Recognition Metrics ===
        if "attributes.lk.transcription_delay" in meta:
            try:
                metrics["transcription_delay_s"] = float(meta["attributes.lk.transcription_delay"])
            except (ValueError, TypeError):
                pass

        if "attributes.lk.end_of_turn_delay" in meta:
            try:
                metrics["end_of_turn_delay_s"] = float(meta["attributes.lk.end_of_turn_delay"])
            except (ValueError, TypeError):
                pass

        if "attributes.lk.transcript_confidence" in meta:
            try:
                metrics["transcript_confidence"] = float(meta["attributes.lk.transcript_confidence"])
            except (ValueError, TypeError):
                pass

        # === End-of-Utterance (EOU) Detection ===
        if "attributes.lk.eou.probability" in meta:
            try:
                metrics["eou_probability"] = float(meta["attributes.lk.eou.probability"])
            except (ValueError, TypeError):
                pass

        if "attributes.lk.eou.endpointing_delay" in meta:
            try:
                metrics["eou_endpointing_delay_s"] = float(meta["attributes.lk.eou.endpointing_delay"])
            except (ValueError, TypeError):
                pass

        if "attributes.lk.eou.source" in meta:
            metrics["eou_source"] = meta["attributes.lk.eou.source"]

        if "attributes.lk.eou.language" in meta:
            metrics["eou_language"] = meta["attributes.lk.eou.language"]

        # === Request/Retry Tracking ===
        if "attributes.lk.provider_request_ids" in meta:
            try:
                req_ids = meta["attributes.lk.provider_request_ids"]
                if isinstance(req_ids, str):
                    req_ids = json.loads(req_ids)
                if isinstance(req_ids, list):
                    metrics["provider_request_ids"] = len(req_ids)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        if "attributes.lk.retry_count" in meta:
            try:
                metrics["retry_count"] = int(meta["attributes.lk.retry_count"])
            except (ValueError, TypeError):
                pass

        # === Session/Resource Info ===
        if "attributes.lk.agent_name" in meta:
            metrics["agent_name"] = meta["attributes.lk.agent_name"]

        if "attributes.lk.job_id" in meta:
            metrics["job_id"] = meta["attributes.lk.job_id"]

        if "attributes.lk.room_name" in meta:
            metrics["room_name"] = meta["attributes.lk.room_name"]

        if "attributes.lk.agent_label" in meta:
            metrics["agent_label"] = meta["attributes.lk.agent_label"]

        if "resourceAttributes.service.name" in meta:
            metrics["service_name"] = meta["resourceAttributes.service.name"]

        # === Embedded JSON: LLM Metrics ===
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

        # === Embedded JSON: TTS Metrics ===
        for tts_key in ["attributes.lk.tts_metrics", "lk.tts_metrics"]:
            if tts_key in meta:
                try:
                    val = meta[tts_key]
                    tts_data = json.loads(val) if isinstance(val, str) else val
                    if isinstance(tts_data, dict):
                        if "ttfb" in tts_data and tts_data["ttfb"] is not None:
                            metrics["ttfb_ms"] = float(tts_data["ttfb"]) * 1000
                        if "duration" in tts_data:
                            metrics["tts_duration_ms"] = float(tts_data["duration"]) * 1000
                        if "audio_duration" in tts_data:
                            metrics["tts_audio_duration_s"] = float(tts_data["audio_duration"])
                        if "characters_count" in tts_data:
                            metrics["tts_characters_count"] = tts_data["characters_count"]
                        if "cancelled" in tts_data:
                            metrics["tts_cancelled"] = tts_data["cancelled"]
                        if "streamed" in tts_data:
                            metrics["tts_streamed"] = tts_data["streamed"]
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

    return metrics


def analyze_langfuse_jsonl(file_path: str):
    """Parses JSONL file and prints comprehensive metrics summary."""
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

    print("\n" + "="*70)
    print("       COMPREHENSIVE AGENT TELEMETRY METRICS BREAKDOWN       ")
    print("="*70 + "\n")
    print(f"Total Telemetry Spans Analyzed: {len(df)}\n")

    # === LLM METRICS ===
    llm_df = df[
        (df["type"].str.upper() == "GENERATION") | 
        (df["ttft_ms"].notna()) |
        (df["name"].str.contains("llm|generation|openai|groq", case=False, na=False))
    ]
    if not llm_df.empty:
        print("▶ LLM METRICS")
        print("-" * 70)
        print(f"  Total LLM Spans:          {len(llm_df)}")
        
        valid_ttft = llm_df.dropna(subset=["ttft_ms"])
        if not valid_ttft.empty:
            print(f"  Time to First Token (TTFT):")
            print(f"    ├─ Avg:                {valid_ttft['ttft_ms'].mean():.2f} ms")
            print(f"    ├─ Min:                {valid_ttft['ttft_ms'].min():.2f} ms")
            print(f"    └─ Max:                {valid_ttft['ttft_ms'].max():.2f} ms")
        
        valid_latency = llm_df.dropna(subset=["latency_ms"])
        if not valid_latency.empty:
            print(f"  Total Latency:")
            print(f"    ├─ Avg:                {valid_latency['latency_ms'].mean():.2f} ms")
            print(f"    └─ Sum:                {valid_latency['latency_ms'].sum():.2f} ms")
        
        # Token breakdown
        input_tokens = llm_df["input_tokens"].dropna().sum()
        output_tokens = llm_df["output_tokens"].dropna().sum()
        total_tokens = llm_df["total_tokens"].dropna().sum()
        print(f"  Token Usage:")
        print(f"    ├─ Input Tokens:        {int(input_tokens)}")
        print(f"    ├─ Output Tokens:       {int(output_tokens)}")
        print(f"    └─ Total Tokens:        {int(total_tokens)}")
        
        # Cost breakdown
        total_cost = llm_df["total_cost"].dropna().sum()
        print(f"  Total Cost:               ${total_cost:.6f}")
        print()
    else:
        print("▶ LLM METRICS: No data found\n")

    # === TTS METRICS ===
    tts_df = df[
        (df["ttfb_ms"].notna()) |
        (df["tts_duration_ms"].notna()) |
        (df["name"].str.contains("tts|elevenlabs|cartesia|deepgram", case=False, na=False))
    ]
    if not tts_df.empty:
        print("▶ TEXT-TO-SPEECH (TTS) METRICS")
        print("-" * 70)
        print(f"  Total TTS Spans:          {len(tts_df)}")
        
        valid_ttfb = tts_df.dropna(subset=["ttfb_ms"])
        if not valid_ttfb.empty:
            print(f"  Time to First Byte (TTFB):")
            print(f"    ├─ Avg:                {valid_ttfb['ttfb_ms'].mean():.2f} ms")
            print(f"    ├─ Min:                {valid_ttfb['ttfb_ms'].min():.2f} ms")
            print(f"    └─ Max:                {valid_ttfb['ttfb_ms'].max():.2f} ms")
        
        valid_duration = tts_df.dropna(subset=["tts_duration_ms"])
        if not valid_duration.empty:
            print(f"  Request Duration:")
            print(f"    ├─ Avg:                {valid_duration['tts_duration_ms'].mean():.2f} ms")
            print(f"    └─ Total:              {valid_duration['tts_duration_ms'].sum():.2f} ms")
        
        valid_audio = tts_df.dropna(subset=["tts_audio_duration_s"])
        if not valid_audio.empty:
            print(f"  Generated Audio Duration:")
            print(f"    ├─ Avg:                {valid_audio['tts_audio_duration_s'].mean():.3f} s")
            print(f"    └─ Total:              {valid_audio['tts_audio_duration_s'].sum():.3f} s")
        
        valid_chars = tts_df.dropna(subset=["tts_characters_count"])
        if not valid_chars.empty:
            print(f"  Characters Synthesized:   {int(valid_chars['tts_characters_count'].sum())}")
        
        # TTS flags
        if tts_df["tts_cancelled"].notna().any():
            cancelled_count = tts_df[tts_df["tts_cancelled"] == True].shape[0]
            print(f"  Cancelled Requests:       {cancelled_count}")
        
        if tts_df["tts_streamed"].notna().any():
            streamed_count = tts_df[tts_df["tts_streamed"] == True].shape[0]
            print(f"  Streamed Requests:        {streamed_count}")
        print()
    else:
        print("▶ TTS METRICS: No data found\n")

    # === SPEECH RECOGNITION (STT) METRICS ===
    stt_df = df[
        (df["transcription_delay_s"].notna()) |
        (df["transcript_confidence"].notna()) |
        (df["name"].str.contains("user_turn|stt|speech|transcrib", case=False, na=False))
    ]
    if not stt_df.empty:
        print("▶ SPEECH RECOGNITION (STT) METRICS")
        print("-" * 70)
        print(f"  Total STT Spans:          {len(stt_df)}")
        
        valid_trans = stt_df.dropna(subset=["transcription_delay_s"])
        if not valid_trans.empty:
            print(f"  Transcription Delay:")
            print(f"    ├─ Avg:                {valid_trans['transcription_delay_s'].mean():.3f} s")
            print(f"    ├─ Min:                {valid_trans['transcription_delay_s'].min():.3f} s")
            print(f"    └─ Max:                {valid_trans['transcription_delay_s'].max():.3f} s")
        
        valid_eot = stt_df.dropna(subset=["end_of_turn_delay_s"])
        if not valid_eot.empty:
            print(f"  End-of-Turn Delay:")
            print(f"    ├─ Avg:                {valid_eot['end_of_turn_delay_s'].mean():.3f} s")
            print(f"    └─ Max:                {valid_eot['end_of_turn_delay_s'].max():.3f} s")
        
        valid_conf = stt_df.dropna(subset=["transcript_confidence"])
        if not valid_conf.empty:
            print(f"  Transcript Confidence:")
            print(f"    ├─ Avg:                {valid_conf['transcript_confidence'].mean():.4f}")
            print(f"    ├─ Min:                {valid_conf['transcript_confidence'].min():.4f}")
            print(f"    └─ Max:                {valid_conf['transcript_confidence'].max():.4f}")
        print()
    else:
        print("▶ SPEECH RECOGNITION (STT) METRICS: No data found\n")

    # === INTERRUPTION DETECTION METRICS ===
    interruption_df = df[
        (df["interruption_detected"].notna()) |
        (df["interruption_probability"].notna()) |
        (df["is_interruption"].notna())
    ]
    if not interruption_df.empty:
        print("▶ INTERRUPTION DETECTION METRICS")
        print("-" * 70)
        print(f"  Total Events Analyzed:    {len(interruption_df)}")
        
        detected = interruption_df[interruption_df["interruption_detected"] == True].shape[0]
        print(f"  Detected Interruptions:   {detected} / {len(interruption_df)}")
        
        actual_int = interruption_df[interruption_df["is_interruption"] == True].shape[0]
        print(f"  Confirmed Interruptions:  {actual_int} / {len(interruption_df)}")
        
        valid_prob = interruption_df.dropna(subset=["interruption_probability"])
        if not valid_prob.empty:
            print(f"  Interruption Probability:")
            print(f"    ├─ Avg:                {valid_prob['interruption_probability'].mean():.4f}")
            print(f"    ├─ Min:                {valid_prob['interruption_probability'].min():.4f}")
            print(f"    └─ Max:                {valid_prob['interruption_probability'].max():.4f}")
        
        valid_det_delay = interruption_df.dropna(subset=["interruption_detection_delay_s"])
        if not valid_det_delay.empty:
            print(f"  Detection Delay:")
            print(f"    ├─ Avg:                {valid_det_delay['interruption_detection_delay_s'].mean():.3f} s")
            print(f"    └─ Max:                {valid_det_delay['interruption_detection_delay_s'].max():.3f} s")
        
        valid_pred = interruption_df.dropna(subset=["interruption_prediction_duration_s"])
        if not valid_pred.empty:
            print(f"  Prediction Duration:      {valid_pred['interruption_prediction_duration_s'].mean():.3f} s (avg)")
        print()
    else:
        print("▶ INTERRUPTION DETECTION METRICS: No data found\n")

    # === END-OF-UTTERANCE (EOU) DETECTION ===
    eou_df = df[
        (df["eou_probability"].notna()) |
        (df["eou_endpointing_delay_s"].notna())
    ]
    if not eou_df.empty:
        print("▶ END-OF-UTTERANCE (EOU) DETECTION")
        print("-" * 70)
        print(f"  Total EOU Events:         {len(eou_df)}")
        
        valid_eou_prob = eou_df.dropna(subset=["eou_probability"])
        if not valid_eou_prob.empty:
            print(f"  EOU Probability:")
            print(f"    ├─ Avg:                {valid_eou_prob['eou_probability'].mean():.4f}")
            print(f"    ├─ Min:                {valid_eou_prob['eou_probability'].min():.4f}")
            print(f"    └─ Max:                {valid_eou_prob['eou_probability'].max():.4f}")
        
        valid_eou_delay = eou_df.dropna(subset=["eou_endpointing_delay_s"])
        if not valid_eou_delay.empty:
            print(f"  Endpointing Delay:")
            print(f"    ├─ Avg:                {valid_eou_delay['eou_endpointing_delay_s'].mean():.3f} s")
            print(f"    └─ Max:                {valid_eou_delay['eou_endpointing_delay_s'].max():.3f} s")
        
        # EOU source breakdown
        if eou_df["eou_source"].notna().any():
            sources = eou_df["eou_source"].value_counts()
            print(f"  EOU Detection Source:")
            for source, count in sources.items():
                print(f"    └─ {source}: {count}")
        
        if eou_df["eou_language"].notna().any():
            lang = eou_df["eou_language"].value_counts().index[0]
            print(f"  Primary Language:         {lang}")
        print()
    else:
        print("▶ END-OF-UTTERANCE (EOU) DETECTION: No data found\n")

    # === AGENT TURN METRICS ===
    turn_df = df[
        (df["e2e_latency_s"].notna()) |
        (df["name"].str.contains("agent_turn|user_turn|session", case=False, na=False))
    ]
    if not turn_df.empty:
        print("▶ AGENT TURN METRICS")
        print("-" * 70)
        print(f"  Total Turns:              {len(turn_df)}")
        
        valid_e2e = turn_df.dropna(subset=["e2e_latency_s"])
        if not valid_e2e.empty:
            print(f"  End-to-End Latency:")
            print(f"    ├─ Avg:                {valid_e2e['e2e_latency_s'].mean():.3f} s")
            print(f"    ├─ Min:                {valid_e2e['e2e_latency_s'].min():.3f} s")
            print(f"    └─ Max:                {valid_e2e['e2e_latency_s'].max():.3f} s")
        
        interrupted = turn_df[turn_df["interruption_detected"] == True].shape[0]
        print(f"  Interrupted Turns:        {interrupted} / {len(turn_df)}")
        print()
    else:
        print("▶ AGENT TURN METRICS: No data found\n")

    # === REQUEST & RETRY TRACKING ===
    retry_df = df[df["retry_count"].notna()]
    if not retry_df.empty:
        print("▶ REQUEST & RETRY TRACKING")
        print("-" * 70)
        print(f"  Total Requests:           {len(retry_df)}")
        retried = retry_df[retry_df["retry_count"] > 0].shape[0]
        print(f"  Requests with Retries:    {retried}")
        if retried > 0:
            avg_retries = retry_df[retry_df["retry_count"] > 0]["retry_count"].mean()
            max_retries = retry_df["retry_count"].max()
            print(f"  Avg Retries (when retried): {avg_retries:.2f}")
            print(f"  Max Retries:              {int(max_retries)}")
        print()
    else:
        print("▶ REQUEST & RETRY TRACKING: No data found\n")

    # === SESSION & RESOURCE INFO ===
    resource_df = df[
        (df["agent_name"].notna()) |
        (df["job_id"].notna()) |
        (df["room_name"].notna())
    ]
    if not resource_df.empty:
        print("▶ SESSION & RESOURCE INFORMATION")
        print("-" * 70)
        if resource_df["agent_name"].notna().any():
            agents = resource_df["agent_name"].unique()
            print(f"  Agent Names:              {', '.join(filter(None, agents)) or 'N/A'}")
        if resource_df["room_name"].notna().any():
            rooms = resource_df["room_name"].unique()
            print(f"  Room Names:               {', '.join(filter(None, rooms)) or 'N/A'}")
        if resource_df["job_id"].notna().any():
            job_ids = resource_df["job_id"].unique()
            print(f"  Job IDs:                  {len(job_ids)} unique")
        if resource_df["service_name"].notna().any():
            services = resource_df["service_name"].value_counts()
            print(f"  Service Names:")
            for svc, count in services.items():
                print(f"    └─ {svc}: {count} spans")
        print()
    else:
        print("▶ SESSION & RESOURCE INFORMATION: No data found\n")

    print("="*70)
    print("END OF COMPREHENSIVE METRICS BREAKDOWN")
    print("="*70)


if __name__ == "__main__":
    trace_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join("metrics", "traces", "traces.jsonl")
    analyze_langfuse_jsonl(trace_file)