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
        self.assertIsNone(stt["utterance_count"])

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
