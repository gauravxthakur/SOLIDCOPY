from pipecat.frames.frames import MetricsFrame
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TextAggregationMetricsData,
    TTFAMetricsData,
    TTFATMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

class MetricsLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, MetricsFrame):
            for d in frame.data:
                if isinstance(d, TTFBMetricsData):
                    print(f"!!! MetricsFrame: {frame}, ttfb: {d.value}")
                elif isinstance(d, TTFAMetricsData):
                    print(f"!!! MetricsFrame: {frame}, ttfa: {d.ttfa}, ttfb: {d.ttfb}, leading_silence: {d.leading_silence}")
                elif isinstance(d, TTFATMetricsData):
                    print(f"!!! MetricsFrame: {frame}, ttfat: {d.ttfat}, ttfb: {d.ttfb}, thinking_time: {d.thinking_time}")
                elif isinstance(d, ProcessingMetricsData):
                    print(f"!!! MetricsFrame: {frame}, processing: {d.value}")
                elif isinstance(d, LLMUsageMetricsData):
                    tokens = d.value
                    print(
                        f"!!! MetricsFrame: {frame}, prompt_tokens: {tokens.prompt_tokens}, completion_tokens: {tokens.completion_tokens}"
                    )
                elif isinstance(d, TextAggregationMetricsData):
                    print(f"!!! MetricsFrame: {frame}, text aggregation: {d.value}")
                elif isinstance(d, TTSUsageMetricsData):
                    print(f"!!! MetricsFrame: {frame}, characters: {d.value}")
        await self.push_frame(frame, direction)