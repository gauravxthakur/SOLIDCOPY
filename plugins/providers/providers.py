from livekit.agents import llm, stt
from livekit.plugins import google, openai, deepgram, assemblyai

def get_llm_provider(provider_type: str) -> llm.LLM:
    
    if provider_type == "google":
        return google.LLM(model="gemini-2.5-flash")

    elif provider_type == "openai":
        return openai.LLM(model="gpt-4o-mini")

    raise ValueError(f"Unknown LLM provider type: {provider_type}")


def get_stt_provider(provider_type: str = "deepgram") -> stt.STT:

    if provider_type == "deepgram":
        return deepgram.STT(model="nova-3", language="en")
        
    elif provider_type == "assemblyai":
        return assemblyai.STT()

    raise ValueError(f"Unknown STT provider type: {provider_type}")