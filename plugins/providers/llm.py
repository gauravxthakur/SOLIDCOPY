from livekit.agents import llm
from livekit.plugins import google, openai

def get_llm_provider(provider_type: str) -> llm.LLM:
    
    if provider_type == "google":
        return google.LLM(model="gemini-2.5-flash")

    elif provider_type == "openai":
        return openai.LLM(model="gpt-4o-mini")

    raise ValueError(f"Unknown provider type: {provider_type}")