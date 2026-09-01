from livekit.agents import llm, stt, tts
from livekit.plugins import google, openai, deepgram, assemblyai, sarvam, cartesia


#-------------------------------LLM----------------------------------------------

def get_llm_provider(provider_type: str) -> llm.LLM:
    
    if provider_type == "google":
        return google.LLM(model="gemini-2.5-flash")

    elif provider_type == "openai":
        return openai.LLM(model="gpt-4o-mini")

    raise ValueError(f"Unknown LLM provider type: {provider_type}")


#-------------------------------STT----------------------------------------------

def get_stt_provider(provider_type: str = "deepgram") -> stt.STT:

    if provider_type == "deepgram":
        return deepgram.STT(model="nova-3", language="en")
        
    elif provider_type == "assemblyai":
        return assemblyai.STT()

    raise ValueError(f"Unknown STT provider type: {provider_type}")


#-------------------------------TTS----------------------------------------------

def get_tts_provider(provider_type: str = "cartesia") -> tts.TTS:

    if provider_type == "cartesia":
        return cartesia.TTS(model="sonic-3", voice="a167e0f3-df7e-4d52-a9c3-f949145efdab")
        
    elif provider_type == "sarvam":
        return sarvam.TTS(model="bulbul:v3", target_language_code="hi-IN", speaker="shubh")

    raise ValueError(f"Unknown TTS provider type: {provider_type}")