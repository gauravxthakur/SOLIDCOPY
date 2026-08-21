## Install dependencies
```bash
uv sync
```
## Add environment variables
Create a .env file with the help of .env.example

## Running the Agent
```bash
uv run simple_agent.py download-files # run only once
uv run simple_agent.py console
```

## Model architecture for fallback
LLM: OpenAI primary, Gemini backup
STT: AssemblyAI primary, Deepgram backup
TTS: Cartesia primary, Inworld backup

## Resources
Building Production-Ready Voice Agents with LiveKit https://worksh.app/tutorials/livekit-voice-agent