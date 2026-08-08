## Installation

To install the core Agents library, along with plugins for popular model providers:

```bash
uv init
uv add "livekit-agents[openai,google,deepgram,cartesia]"
uv add "livekit-agents[silero,turn-detector]~=1.3" "livekit-plugins-noise-cancellation~=0.2" "python-dotenv"
```

Get a LiveKit Cloud API, create a .env file in root and paste your copied credentials in it:

```bash
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
```

Copy paste the starter agent code from https://worksh.app/tutorials/livekit-voice-agent/foundations and save it as `agent.py`

## Running the starter agent
```bash
uv run agent.py download-files # run only once
uv run agent.py console
```

## Adding Turn Detection
```bash
uv add "livekit-agents[turn-detector]"
```
Run the below command again:
```bash
uv run agent.py download-files # run only once
uv run agent.py console
```

## Personality
You can change the personality of the agent by altering the system prompt and the voice.
For the latter, choose a specific voice ID from https://docs.livekit.io/agents/models/tts/cartesia/#voices and paste it in the Cartesia model instance.

## Model architecture for fallback
LLM: OpenAI primary, Gemini backup
STT: AssemblyAI primary, Deepgram backup
TTS: Cartesia primary, Inworld backup

## Install LiveKit CLI
```bash
winget install LiveKit.LiveKitCLI
```
Restart your editor and check if it is installed:
```bash
lk --version
```

```bash
lk cloud auth
lk agent create
```

## Adding MCP Support
```bash
uv add "livekit-agents[mcp]"
```
