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
