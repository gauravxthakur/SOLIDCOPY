## Install dependencies
```bash
uv sync
```
## Add environment variables
Create a .env file with the help of .env.example

## Download the Turn Detection Model
```bash
uv run python -m livekit.agents download-files
```

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

## License & Trademark Policy

This project is licensed under the [MIT License](LICENSE.md). You are free to copy, modify, and distribute the software.

However, the SOLIDCOPY name, logo, and branding are protected assets. 
* Allowed: You may use the name to truthfully state that your project is "a fork of SOLIDCOPY" or "based on SOLIDCOPY".
* Not Allowed: You may not use the name SOLIDCOPY as the primary name of your fork, or in any way that implies official endorsement.