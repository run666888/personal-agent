# My Personal Agent

This project includes both a command-line agent and a desktop window app built with the OpenAI Python SDK and the Responses API.

## Features

- Defaults to `gpt-5.4`
- Faster default responses with `low` reasoning, plus smarter presets you can switch to
- Multi-turn conversation with saved response state
- Terminal agent and desktop window app
- Optional OpenAI web search
- Optional local tools for file reads, file writes, directory creation, and PowerShell commands
- Safer local tools with workspace limits and approval prompts

## Files

- `agent_core.py` - shared agent logic
- `agent.py` - command-line app
- `agent_app.py` - desktop window app
- `system_prompt.txt` - editable agent behavior prompt
- `.env.example` - environment variable template
- `.agent_session.json` - command-line conversation state
- `build_app.ps1` - Windows packaging script

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure

Use an environment variable:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

Optional environment variables:

```powershell
$env:OPENAI_AGENT_MODEL="gpt-5.4"
$env:OPENAI_REASONING_EFFORT="low"
```

You can also copy the template:

```powershell
Copy-Item .env.example .env
```

## Run the CLI Agent

```powershell
python agent.py
```

Start fresh and ignore the last saved conversation:

```powershell
python agent.py --fresh
```

Start with web search enabled:

```powershell
python agent.py --web-search
```

Choose a different reasoning effort:

```powershell
python agent.py --reasoning-effort high
```

Enable local file and shell tools inside the current workspace:

```powershell
python agent.py --local-tools
```

Set a workspace and auto-approve tool actions:

```powershell
python agent.py --local-tools --auto-confirm-tools --workspace D:\selfagent
```

## Run the Desktop App

```powershell
python agent_app.py
```

The desktop app lets you:

- paste an API key or use `OPENAI_API_KEY`
- choose a preset like `Balanced`, `Smart`, `Deep`, or `Fast`
- enable web search
- enable local tools
- choose the workspace the local tools are allowed to access

## Build a Standalone EXE

Install the build dependency and package the app:

```powershell
pip install -r requirements-build.txt
powershell -ExecutionPolicy Bypass -File .\build_app.ps1
```

After the build finishes, the standalone Windows app will be in:

```text
dist\PersonalAgentApp.exe
```

## CLI Commands

- `/help`
- `/status`
- `/reset`
- `/model gpt-5.4`
- `/reasoning medium`
- `/web on`
- `/web off`
- `/tools on`
- `/tools off`
- `/quit`
