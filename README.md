# Langworld

LangChain + LangGraph learning project with Ollama and Langfuse tracing.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Ollama](https://ollama.ai/) running locally
- Docker (for Langfuse)

## Setup

1. **Clone and install dependencies**
   ```bash
   git clone https://github.com/jeffersonest/langworld.git
   cd langworld
   uv sync
   ```

2. **Pull a model in Ollama**
   ```bash
   ollama pull llama3.2
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your Langfuse keys (optional for tracing).

4. **Start Langfuse (optional)**
   ```bash
   docker compose up -d
   ```
   Access at http://localhost:3000

## Run

```bash
uv run main.py
```

## Features

- Tool calling with LangChain + Ollama
- Tracing with Langfuse
- Example tools: `calculate`, `get_dollar_rate`
