# Stateful PDF Chat

This repo is a project I built to experiment with a **stateful RAG (Retrieval-Augmented Generation) chat assistant**. It lets you upload PDF papers, indexes them, and lets you ask questions about them over a web interface.

The key thing here is that the chat history is managed by **Temporal**. That means the chat session state is durable, even if the worker crashes, the server restarts, or if the laptop sleeps during the conversation, the chat history and workflow state are preserved and will resume exactly where it left off.

---

## How It Actually Works Under the Hood

1. **Ingestion (`ingest.py`)**:
   - Reads PDFs from `./research_papers`.
   - Uses `PyMuPDF` to pull out plain text. It also crops out tables and charts.
   - Feeds those images to a VLM (`nvidia/llama-3.1-nemotron-nano-vl-8b-v1`) to turn them into clean Markdown.
   - Combines the page text + VLM markdown, embeds it using a local BGE-M3 model, and stores it in Qdrant (stored locally on disk).

2. **Durable Chat Workflow (`deep_research/workflows.py`)**:
   - Built as a Temporal Workflow. The workflow holds the conversation history in its memory.
   - When a user sends a message, it signals the workflow.
   - The workflow runs an Activity (`answer_question`) that queries Qdrant for context, constructs a prompt, and calls the Nvidia Integrate completions API (specifically the `nvidia/nemotron-3-ultra-550b-a55b` model).
   - If the LLM call takes long or fails due to network issues, Temporal automatically retries it and handles timeouts.

3. **FastAPI Backend and UI (`server.py` & `deep_research/templates/index.html`)**:
   - A minimalist, standard web UI (plain CSS, browser fonts, rectangular boxes).
   - Long-polls the FastAPI backend to wait for workflow updates.
   - Communicates with a local Temporal cluster.

---

## Setup and Running it

Make sure you have `temporal` CLI installed on your machine (`brew install temporal` on macOS).

### 1. Keys and Environment
Copy the example env file and add your keys:
```bash
cp .env.example .env
```
Inside `.env`, make sure to add:
- `NVIDIA_API_KEY` (for NIM completions and embeddings)
- `OPENROUTER_API_KEY` (if using OpenRouter)
- `LANGSMITH_API_KEY` (if you want to trace LLM calls with LangSmith)

### 2. Ingest the papers
Throw some PDFs into the `research_papers` folder, then run:
```bash
uv run python ingest.py
```

### 3. Start the Temporal Server
Open a terminal and launch Temporal locally:
```bash
temporal server start-dev --db-filename ./local_research_db/temporal_history.db
```
*(Passing the `--db-filename` ensures the chat history database actually persists when you close the server).*

### 4. Start the Worker
In another terminal, start the Python worker that runs the workflow and activity logic:
```bash
uv run python worker.py
```

### 5. Start the FastAPI Web Server
In a third terminal, start the web server:
```bash
uv run uvicorn server:app --reload
```

Now open **http://localhost:8000** in your browser.

---

## Observability

- **Temporal UI**: Open **http://localhost:8233** to see the active workflows, state histories, inputs, outputs, and stack traces.
- **LangSmith Tracing**: If `LANGSMITH_TRACING=true` is set, every query and LLM completion is logged to LangSmith. The Python worker uses `temporalio.contrib.langsmith.LangSmithInterceptor` to link the LangSmith traces directly to the Temporal `workflow_id`. (We set a fast 2 second timeout on LangSmith API requests so that if LangSmith is unreachable, it fails fast instead of hanging your chat loop).
