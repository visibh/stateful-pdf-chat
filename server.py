"""
FastAPI Server

Endpoints:
  POST /chat/start -> start a new ResearchChatWorkflow
  POST /chat/{session_id}/message -> send a message (signal), long-poll for answer
  GET  /chat/{session_id}/history -> retrieve full conversation history
  GET  / -> serve the chat UI

Run:
    uvicorn server:app --reload
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from temporalio.client import Client
from temporalio.exceptions import WorkflowAlreadyStartedError

from deep_research.models import (
    SendMessageRequest,
    SendMessageResponse,
    StartSessionRequest,
    StartSessionResponse,
)
from deep_research.workflows import ResearchChatWorkflow

load_dotenv()

# Sanitize LangSmith environment variables to prevent UnicodeEncodeError
for env_key in ["LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"]:
    if val := os.getenv(env_key):
        sanitized = val.replace("—", "-").replace("–", "-")
        sanitized = "".join(c for c in sanitized if ord(c) < 128).strip()
        os.environ[env_key] = sanitized

TEMPORAL_HOST = os.getenv("TEMPORAL_HOST", "localhost:7233")
TASK_QUEUE = "research-chat-queue"
TEMPLATES_DIR = Path(__file__).parent / "deep_research" / "templates"

app = FastAPI(
    title="Stateful PDF Chat",
    description="Durable RAG agent for academic papers",
    version="0.1.0",
)

# Temporal client (shared, initialised at startup)
_temporal_client: Client | None = None


async def get_temporal_client() -> Client:
    global _temporal_client
    if _temporal_client is None:
        from langsmith import Client as LangsmithClient
        from temporalio.contrib.langsmith import LangSmithInterceptor
        
        langsmith_project = os.getenv("LANGSMITH_PROJECT", "stateful-pdf-chat")
        langsmith_client = LangsmithClient(timeout_ms=(2000, 2000))
        langsmith_interceptor = LangSmithInterceptor(
            client=langsmith_client,
            project_name=langsmith_project,
            add_temporal_runs=True,
        )
        _temporal_client = await Client.connect(
            TEMPORAL_HOST,
            interceptors=[langsmith_interceptor],
        )
    return _temporal_client


@app.on_event("startup")
async def startup() -> None:
    await get_temporal_client()


# Routes


@app.get("/", include_in_schema=False)
async def serve_ui() -> FileResponse:
    return FileResponse(TEMPLATES_DIR / "index.html")


@app.get("/chats")
async def list_chats() -> JSONResponse:
    """List all previous stateful chat sessions from Temporal."""
    client = await get_temporal_client()

    try:
        workflow_runs = []
        async for workflow_info in client.list_workflows(
            query="WorkflowType = 'ResearchChatWorkflow'"
        ):
            workflow_runs.append(workflow_info)

        async def fetch_session_details(w_info):
            if not w_info.id.startswith("pdf-chat-"):
                return None
            session_id = w_info.id[len("pdf-chat-") :]

            # Default title
            title = f"Chat Session ({w_info.start_time.strftime('%b %d, %H:%M')})"
            try:
                handle = client.get_workflow_handle(w_info.id)
                history = await handle.query(ResearchChatWorkflow.get_history)
                if history and len(history) > 0:
                    first_msg = history[0]["content"]
                    if len(first_msg) > 30:
                        title = first_msg[:30] + "..."
                    else:
                        title = first_msg
            except Exception:
                pass

            return {
                "session_id": session_id,
                "title": title,
                "start_time": w_info.start_time.isoformat(),
                "status": int(w_info.status),
            }

        tasks = [fetch_session_details(w) for w in workflow_runs]
        results = await asyncio.gather(*tasks)
        sessions = [r for r in results if r is not None]

        sessions.sort(key=lambda x: x["start_time"], reverse=True)
        return JSONResponse({"sessions": sessions})

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/start", response_model=StartSessionResponse)
async def start_session(
    body: StartSessionRequest = StartSessionRequest(),
) -> StartSessionResponse:
    """
    Start a new durable chat session.
    Returns a session_id that must be used for subsequent messages.
    """
    client = await get_temporal_client()
    session_id = str(uuid.uuid4())
    workflow_id = f"pdf-chat-{session_id}"

    try:
        await client.start_workflow(
            ResearchChatWorkflow.run,
            args=[session_id],
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
    except WorkflowAlreadyStartedError:
        pass

    return StartSessionResponse(session_id=session_id, workflow_id=workflow_id)


@app.post("/chat/{session_id}/message", response_model=SendMessageResponse)
async def send_message(
    session_id: str, body: SendMessageRequest
) -> SendMessageResponse:
    """
    Send a user message to the workflow via signal, then long poll until
    the assistant response appears in the workflow query result.
    Time out is 90 seconds.
    """
    client = await get_temporal_client()
    workflow_id = f"pdf-chat-{session_id}"

    try:
        handle = client.get_workflow_handle(workflow_id)

        # Record current history length before sending the signal
        history_before: list[dict] = await handle.query(
            ResearchChatWorkflow.get_history
        )
        turns_before = len(history_before)

        # Send the user message signal
        await handle.signal(ResearchChatWorkflow.receive_message, body.message)

        # Long-poll until the assistant reply appears (2 new turns: user + assistant)
        deadline = 300.0
        poll_interval = 0.5
        elapsed = 0.0

        while elapsed < deadline:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            history: list[dict] = await handle.query(ResearchChatWorkflow.get_history)
            if len(history) >= turns_before + 2:
                last = history[-1]
                return SendMessageResponse(
                    answer=last["content"],
                    sources=last.get("sources", []),
                )

        raise HTTPException(
            status_code=504, detail="Timed out waiting for assistant response"
        )

    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/chat/{session_id}/history")
async def get_history(session_id: str) -> JSONResponse:
    """Return the full conversation history for a session."""
    client = await get_temporal_client()
    workflow_id = f"pdf-chat-{session_id}"

    try:
        handle = client.get_workflow_handle(workflow_id)
        history: list[dict] = await handle.query(ResearchChatWorkflow.get_history)
        return JSONResponse({"session_id": session_id, "history": history})
    except Exception as exc:
        raise HTTPException(
            status_code=404, detail=f"Session not found: {exc}"
        ) from exc
