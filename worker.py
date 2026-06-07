"""
Temporal Worker

Listens on the 'research-chat-queue' task queue and executes:
  - ResearchChatWorkflow
  - answer_question activity

LangSmith integration:
  The LangSmithPlugin interceptor creates a root LangSmith trace for each
  workflow execution and propagates trace context into all child activity
  runs. Combined with wrap_openai() in the activity, every *LM call
  appears as a nested span in the LangSmith UI, linked to the Temporal
  workflow_id.

Run:
    python worker.py
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from temporalio.client import Client
from temporalio.worker import Worker, UnsandboxedWorkflowRunner

from deep_research.workflows import ResearchChatWorkflow, answer_question

load_dotenv()

# Sanitize LangSmith environment variables to prevent UnicodeEncodeError
for env_key in ["LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"]:
    if val := os.getenv(env_key):
        sanitized = val.replace("—", "-").replace("–", "-")
        sanitized = "".join(c for c in sanitized if ord(c) < 128).strip()
        os.environ[env_key] = sanitized

console = Console()

TASK_QUEUE = "research-chat-queue"


async def main() -> None:
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    langsmith_project = os.getenv("LANGSMITH_PROJECT", "stateful-pdf-chat")

    console.print(
        Panel(
            f"[bold green]Stateful PDF Chat[/bold green]\n"
            f"Temporal server:  [cyan]{temporal_host}[/cyan]\n"
            f"Task queue:       [cyan]{TASK_QUEUE}[/cyan]\n"
            f"LangSmith project:[cyan]{langsmith_project}[/cyan]\n"
            f"Temporal UI:      [cyan]http://localhost:8233[/cyan]",
            title="Temporal Worker",
        )
    )

    # Configure a custom LangSmith Client with a fast 2-second timeout to prevent network hangs
    from langsmith import Client as LangsmithClient
    from temporalio.contrib.langsmith import LangSmithInterceptor

    langsmith_client = LangsmithClient(timeout_ms=(2000, 2000))
    langsmith_interceptor = LangSmithInterceptor(
        client=langsmith_client,
        project_name=langsmith_project,
        add_temporal_runs=True,
    )

    client = await Client.connect(
        temporal_host,
        interceptors=[langsmith_interceptor],
    )

    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[ResearchChatWorkflow],
        activities=[answer_question],
        interceptors=[langsmith_interceptor],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )

    console.print(
        f"\n[bold]Worker running[/bold] waiting for workflows on [cyan]{TASK_QUEUE}[/cyan]"
    )
    console.print("Press Ctrl+C to stop.\n")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
