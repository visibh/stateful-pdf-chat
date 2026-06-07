"""
Temporal Workflow and Activity for chat agent.

Key design decisions:
- ResearchChatWorkflow: durable state machine holding the full chat history
  in workflow memory. It survives worker restarts via Temporal event replay.
- answer_question activity: all I/O (Qdrant vector search + LLM call)
  lives here. It is auto traced by LangSmith via wrap_openai().
- Human sends messages via workflow.signal and retrieves history via workflow.query.
"""

from __future__ import annotations

import os
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    # These imports have side effects (env reads, etc.) so must be passed through
    from dotenv import load_dotenv
    from deep_research.models import AnswerRequest, AnswerResult, ChatTurn


# Activity
@activity.defn
async def answer_question(request_dict: dict) -> dict:
    """
    Retrieve relevant chunks from Qdrant and call LLM to generate an answer.

    All LLM calls are made through an openai.AsyncOpenAI client wrapped with
    langsmith.wrappers.wrap_openai, so every call appears as a child span in
    LangSmith under the parent trace created by the LangSmithPlugin interceptor.
    """
    load_dotenv()

    # Sanitize LangSmith environment variables to prevent UnicodeEncodeError
    for env_key in ["LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"]:
        if val := os.getenv(env_key):
            sanitized = val.replace("—", "-").replace("–", "-")
            sanitized = "".join(c for c in sanitized if ord(c) < 128).strip()
            os.environ[env_key] = sanitized

    # Activities run in worker process
    from openai import AsyncOpenAI
    from qdrant_client import AsyncQdrantClient
    from langsmith.wrappers import wrap_openai

    request = AnswerRequest(**request_dict)

    qdrant_path = os.getenv("QDRANT_PATH", "./local_research_db")
    collection = os.getenv("DB_COLLECTION", "research_papers")
    embed_model = os.getenv("EMBED_MODEL", "baai/bge-m3")
    chat_model = os.getenv("CHAT_MODEL")
    if not chat_model or "deepseek" in chat_model:
        chat_model = "nvidia/nemotron-3-ultra-550b-a55b"

    # 1. Embed the query with Nvidia NIM
    # NOTE: wrap_openai only supports chat completions. Currently Nvidia client is unwrapped
    nvidia_client = AsyncOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        timeout=600.0,
    )

    embed_response = await nvidia_client.embeddings.create(
        model=embed_model,
        input=request.query,
        encoding_format="float",
    )
    query_vector = embed_response.data[0].embedding

    # 2. Search Qdrant for top 5 relevant chunks. Configurable top k results
    qdrant = AsyncQdrantClient(path=qdrant_path)
    query_response = await qdrant.query_points(
        collection_name=collection,
        query=query_vector,
        limit=5,
    )
    search_results = query_response.points
    await qdrant.close()

    retrieved_chunks: list[str] = []
    sources: list[str] = []
    for hit in search_results:
        payload = hit.payload or {}
        text = payload.get("text", "")
        source = (
            f"{payload.get('filename', 'unknown')} — page {payload.get('page', '?')}"
        )
        retrieved_chunks.append(f"[{source}]\n{text}")
        sources.append(source)

    context = (
        "\n\n---\n\n".join(retrieved_chunks)
        if retrieved_chunks
        else "No relevant context found."
    )

    # 3. Build the prompt
    system_prompt = (
        "You are an expert research assistant helping academics and researchers understand papers. "
        "Answer the user's question using ONLY the provided context from the retrieved chunks. "
        "Be precise, cite specific sections, and acknowledge uncertainty when the context is insufficient. "
        "Format your response in clear markdown."
    )

    history_msgs = []
    for turn in request.history[-6:]:  # last 3 exchanges = 6 turns max context
        history_msgs.append({"role": turn.role, "content": turn.content})

    user_msg = (
        f"Context from the research papers:\n\n{context}\n\n"
        f"---\n\nQuestion: {request.query}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *history_msgs,
        {"role": "user", "content": user_msg},
    ]

    # 4. Call Nemotron via Nvidia Integrate API
    nvidia_chat_raw = AsyncOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY"),
        timeout=600.0,
    )
    nvidia_chat_client = wrap_openai(nvidia_chat_raw)  # LangSmith traces this

    # Temperature is 0.3 for less hallucination. Max tokens is 2048.
    completion = await nvidia_chat_client.chat.completions.create(
        model=chat_model,
        messages=messages,
        max_tokens=2048,
        temperature=0.3,
    )

    answer = completion.choices[0].message.content or "I could not generate an answer."
    return AnswerResult(answer=answer, sources=sources).model_dump()


# Workflow


@workflow.defn
class ResearchChatWorkflow:
    """
    Durable stateful chat workflow.

    The chat history lives in this workflow's memory. Temporal replays the
    event history on restart, so the state is never lost even if the worker
    crashes mid-execution.

    Signals:   receive_message(user_msg: str) — send a new user message
    Queries:   get_history() -> list[dict] — retrieve the full chat history
    """

    def __init__(self) -> None:
        self._history: list[dict] = []
        self._pending: list[str] = []

    @workflow.signal
    def receive_message(self, user_msg: str) -> None:
        """
        Operator/user sends this signal with their message.
        """
        self._pending.append(user_msg)

    @workflow.query
    def get_history(self) -> list[dict]:
        """
        Return the complete conversation history.
        """
        return self._history

    @workflow.run
    async def run(self, session_id: str) -> None:
        """
        Main loop: wait for a message signal, call the answer activity,
        append both turns to history, repeat indefinitely.
        """
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=2),
            maximum_interval=timedelta(seconds=30),
            maximum_attempts=3,
        )

        while True:
            # Block until at least one pending message arrives
            await workflow.wait_condition(lambda: len(self._pending) > 0)
            user_msg = self._pending.pop(0)

            # Record the user turn
            self._history.append({"role": "user", "content": user_msg, "sources": []})

            # Build the request. Exclude the just added user turn from history
            # so the activity sees clean prior context
            prior_history = [ChatTurn(**t) for t in self._history[:-1]]
            request = AnswerRequest(
                query=user_msg,
                history=prior_history,
                session_id=session_id,
            )

            result_dict = await workflow.execute_activity(
                answer_question,
                args=[request.model_dump(mode="json")],
                start_to_close_timeout=timedelta(seconds=600),
                retry_policy=retry_policy,
            )

            result = AnswerResult(**result_dict)
            self._history.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": result.sources,
                }
            )
