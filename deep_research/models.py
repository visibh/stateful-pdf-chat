"""
Data models for the Stateful PDF Chat agent.

All models use Pydantic v2 and .model_dump(mode="json") before passing
through Temporal activities.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    """
    A single turn in the conversation.
    """

    role: str
    content: str
    sources: list[str] = Field(default_factory=list)


class AnswerRequest(BaseModel):
    """
    Input to the answer_question activity.
    """

    query: str
    history: list[ChatTurn]
    session_id: str


class AnswerResult(BaseModel):
    """
    Output from the answer_question activity.
    """

    answer: str
    sources: list[str] = Field(default_factory=list)


class StartSessionRequest(BaseModel):
    """
    Request to start a chat session.
    """

    title: str = "New Research Session"


class StartSessionResponse(BaseModel):
    session_id: str
    workflow_id: str


class SendMessageRequest(BaseModel):
    """
    Request to send a message to the chat session.
    """

    message: str


class SendMessageResponse(BaseModel):
    """
    Response to a user message.
    """

    answer: str
    sources: list[str] = Field(default_factory=list)
