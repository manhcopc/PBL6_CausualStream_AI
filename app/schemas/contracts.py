from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SessionMode(StrEnum):
    DEMO = "DEMO"
    LIVE = "LIVE"


class SessionStatus(StrEnum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"


class Decision(StrEnum):
    WAIT = "WAIT"
    SPEAK = "SPEAK"
    UPDATE = "UPDATE"


class CreateInferenceSessionRequest(BaseModel):
    session_id: str
    mode: SessionMode
    scenario_id: str | None = None


class InferenceSessionResponse(BaseModel):
    session_id: str
    mode: SessionMode
    status: SessionStatus
    model_version: str
    is_demo: bool


class PredictionResponse(BaseModel):
    prediction_id: str
    session_id: str
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    action_id: str
    action_label: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    decision: Decision
    message: str | None = None
    previous_prediction_id: str | None = None
    model_version: str
    inference_time_ms: float | None = None
    is_demo: bool


class CompletionResponse(BaseModel):
    completed: bool = True
    session_id: str
    message: str


class ModelInfoResponse(BaseModel):
    provider: str
    provider_type: str
    model_name: str
    model_version: str
    real_model_ready: bool
    supported_modes: list[SessionMode]


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    provider: str
    real_model_ready: bool


class SystemStatusResponse(BaseModel):
    status: str
    provider: str
    model_version: str
    real_model_ready: bool
    active_inference_sessions: int
    details: dict[str, Any] | None = None

