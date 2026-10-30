from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FrameMessage(BaseModel):
    type: Literal["frame"] = "frame"
    stream_id: str = Field(min_length=1, max_length=128)
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)
    image_b64: str = Field(min_length=1)


class CaptionMessage(BaseModel):
    type: Literal["caption"] = "caption"
    stream_id: str
    event_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    caption: str = Field(min_length=1)
    model_id: str
    inference_ms: float | None = Field(default=None, ge=0)
    processing_delay_ms: float | None = Field(default=None, ge=0)
    is_mock: bool = False

    @model_validator(mode="after")
    def validate_interval(self) -> "CaptionMessage":
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must be less than or equal to end_ms")
        return self


class NoEventMessage(BaseModel):
    type: Literal["no_event"] = "no_event"
    stream_id: str
    frame_index: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)


class StreamErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
    details: dict[str, object] | None = None


class StreamConfig(BaseModel):
    memory_enabled: bool = True
    memory_k: int = Field(default=2, ge=0, le=16)
    sample_frames: int = Field(default=4, ge=1, le=32)
    mock_latency_ms: int = Field(default=0, ge=0, le=30_000)
    mock_fail_after_frames: int | None = Field(default=None, ge=1)


class StreamConfigUpdate(BaseModel):
    memory_enabled: bool | None = None
    memory_k: int | None = Field(default=None, ge=0, le=16)
    sample_frames: int | None = Field(default=None, ge=1, le=32)
    mock_latency_ms: int | None = Field(default=None, ge=0, le=30_000)
    mock_fail_after_frames: int | None = Field(default=None, ge=1)


class ModelListItem(BaseModel):
    model_id: str
    provider_type: str
    streaming_ready: bool
    is_mock: bool


class ModelListResponse(BaseModel):
    items: list[ModelListItem]


class ResetStreamResponse(BaseModel):
    stream_id: str
    reset: bool = True
