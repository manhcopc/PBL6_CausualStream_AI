from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.errors import ServiceError
from app.schemas.contracts import HealthResponse, SessionMode, SessionStatus
from app.schemas.streaming import (
    FrameMessage,
    ModelListItem,
    ModelListResponse,
    ResetStreamResponse,
    StreamConfig,
    StreamConfigUpdate,
    StreamErrorMessage,
)

router = APIRouter()


def _managers():
    from app.main import manager, streaming_manager

    return manager, streaming_manager


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/v1/models", response_model=ModelListResponse)
async def models() -> ModelListResponse:
    manager, _ = _managers()
    info = manager.provider.model_info()
    return ModelListResponse(
        items=[
            ModelListItem(
                model_id=info.model_version,
                provider_type=info.provider_type,
                streaming_ready=SessionMode.LIVE in info.supported_modes,
                is_mock=info.provider_type == "MOCK",
            )
        ]
    )


@router.get("/v1/config", response_model=StreamConfig)
async def get_config() -> StreamConfig:
    _, streaming_manager = _managers()
    return streaming_manager.config


@router.put("/v1/config", response_model=StreamConfig)
async def update_config(payload: StreamConfigUpdate) -> StreamConfig:
    _, streaming_manager = _managers()
    return streaming_manager.update_config(payload.model_dump(exclude_unset=True))


@router.post("/v1/streams/{stream_id}/reset", response_model=ResetStreamResponse)
async def reset_stream(stream_id: str) -> ResetStreamResponse:
    _, streaming_manager = _managers()
    streaming_manager.reset(stream_id)
    return ResetStreamResponse(stream_id=stream_id)


@router.websocket("/v1/streams/{stream_id}")
async def stream_frames(websocket: WebSocket, stream_id: str) -> None:
    manager, streaming_manager = _managers()
    await websocket.accept()
    try:
        state = manager.get(stream_id)
        if state.mode != SessionMode.LIVE:
            raise ServiceError("INVALID_SESSION_MODE", "Only LIVE sessions can stream frames.", 409)
        while True:
            state = manager.get(stream_id)
            if state.status != SessionStatus.ACTIVE:
                raise ServiceError("SESSION_NOT_ACTIVE", "Inference session is not active.", 409)
            payload = await websocket.receive_json()
            message = FrameMessage.model_validate(payload)
            if message.stream_id != stream_id:
                raise ServiceError(
                    "STREAM_ID_MISMATCH",
                    "Message stream_id does not match the WebSocket stream.",
                    409,
                )
            response = await streaming_manager.process_frame(message)
            await websocket.send_json(response.model_dump(mode="json"))
    except WebSocketDisconnect:
        return
    except ValidationError as exc:
        await websocket.send_json(
            StreamErrorMessage(
                code="INVALID_MESSAGE",
                message="Streaming message does not match contract v1.",
                details={"errors": exc.errors(include_url=False)},
            ).model_dump(mode="json")
        )
        await websocket.close(code=4400)
    except ServiceError as exc:
        await websocket.send_json(
            StreamErrorMessage(
                code=exc.code,
                message=exc.message,
                details=exc.details if isinstance(exc.details, dict) else None,
            ).model_dump(mode="json")
        )
        await websocket.close(code=4409 if exc.status_code == 409 else 4503)
