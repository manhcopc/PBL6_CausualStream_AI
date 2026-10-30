from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.schemas.contracts import (
    CompletionResponse,
    CreateInferenceSessionRequest,
    HealthResponse,
    InferenceSessionResponse,
    ModelInfoResponse,
    PredictionResponse,
    ReadinessResponse,
    SystemStatusResponse,
)
from app.sessions.manager import InferenceSessionManager

router = APIRouter()


def get_manager() -> InferenceSessionManager:
    from app.main import manager

    return manager


@router.post("/internal/v1/sessions", response_model=InferenceSessionResponse)
async def create_session(
    payload: CreateInferenceSessionRequest,
    session_manager: InferenceSessionManager = Depends(get_manager),
) -> InferenceSessionResponse:
    return await session_manager.create(payload.session_id, payload.mode, payload.scenario_id)


@router.post("/internal/v1/sessions/{session_id}/infer", response_model=PredictionResponse)
async def infer_frame(
    session_id: str,
    frame_index: int = Form(..., ge=0),
    timestamp_ms: int = Form(..., ge=0),
    frame: UploadFile = File(...),
    session_manager: InferenceSessionManager = Depends(get_manager),
) -> PredictionResponse:
    payload = await frame.read()
    return await session_manager.infer(session_id, payload, frame_index, timestamp_ms)


@router.post("/internal/v1/sessions/{session_id}/advance", response_model=PredictionResponse)
async def advance_demo(
    session_id: str,
    session_manager: InferenceSessionManager = Depends(get_manager),
) -> PredictionResponse:
    return await session_manager.advance(session_id)


@router.post("/internal/v1/sessions/{session_id}/pause", response_model=InferenceSessionResponse)
async def pause_session(
    session_id: str,
    session_manager: InferenceSessionManager = Depends(get_manager),
) -> InferenceSessionResponse:
    return await session_manager.pause(session_id)


@router.post("/internal/v1/sessions/{session_id}/resume", response_model=InferenceSessionResponse)
async def resume_session(
    session_id: str,
    session_manager: InferenceSessionManager = Depends(get_manager),
) -> InferenceSessionResponse:
    return await session_manager.resume(session_id)


@router.delete("/internal/v1/sessions/{session_id}", response_model=CompletionResponse)
async def delete_session(
    session_id: str,
    session_manager: InferenceSessionManager = Depends(get_manager),
) -> CompletionResponse:
    from app.main import streaming_manager

    streaming_manager.reset(session_id)
    return await session_manager.delete(session_id)


@router.get("/internal/v1/models/current", response_model=ModelInfoResponse)
async def model_info(session_manager: InferenceSessionManager = Depends(get_manager)) -> ModelInfoResponse:
    return session_manager.provider.model_info()


@router.get("/internal/v1/system/status", response_model=SystemStatusResponse)
async def system_status(session_manager: InferenceSessionManager = Depends(get_manager)) -> SystemStatusResponse:
    info = session_manager.provider.model_info()
    return SystemStatusResponse(
        status="ONLINE",
        provider=info.provider,
        model_version=info.model_version,
        real_model_ready=info.real_model_ready,
        active_inference_sessions=session_manager.active_count,
        details={"mock_available": info.provider_type == "MOCK"},
    )


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=ReadinessResponse)
async def readyz(session_manager: InferenceSessionManager = Depends(get_manager)) -> ReadinessResponse:
    info = session_manager.provider.model_info()
    return ReadinessResponse(status="ready", provider=info.provider, real_model_ready=info.real_model_ready)
