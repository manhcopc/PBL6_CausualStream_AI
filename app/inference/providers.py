import asyncio
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.core.config import Settings
from app.core.errors import ServiceError
from app.inference.scenarios import SCENARIOS
from app.schemas.contracts import Decision, ModelInfoResponse, PredictionResponse, SessionMode


class InferenceProvider(Protocol):
    def initialize_session(self, session_id: str, mode: SessionMode, scenario_id: str | None) -> None:
        ...

    async def advance_demo(self, session_id: str, scenario_id: str, cursor: int, previous_prediction_id: str | None) -> PredictionResponse:
        ...

    async def infer(self, session_id: str, frame: bytes, frame_index: int, timestamp_ms: int) -> PredictionResponse:
        ...

    def release_session(self, session_id: str) -> None:
        ...

    def model_info(self) -> ModelInfoResponse:
        ...


class MockInferenceProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def initialize_session(self, session_id: str, mode: SessionMode, scenario_id: str | None) -> None:
        if mode == SessionMode.DEMO:
            if scenario_id not in SCENARIOS:
                raise ServiceError("INVALID_SCENARIO", "Demo scenario is not available.", 422)
            return
        if mode == SessionMode.LIVE:
            return
        raise ServiceError("INVALID_SESSION_MODE", "Session mode is not supported.", 409)

    async def advance_demo(self, session_id: str, scenario_id: str, cursor: int, previous_prediction_id: str | None) -> PredictionResponse:
        scenario = SCENARIOS[scenario_id]
        if cursor >= len(scenario):
            raise ServiceError("SCENARIO_COMPLETED", "Demo scenario has no remaining events.", 409)
        event = scenario[cursor]
        prediction_id = str(uuid4())
        return PredictionResponse(
            prediction_id=prediction_id,
            session_id=session_id,
            frame_index=event.frame_index,
            timestamp_ms=event.timestamp_ms,
            action_id=event.action_id,
            action_label=event.action_label,
            confidence=event.confidence,
            decision=event.decision,
            message=event.message,
            previous_prediction_id=previous_prediction_id if event.decision == Decision.UPDATE else None,
            model_version=self.settings.model_version,
            inference_time_ms=None,
            is_demo=True,
        )

    async def infer(self, session_id: str, frame: bytes, frame_index: int, timestamp_ms: int) -> PredictionResponse:
        raise ServiceError("MODEL_NOT_READY", "Real inference model is not available.", 503)

    def release_session(self, session_id: str) -> None:
        return None

    def model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            provider="mock",
            provider_type="MOCK",
            model_name="Deterministic Streaming Mock",
            model_version=self.settings.model_version,
            real_model_ready=False,
            supported_modes=[SessionMode.DEMO, SessionMode.LIVE],
        )


class RealInferenceProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.runtime = None
        self.ready = False

    def load(self) -> None:
        from app.inference.full075 import Full075Runtime

        if not self.settings.model_checkpoint_path:
            raise RuntimeError("MODEL_CHECKPOINT_PATH is required for real inference.")
        runtime = Full075Runtime(Path(self.settings.model_checkpoint_path))
        runtime.load()
        self.runtime = runtime
        self.ready = True

    def initialize_session(self, session_id: str, mode: SessionMode, scenario_id: str | None) -> None:
        if not self.ready:
            raise ServiceError("MODEL_NOT_READY", "Real inference model is not available.", 503)
        if mode != SessionMode.LIVE:
            raise ServiceError("INVALID_SESSION_MODE", "Real provider only supports LIVE.", 409)

    async def caption(self, frames: list[bytes], timestamps_ms: list[int]) -> tuple[str, float]:
        if not self.ready or self.runtime is None:
            raise ServiceError("MODEL_NOT_READY", "Real inference model is not available.", 503)
        try:
            return await asyncio.to_thread(self.runtime.generate, frames, timestamps_ms)
        except Exception as exc:
            raise ServiceError("INFERENCE_FAILED", "Full075 inference failed.", 503) from exc

    async def advance_demo(self, session_id: str, scenario_id: str, cursor: int, previous_prediction_id: str | None) -> PredictionResponse:
        raise ServiceError("INVALID_SESSION_MODE", "Real provider does not serve demo scenarios.", 409)

    async def infer(self, session_id: str, frame: bytes, frame_index: int, timestamp_ms: int) -> PredictionResponse:
        raise ServiceError("MODEL_NOT_READY", "Real inference adapter is not implemented yet.", 503)

    def release_session(self, session_id: str) -> None:
        return None

    def model_info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            provider="real",
            provider_type="REAL",
            model_name="Full075 Qwen3-VL-2B",
            model_version=self.settings.model_version,
            real_model_ready=self.ready,
            supported_modes=[SessionMode.LIVE] if self.ready else [],
        )


def build_provider(settings: Settings) -> InferenceProvider:
    if settings.inference_provider.lower() == "real":
        return RealInferenceProvider(settings)
    return MockInferenceProvider(settings)

