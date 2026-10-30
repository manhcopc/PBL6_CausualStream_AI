import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.core.config import Settings
from app.core.errors import ServiceError
from app.decision.policy import DecisionPolicy
from app.inference.providers import InferenceProvider
from app.inference.scenarios import SCENARIOS
from app.preprocessing.validation import validate_image_bytes
from app.schemas.contracts import (
    CompletionResponse,
    InferenceSessionResponse,
    PredictionResponse,
    SessionMode,
    SessionStatus,
)


@dataclass
class InferenceSessionState:
    session_id: str
    mode: SessionMode
    status: SessionStatus
    scenario_id: str | None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_accessed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_frame_index: int = -1
    last_timestamp_ms: int = -1
    cursor: int = 0
    policy: DecisionPolicy = field(default_factory=DecisionPolicy)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class InferenceSessionManager:
    def __init__(self, provider: InferenceProvider, settings: Settings) -> None:
        self.provider = provider
        self.settings = settings
        self._sessions: dict[str, InferenceSessionState] = {}
        self._global_lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    async def create(self, session_id: str, mode: SessionMode, scenario_id: str | None) -> InferenceSessionResponse:
        async with self._global_lock:
            self.cleanup_expired()
            if len(self._sessions) >= self.settings.max_active_sessions:
                raise ServiceError("MAX_SESSIONS_REACHED", "Maximum active inference sessions reached.", 503)
            self.provider.initialize_session(session_id, mode, scenario_id)
            state = InferenceSessionState(
                session_id=session_id,
                mode=mode,
                status=SessionStatus.ACTIVE,
                scenario_id=scenario_id,
            )
            self._sessions[session_id] = state
            return InferenceSessionResponse(
                session_id=session_id,
                mode=mode,
                status=state.status,
                model_version=self.provider.model_info().model_version,
                is_demo=mode == SessionMode.DEMO,
            )

    def get(self, session_id: str) -> InferenceSessionState:
        state = self._sessions.get(session_id)
        if state is None:
            raise ServiceError("SESSION_NOT_FOUND", "Inference session was not found.", 404)
        state.last_accessed_at = datetime.now(UTC)
        return state

    async def advance(self, session_id: str) -> PredictionResponse:
        state = self.get(session_id)
        async with state.lock:
            if state.status != SessionStatus.ACTIVE:
                raise ServiceError("SESSION_NOT_ACTIVE", "Inference session is not active.", 409)
            if state.mode != SessionMode.DEMO or state.scenario_id is None:
                raise ServiceError("INVALID_SESSION_MODE", "Only DEMO sessions can be advanced.", 409)
            if state.cursor >= len(SCENARIOS[state.scenario_id]):
                raise ServiceError("SCENARIO_COMPLETED", "Demo scenario has no remaining events.", 409)
            previous = state.policy.previous_for(SCENARIOS[state.scenario_id][state.cursor].decision)
            prediction = await self.provider.advance_demo(session_id, state.scenario_id, state.cursor, previous)
            state.cursor += 1
            state.last_frame_index = prediction.frame_index
            state.last_timestamp_ms = prediction.timestamp_ms
            state.policy.record(prediction.decision, prediction.prediction_id)
            return prediction

    async def infer(self, session_id: str, frame: bytes, frame_index: int, timestamp_ms: int) -> PredictionResponse:
        state = self.get(session_id)
        async with state.lock:
            if state.status != SessionStatus.ACTIVE:
                raise ServiceError("SESSION_NOT_ACTIVE", "Inference session is not active.", 409)
            if state.mode != SessionMode.LIVE:
                raise ServiceError("INVALID_SESSION_MODE", "Only LIVE sessions accept frames.", 409)
            if frame_index <= state.last_frame_index or timestamp_ms <= state.last_timestamp_ms:
                raise ServiceError("FRAME_OUT_OF_ORDER", "Frame index and timestamp must increase.", 409)
            validate_image_bytes(frame, self.settings.max_frame_size_mb)
            prediction = await self.provider.infer(session_id, frame, frame_index, timestamp_ms)
            state.last_frame_index = frame_index
            state.last_timestamp_ms = timestamp_ms
            state.policy.record(prediction.decision, prediction.prediction_id)
            return prediction

    async def pause(self, session_id: str) -> InferenceSessionResponse:
        state = self.get(session_id)
        async with state.lock:
            if state.status != SessionStatus.ACTIVE:
                raise ServiceError("INVALID_STATE_TRANSITION", "Only ACTIVE sessions can be paused.", 409)
            state.status = SessionStatus.PAUSED
            return self._response(state)

    async def resume(self, session_id: str) -> InferenceSessionResponse:
        state = self.get(session_id)
        async with state.lock:
            if state.status != SessionStatus.PAUSED:
                raise ServiceError("INVALID_STATE_TRANSITION", "Only PAUSED sessions can be resumed.", 409)
            state.status = SessionStatus.ACTIVE
            return self._response(state)

    async def delete(self, session_id: str) -> CompletionResponse:
        state = self._sessions.pop(session_id, None)
        if state is not None:
            self.provider.release_session(session_id)
        return CompletionResponse(session_id=session_id, message="Inference session released.")

    def cleanup_expired(self) -> int:
        cutoff = datetime.now(UTC) - timedelta(seconds=self.settings.session_timeout_seconds)
        expired = [
            session_id for session_id, state in self._sessions.items()
            if state.last_accessed_at < cutoff
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
            self.provider.release_session(session_id)
        return len(expired)

    def _response(self, state: InferenceSessionState) -> InferenceSessionResponse:
        return InferenceSessionResponse(
            session_id=state.session_id,
            mode=state.mode,
            status=state.status,
            model_version=self.provider.model_info().model_version,
            is_demo=state.mode == SessionMode.DEMO,
        )
