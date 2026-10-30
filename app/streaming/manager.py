import asyncio
import base64
import binascii
from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

from app.core.config import Settings
from app.core.errors import ServiceError
from app.inference.providers import InferenceProvider, RealInferenceProvider
from app.preprocessing.validation import validate_image_bytes
from app.schemas.streaming import CaptionMessage, FrameMessage, NoEventMessage, StreamConfig


@dataclass(frozen=True)
class MockCaptionEvent:
    start_ms: int
    end_ms: int
    caption_without_memory: str
    caption_with_memory: str


MOCK_CAPTION_EVENTS = (
    MockCaptionEvent(
        400,
        2_300,
        "A person is picking up a ball.",
        "A person is picking up a ball.",
    ),
    MockCaptionEvent(
        4_100,
        8_700,
        "A person throws the ball.",
        "The same person throws the ball.",
    ),
    MockCaptionEvent(
        10_300,
        12_000,
        "A person is standing up.",
        "The person is standing up after throwing the ball.",
    ),
)


@dataclass
class StreamingState:
    stream_id: str
    last_frame_index: int = -1
    last_timestamp_ms: int = -1
    last_frame_digest: str | None = None
    last_response: CaptionMessage | NoEventMessage | None = None
    next_event_index: int = 0
    frames_received: int = 0
    caption_history: list[str] = field(default_factory=list)
    first_frame_ms: int | None = None
    last_inference_ms: int | None = None
    frames: deque[tuple[int, bytes]] = field(default_factory=lambda: deque(maxlen=16))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class StreamingManager:
    def __init__(self, settings: Settings, provider: InferenceProvider) -> None:
        self.settings = settings
        self.provider = provider
        self.config = StreamConfig(
            memory_enabled=settings.memory_enabled,
            memory_k=settings.memory_k,
            sample_frames=settings.sample_frames,
            mock_latency_ms=settings.mock_latency_ms,
            mock_fail_after_frames=settings.mock_fail_after_frames,
        )
        self._states: dict[str, StreamingState] = {}
        self._global_lock = asyncio.Lock()

    async def process_frame(
        self, message: FrameMessage
    ) -> CaptionMessage | NoEventMessage:
        state = await self._get_or_create(message.stream_id)
        async with state.lock:
            digest = sha256(message.image_b64.encode()).hexdigest()
            if (
                message.frame_index == state.last_frame_index
                and message.timestamp_ms == state.last_timestamp_ms
                and digest == state.last_frame_digest
                and state.last_response is not None
            ):
                return state.last_response
            if message.frame_index <= state.last_frame_index:
                raise ServiceError(
                    "FRAME_OUT_OF_ORDER", "Frame index must increase within a stream.", 409
                )
            if message.timestamp_ms <= state.last_timestamp_ms:
                raise ServiceError(
                    "TIMESTAMP_OUT_OF_ORDER",
                    "Frame timestamp must increase within a stream.",
                    409,
                )
            if (
                not isinstance(self.provider, RealInferenceProvider)
                and self.config.mock_fail_after_frames is not None
                and state.frames_received >= self.config.mock_fail_after_frames
            ):
                raise ServiceError(
                    "MOCK_STREAM_FAILURE",
                    "Mock stream failure was triggered by configuration.",
                    503,
                )

            frame = self._decode_frame(message.image_b64)
            validate_image_bytes(frame, self.settings.max_frame_size_mb)
            started = perf_counter()
            if not isinstance(self.provider, RealInferenceProvider) and self.config.mock_latency_ms:
                await asyncio.sleep(self.config.mock_latency_ms / 1_000)

            if isinstance(self.provider, RealInferenceProvider):
                response = await self._process_real_frame(state, message, frame)
                self._record_processed_frame(state, message, digest, response)
                return response

            event = self._next_supported_event(state, message.timestamp_ms)
            if event is None:
                response = NoEventMessage(
                    stream_id=message.stream_id,
                    frame_index=message.frame_index,
                    timestamp_ms=message.timestamp_ms,
                )
                self._record_processed_frame(state, message, digest, response)
                return response

            use_memory = self.config.memory_enabled and self.config.memory_k > 0
            caption = (
                event.caption_with_memory
                if use_memory and state.caption_history
                else event.caption_without_memory
            )
            event_index = state.next_event_index
            state.next_event_index += 1
            state.caption_history.append(caption)
            if self.config.memory_k == 0:
                state.caption_history.clear()
            else:
                state.caption_history[:] = state.caption_history[-self.config.memory_k :]

            inference_ms = round((perf_counter() - started) * 1_000, 3)
            response = CaptionMessage(
                stream_id=message.stream_id,
                event_id=str(
                    uuid5(
                        NAMESPACE_URL,
                        f"causalstream:stream:{message.stream_id}:event:{event_index}",
                    )
                ),
                start_ms=event.start_ms,
                end_ms=event.end_ms,
                caption=caption,
                model_id=self.settings.model_version,
                inference_ms=inference_ms,
                processing_delay_ms=float(self.config.mock_latency_ms),
                is_mock=True,
            )
            self._record_processed_frame(state, message, digest, response)
            return response

    @staticmethod
    def _record_processed_frame(
        state: StreamingState,
        message: FrameMessage,
        digest: str,
        response: CaptionMessage | NoEventMessage,
    ) -> None:
        state.last_frame_index = message.frame_index
        state.last_timestamp_ms = message.timestamp_ms
        state.last_frame_digest = digest
        state.last_response = response
        state.frames_received += 1

    async def _process_real_frame(
        self, state: StreamingState, message: FrameMessage, frame: bytes
    ) -> CaptionMessage | NoEventMessage:
        if state.first_frame_ms is None:
            state.first_frame_ms = message.timestamp_ms
        state.frames.append((message.timestamp_ms, frame))
        due_at = (
            state.first_frame_ms + 1_500
            if state.last_inference_ms is None
            else state.last_inference_ms + 3_000
        )
        if message.timestamp_ms < due_at:
            return self._no_event(message)

        selected: list[tuple[int, bytes]] = []
        for offset in (1_500, 1_000, 500, 0):
            target = message.timestamp_ms - offset
            candidates = [item for item in state.frames if item[0] <= target]
            if not candidates or target - candidates[-1][0] > 350:
                return self._no_event(message)
            selected.append(candidates[-1])
        if len({timestamp for timestamp, _ in selected}) != 4:
            return self._no_event(message)

        caption, inference_ms = await self.provider.caption(
            [payload for _, payload in selected],
            [timestamp for timestamp, _ in selected],
        )
        state.last_inference_ms = message.timestamp_ms
        if not caption:
            return self._no_event(message)
        event_index = state.next_event_index
        state.next_event_index += 1
        state.caption_history.append(caption)
        state.caption_history[:] = state.caption_history[-self.config.memory_k :] if self.config.memory_k else []
        return CaptionMessage(
            stream_id=message.stream_id,
            event_id=str(uuid5(NAMESPACE_URL, f"causalstream:stream:{message.stream_id}:event:{event_index}")),
            start_ms=selected[0][0],
            end_ms=selected[-1][0],
            caption=caption,
            model_id=self.settings.model_version,
            inference_ms=inference_ms,
            processing_delay_ms=0,
            is_mock=False,
        )

    @staticmethod
    def _no_event(message: FrameMessage) -> NoEventMessage:
        return NoEventMessage(
            stream_id=message.stream_id,
            frame_index=message.frame_index,
            timestamp_ms=message.timestamp_ms,
        )

    def reset(self, stream_id: str) -> bool:
        self._states.pop(stream_id, None)
        return True

    def update_config(self, values: dict[str, object]) -> StreamConfig:
        self.config = self.config.model_copy(update=values)
        return self.config

    def history(self, stream_id: str) -> list[str]:
        state = self._states.get(stream_id)
        return list(state.caption_history) if state is not None else []

    async def _get_or_create(self, stream_id: str) -> StreamingState:
        state = self._states.get(stream_id)
        if state is not None:
            return state
        async with self._global_lock:
            return self._states.setdefault(stream_id, StreamingState(stream_id=stream_id))

    @staticmethod
    def _decode_frame(image_b64: str) -> bytes:
        try:
            return base64.b64decode(image_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ServiceError("INVALID_FRAME", "image_b64 is not valid base64.", 422) from exc

    @staticmethod
    def _next_supported_event(
        state: StreamingState, timestamp_ms: int
    ) -> MockCaptionEvent | None:
        if state.next_event_index >= len(MOCK_CAPTION_EVENTS):
            return None
        event = MOCK_CAPTION_EVENTS[state.next_event_index]
        if event.end_ms > timestamp_ms:
            return None
        return event
