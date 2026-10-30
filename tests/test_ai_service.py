import asyncio
import base64

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.core.errors import ServiceError
from app.inference.providers import RealInferenceProvider
from app.main import app, manager, streaming_manager
from app.schemas.streaming import FrameMessage, StreamConfig
from app.streaming.manager import StreamingManager


@pytest.fixture(autouse=True)
async def clear_sessions():
    manager._sessions.clear()
    streaming_manager._states.clear()
    streaming_manager.config = StreamConfig()
    yield
    manager._sessions.clear()
    streaming_manager._states.clear()


async def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


async def create_demo(ac: AsyncClient, session_id: str, scenario_id: str = "cup_interaction"):
    return await ac.post(
        "/internal/v1/sessions",
        json={"session_id": session_id, "mode": "DEMO", "scenario_id": scenario_id},
    )


@pytest.mark.asyncio
async def test_mock_demo_sequence_is_deterministic():
    async with await client() as ac:
        assert (await create_demo(ac, "s1")).status_code == 200
        decisions = []
        for _ in range(4):
            response = await ac.post("/internal/v1/sessions/s1/advance")
            assert response.status_code == 200
            decisions.append(response.json()["decision"])
        assert decisions == ["WAIT", "WAIT", "SPEAK", "UPDATE"]
        assert (await ac.post("/internal/v1/sessions/s1/advance")).status_code == 409


@pytest.mark.asyncio
async def test_sessions_keep_independent_cursors():
    async with await client() as ac:
        await create_demo(ac, "s1")
        await create_demo(ac, "s2")
        await ac.post("/internal/v1/sessions/s1/advance")
        second = await ac.post("/internal/v1/sessions/s2/advance")
        assert second.json()["frame_index"] == 0


@pytest.mark.asyncio
async def test_pause_resume_and_delete():
    async with await client() as ac:
        await create_demo(ac, "s1")
        assert (await ac.post("/internal/v1/sessions/s1/pause")).status_code == 200
        assert (await ac.post("/internal/v1/sessions/s1/advance")).status_code == 409
        assert (await ac.post("/internal/v1/sessions/s1/resume")).status_code == 200
        assert (await ac.post("/internal/v1/sessions/s1/advance")).status_code == 200
        assert (await ac.delete("/internal/v1/sessions/s1")).status_code == 200
        assert (await ac.post("/internal/v1/sessions/s1/advance")).status_code == 404


@pytest.mark.asyncio
async def test_mock_provider_accepts_live_session():
    async with await client() as ac:
        response = await ac.post("/internal/v1/sessions", json={"session_id": "live1", "mode": "LIVE"})
        assert response.status_code == 200
        missing = await ac.post("/internal/v1/sessions/missing/advance")
        assert missing.status_code == 404


@pytest.mark.asyncio
async def test_concurrent_advances_do_not_corrupt_state():
    async with await client() as ac:
        await create_demo(ac, "s1")
        responses = await asyncio.gather(
            ac.post("/internal/v1/sessions/s1/advance"),
            ac.post("/internal/v1/sessions/s1/advance"),
        )
        frames = sorted(response.json()["frame_index"] for response in responses)
        assert frames == [0, 1]


def test_streaming_contract_is_causal_and_resettable():
    jpeg = base64.b64encode(b"\xff\xd8\xffmock-frame").decode()
    with TestClient(app) as ac:
        created = ac.post(
            "/internal/v1/sessions",
            json={"session_id": "stream-1", "mode": "LIVE"},
        )
        assert created.status_code == 200

        with ac.websocket_connect("/v1/streams/stream-1") as websocket:
            websocket.send_json(
                {
                    "type": "frame",
                    "stream_id": "stream-1",
                    "frame_index": 0,
                    "timestamp_ms": 1_000,
                    "image_b64": jpeg,
                }
            )
            assert websocket.receive_json()["type"] == "no_event"

            websocket.send_json(
                {
                    "type": "frame",
                    "stream_id": "stream-1",
                    "frame_index": 1,
                    "timestamp_ms": 2_300,
                    "image_b64": jpeg,
                }
            )
            first = websocket.receive_json()
            assert first["type"] == "caption"
            assert first["end_ms"] <= 2_300
            assert first["caption"] == "A person is picking up a ball."

            websocket.send_json({
                "type": "frame", "stream_id": "stream-1", "frame_index": 1,
                "timestamp_ms": 2_300, "image_b64": jpeg,
            })
            assert websocket.receive_json() == first

            websocket.send_json(
                {
                    "type": "frame",
                    "stream_id": "stream-1",
                    "frame_index": 2,
                    "timestamp_ms": 8_700,
                    "image_b64": jpeg,
                }
            )
            second = websocket.receive_json()
            assert second["caption"] == "The same person throws the ball."

        with ac.websocket_connect("/v1/streams/stream-1") as reconnected:
            reconnected.send_json({
                "type": "frame", "stream_id": "stream-1", "frame_index": 2,
                "timestamp_ms": 8_700, "image_b64": jpeg,
            })
            assert reconnected.receive_json() == second

        reset = ac.post("/v1/streams/stream-1/reset")
        assert reset.status_code == 200
        assert streaming_manager.history("stream-1") == []


def test_stream_config_supports_memory_ab_test():
    with TestClient(app) as ac:
        updated = ac.put(
            "/v1/config",
            json={"memory_enabled": False, "memory_k": 0, "sample_frames": 2},
        )
        assert updated.status_code == 200
        assert updated.json()["memory_enabled"] is False
        assert updated.json()["memory_k"] == 0
        models = ac.get("/v1/models")
        assert models.json()["items"][0]["streaming_ready"] is True


@pytest.mark.asyncio
async def test_real_stream_uses_four_past_frames_and_skips_empty_caption():
    class FakeRuntime:
        def __init__(self):
            self.windows = []

        def generate(self, frames, timestamps_ms):
            self.windows.append((frames, timestamps_ms))
            return ("A person washes their hands." if len(self.windows) == 1 else "", 1200.0)

    settings = Settings(_env_file=None, INFERENCE_PROVIDER="real", MODEL_VERSION="full075-test")
    provider = RealInferenceProvider(settings)
    provider.runtime = FakeRuntime()
    provider.ready = True
    streaming = StreamingManager(settings, provider)
    image = base64.b64encode(b"\xff\xd8\xffreal-frame").decode()
    responses = []
    for index, timestamp in enumerate(range(0, 5_000, 500)):
        frame = FrameMessage(
            stream_id="real-1", frame_index=index, timestamp_ms=timestamp, image_b64=image
        )
        responses.append(await streaming.process_frame(frame))
        if index in {0, 3, 9}:
            assert await streaming.process_frame(frame) == responses[index]

    with pytest.raises(ServiceError):
        await streaming.process_frame(FrameMessage(
            stream_id="real-1", frame_index=9, timestamp_ms=4_500,
            image_b64=base64.b64encode(b"\xff\xd8\xffchanged").decode(),
        ))

    assert [response.type for response in responses[:3]] == ["no_event"] * 3
    assert responses[3].type == "caption"
    assert responses[3].start_ms == 0
    assert responses[3].end_ms == 1_500
    assert responses[3].is_mock is False
    assert responses[3].model_id == "full075-test"
    assert responses[9].type == "no_event"
    assert len(provider.runtime.windows) == 2
    assert provider.runtime.windows[0][1] == [0, 500, 1_000, 1_500]
    assert provider.runtime.windows[1][1] == [3_000, 3_500, 4_000, 4_500]

