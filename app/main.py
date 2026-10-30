import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.internal.routes import router
from app.api.v1.routes import router as streaming_router
from app.core.config import get_settings
from app.core.errors import ServiceError, service_error_handler
from app.inference.providers import RealInferenceProvider, build_provider
from app.sessions.manager import InferenceSessionManager
from app.streaming.manager import StreamingManager

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())
provider = build_provider(settings)
manager = InferenceSessionManager(provider, settings)
streaming_manager = StreamingManager(settings, provider)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if isinstance(provider, RealInferenceProvider):
        await asyncio.to_thread(provider.load)
    yield

app = FastAPI(
    title="CausalStream AI Inference Service",
    version="0.1.0",
    description=(
        "Internal AI service for CausalStream. Mock mode is deterministic and does not "
        "perform real visual action recognition."
    ),
    lifespan=lifespan,
)
app.add_exception_handler(ServiceError, service_error_handler)
app.include_router(router)
app.include_router(streaming_router)

