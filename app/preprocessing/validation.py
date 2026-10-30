from app.core.errors import ServiceError

SUPPORTED_MAGIC = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",
}


def validate_image_bytes(payload: bytes, max_size_mb: int) -> None:
    if not payload:
        raise ServiceError("INVALID_FRAME", "Frame payload is empty.", 422)
    if len(payload) > max_size_mb * 1024 * 1024:
        raise ServiceError("FRAME_TOO_LARGE", "Frame payload exceeds the configured size limit.", 413)
    if not any(payload.startswith(magic) for magic in SUPPORTED_MAGIC):
        raise ServiceError("INVALID_FRAME", "Unsupported or invalid image format.", 422)

