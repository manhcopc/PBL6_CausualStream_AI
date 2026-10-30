# CausalStream AI Service

FastAPI inference service with a deterministic mock provider and the Full075 Qwen3-VL-2B caption provider for Apple Silicon MPS. This directory is an independent repository. The Backend connects to it; browsers should connect only to the Backend.

## Mock mode

Requirements: Python 3.11+. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Health endpoints: `/healthz` and `/readyz`. API documentation: `http://127.0.0.1:8001/docs`.

## Real model on macOS

The checkpoint is **not part of this repository**. Obtain it separately under the applicable model notice, then place the complete export (including `full-manifest.json` and `model.safetensors`) in `PBL6_Full075_Test_077/model075/`. See [MACOS_RUNTIME.md](MACOS_RUNTIME.md) for the MPS setup and startup command. The runtime checks the export manifest before loading it. Do not add the checkpoint, tar bundle, demo videos or inference results to Git.

The current real provider generates captions only. LIVE stores new captions as `EMIT`; it does not infer `REVISE`, `WITHDRAW` or a decision confidence score.

## Verify

```bash
python -m pytest -q
python -m ruff check app tests
```

The included Dockerfile installs the mock service dependencies only. The MPS runtime requires a macOS host and a separately supplied checkpoint.
