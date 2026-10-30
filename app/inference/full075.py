"""Native MPS inference for the verified Full075 export."""

import json
from pathlib import Path
from threading import Lock
from time import perf_counter

from PBL6_Full075_Test_077.demo_video_077 import (
    PARAMETERS,
    PROMPT,
    grid_from_frames,
    verify_export,
)


class Full075Runtime:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._lock = Lock()
        self.model = None
        self.processor = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS is unavailable on this host.")
        manifest = json.loads((self.model_path / "full-manifest.json").read_text())
        verify_export(self.model_path, manifest)
        processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            min_pixels=262144,
            max_pixels=655360,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=torch.float16,
            attn_implementation="eager",
            device_map={"": "mps"},
        ).eval().requires_grad_(False)
        model.generation_config = GenerationConfig.from_pretrained(
            self.model_path, local_files_only=True
        )
        if sum(parameter.numel() for parameter in model.parameters()) != PARAMETERS:
            raise RuntimeError("Unexpected Full075 parameter count.")
        torch.mps.synchronize()
        self.processor = processor
        self.model = model

    def generate(self, frames: list[bytes], timestamps_ms: list[int]) -> tuple[str, float]:
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        if self.model is None or self.processor is None:
            raise RuntimeError("Full075 model has not been loaded.")
        if len(frames) != 4 or len(timestamps_ms) != 4:
            raise ValueError("Full075 requires four frames and timestamps.")
        with self._lock:
            started = perf_counter()
            images = [cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR) for frame in frames]
            grid = grid_from_frames(images)
            prompt = PROMPT + "\nFrame timestamps: " + ", ".join(
                f"{timestamp / 1000:.6f}s" for timestamp in timestamps_ms
            )
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.fromarray(grid[:, :, ::-1])},
                    {"type": "text", "text": prompt},
                ],
            }]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)
            inputs = {
                key: value.to(device="mps", dtype=torch.float16)
                if value.is_floating_point() else value.to("mps")
                for key, value in inputs.items()
            }
            length = inputs["input_ids"].shape[1]
            if length > 591:
                raise ValueError("Input exceeds the Full075 591-token limit.")
            with torch.inference_mode():
                tokens = self.model.generate(
                    **inputs, do_sample=False, max_new_tokens=64, use_cache=True
                )
            torch.mps.synchronize()
            caption = self.processor.batch_decode(
                tokens[:, length:].cpu(), skip_special_tokens=True
            )[0]
            return caption.strip(), round((perf_counter() - started) * 1000, 3)
