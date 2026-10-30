"""Run the Full075 offline video pipeline on an Apple Silicon GPU."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from PBL6_Full075_Test_077.demo_video_077 import (
    PARAMETERS,
    PROMPT,
    WEIGHTS_SHA,
    grid_from_frames,
    sample,
    verify_export,
    write,
)


def require_mps(torch) -> None:
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError(
            "PyTorch cannot access the Apple GPU through MPS. Check the native arm64 "
            "Python/PyTorch installation and run 'python macos_runtime.py check'."
        )


def infer(model_root: Path, samples: Path, output: Path, attention: str = "eager") -> None:
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

    require_mps(torch)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    manifest = json.loads((model_root / "full-manifest.json").read_text(encoding="utf-8"))
    assets = verify_export(model_root, manifest)
    sampling = json.loads((samples / "sampling.json").read_text(encoding="utf-8"))
    rows = sampling["rows"]
    if not rows:
        raise ValueError("No sampled windows to infer")

    started = perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        min_pixels=262144,
        max_pixels=655360,
    )
    processor_sec = perf_counter() - started
    started_load = perf_counter()
    device = torch.device("mps")
    dtype = torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=dtype,
        attn_implementation=attention,
        device_map={"": "mps"},
    ).eval().requires_grad_(False)
    model.generation_config = GenerationConfig.from_pretrained(model_root, local_files_only=True)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETERS:
        raise ValueError("Unexpected model size")
    torch.mps.synchronize()
    model_load_sec = perf_counter() - started_load

    output.mkdir(parents=True, exist_ok=False)
    write(
        output / "load.json",
        {
            "model": "full075",
            "weights_sha256": WEIGHTS_SHA,
            "assets": assets,
            "parameters": PARAMETERS,
            "dtype": "float16",
            "frozen": True,
            "device": "mps",
            "attention": attention,
            "verification_processor_sec": processor_sec,
            "model_load_sec": model_load_sec,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
    )

    began = perf_counter()
    with (output / "captions.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            window_start = perf_counter()
            frames = [cv2.imread(str(samples / name)) for name in row["frames"]]
            if any(frame is None for frame in frames):
                raise ValueError("Missing sampled frame")
            grid = grid_from_frames(frames)
            text = PROMPT + "\nFrame timestamps: " + ", ".join(
                f"{timestamp:.6f}s" for timestamp in row["timestamps"]
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.fromarray(grid[:, :, ::-1])},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs.pop("token_type_ids", None)
            inputs = {
                key: value.to(device=device, dtype=dtype)
                if value.is_floating_point()
                else value.to(device)
                for key, value in inputs.items()
            }
            length = inputs["input_ids"].shape[1]
            if length > 591:
                raise ValueError("Input exceeds unchanged 591-token limit")
            torch.mps.synchronize()
            generation_start = perf_counter()
            with torch.inference_mode():
                tokens = model.generate(
                    **inputs, do_sample=False, max_new_tokens=64, use_cache=True
                )
            torch.mps.synchronize()
            cpu_tokens = tokens.cpu().numpy()
            result = {
                **row,
                "model": "full075",
                "weights_sha256": WEIGHTS_SHA,
                "raw_text": processor.batch_decode(
                    tokens[:, length:].cpu(), skip_special_tokens=True
                )[0],
                "input_tokens": length,
                "generated_tokens": tokens.shape[1] - length,
                "sequence_sha256": hashlib.sha256(cpu_tokens.tobytes()).hexdigest(),
                "processor_sec": generation_start - window_start,
                "generation_sec": perf_counter() - generation_start,
                "total_sec": perf_counter() - window_start,
            }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            summary = {key: result[key] for key in ("index", "end", "raw_text", "total_sec")}
            print(json.dumps(summary), flush=True)

    write(
        output / "complete.json",
        {
            "status": "COMPLETE",
            "model": "full075",
            "windows": len(rows),
            "elapsed_sec": perf_counter() - began,
            "device": "mps",
            "vram_peak_allocated_bytes": None,
            "vram_peak_reserved_bytes": None,
            "training": False,
            "independent_accuracy": False,
            "realtime": False,
            "cloud_requests": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    sample_command = commands.add_parser("sample")
    sample_command.add_argument("video", type=Path)
    sample_command.add_argument("output", type=Path)
    infer_command = commands.add_parser("infer")
    infer_command.add_argument("model", type=Path)
    infer_command.add_argument("samples", type=Path)
    infer_command.add_argument("output", type=Path)
    infer_command.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    arguments = parser.parse_args()

    if arguments.command == "check":
        import torch

        require_mps(torch)
        print(f"MPS ready: PyTorch {torch.__version__}")
    elif arguments.command == "sample":
        rows = sample(arguments.video, arguments.output)
        print(f"Sampled {len(rows)} windows")
    else:
        infer(arguments.model, arguments.samples, arguments.output, arguments.attention)


if __name__ == "__main__":
    main()
