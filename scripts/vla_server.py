"""FastAPI server for OpenVLA inference."""

import argparse
import base64
import io
import json
from pathlib import Path
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from transformers import AutoModelForVision2Seq, AutoProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        type=str,
        default="openvla/openvla-7b",
        help="Local model directory or Hugging Face model ID.",
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Load the model with 4-bit quantization.")
    p.add_argument("--load-in-8bit", action="store_true",
                   help="Load the model with 8-bit quantization.")
    p.add_argument("--unnorm-key", type=str, default="ur3e_vla_dataset",
                   help="Action unnormalization key.")
    return p.parse_args()


ARGS = parse_args()

MODEL = None
PROCESSOR = None
DEVICE = None


def resolve_model_path(model_path: str) -> str:
    """Return a valid local model path or Hugging Face model ID."""
    path = Path(model_path).expanduser()
    looks_like_path = (
        model_path.startswith("/")
        or model_path.startswith("./")
        or model_path.startswith("../")
        or model_path.startswith("~")
    )
    if looks_like_path:
        if not path.exists():
            raise RuntimeError(
                f"Local model path does not exist: {path}. "
                "Use --model-path openvla/openvla-7b to download from Hugging Face, "
                "or provide a directory that already contains the model files."
            )
        return str(path)
    return model_path


def load_local_dataset_statistics(model_path: str) -> Optional[dict]:
    """Load fine-tuning dataset statistics saved next to a local model export."""
    path = Path(model_path).expanduser()
    if not path.exists() or not path.is_dir():
        return None

    stats_path = path / "dataset_statistics.json"
    if not stats_path.exists():
        return None

    with stats_path.open("r") as f:
        stats = json.load(f)
    print(f"[VLA Server] Loaded dataset statistics: {stats_path}")
    print(f"[VLA Server] Available unnorm keys: {list(stats.keys())}")
    return stats


def load_model():
    global MODEL, PROCESSOR, DEVICE

    print(f"[VLA Server] Loading model")
    print(f"[VLA Server] Device: {ARGS.device}")
    t0 = time.monotonic()

    DEVICE = ARGS.device

    model_path = resolve_model_path(ARGS.model_path)
    print(f"[VLA Server] Model path: {model_path}")

    PROCESSOR = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    load_kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if ARGS.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
        load_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
        print("[VLA Server] Using 4-bit quantization")
    elif ARGS.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
        print("[VLA Server] Using 8-bit quantization")
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        print("[VLA Server] Using bfloat16")

    MODEL = AutoModelForVision2Seq.from_pretrained(
        model_path,
        **load_kwargs,
    )

    local_stats = load_local_dataset_statistics(model_path)
    if local_stats is not None:
        MODEL.norm_stats = local_stats
        if hasattr(MODEL, "config"):
            MODEL.config.norm_stats = local_stats

    if ARGS.unnorm_key not in getattr(MODEL, "norm_stats", {}):
        available = list(getattr(MODEL, "norm_stats", {}).keys())
        raise RuntimeError(
            f"Unnorm key '{ARGS.unnorm_key}' is not available. "
            f"Available keys: {available}"
        )

    if not ARGS.load_in_4bit and not ARGS.load_in_8bit:
        MODEL = MODEL.to(DEVICE)

    MODEL.eval()

    elapsed = time.monotonic() - t0
    print(f"[VLA Server] Model loaded in {elapsed:.1f}s")

    print("[VLA Server] Running warm-up inference...")
    _warmup()
    print("[VLA Server] Ready.")


def _move_inputs_to_device(inputs):
    model_dtype = None
    try:
        model_dtype = MODEL.dtype
    except Exception:
        model_dtype = None

    new_inputs = {}
    for k, v in inputs.items():
        if hasattr(v, "to"):
            if k in {"input_ids", "attention_mask", "token_type_ids", "position_ids"}:
                v = v.to(device=DEVICE, dtype=torch.long)
            else:
                v = v.to(DEVICE)
                if model_dtype is not None and isinstance(v, torch.Tensor) and v.is_floating_point:
                    v = v.to(model_dtype)
        new_inputs[k] = v
    return new_inputs


def build_prompt(instruction: str) -> str:
    """Return the prompt format used by OpenVLA fine-tuning."""
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def _warmup():
    dummy_img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    inputs = PROCESSOR(
        text=build_prompt("pick up the object"),
        images=dummy_img,
        return_tensors="pt",
    )
    inputs = _move_inputs_to_device(inputs)
    with torch.no_grad():
        MODEL.predict_action(
            **inputs,
            unnorm_key=ARGS.unnorm_key,
            do_sample=False,
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="OpenVLA Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    image_b64: str
    instruction: str
    unnorm_key: Optional[str] = None


class PredictResponse(BaseModel):
    action: list[float]
    action_raw: list[float]
    latency_ms: float


@app.get("/health")
def health():
    return {
        "status": "ok" if MODEL is not None else "loading",
        "model_path": ARGS.model_path,
        "device": ARGS.device,
        "unnorm_key": ARGS.unnorm_key,
        "quantization": "4bit" if ARGS.load_in_4bit else "8bit" if ARGS.load_in_8bit else "bf16",
    }


@app.get("/config")
def config():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    try:
        norm_stats = MODEL.norm_stats
        available_keys = list(norm_stats.keys()) if norm_stats else []
    except AttributeError:
        available_keys = ["unknown"]
    return {
        "available_unnorm_keys": available_keys,
        "default_unnorm_key": ARGS.unnorm_key,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    t0 = time.monotonic()

    try:
        img_bytes = base64.b64decode(req.image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    img = img.resize((256, 256), Image.LANCZOS)

    unnorm_key = req.unnorm_key or ARGS.unnorm_key
    inputs = PROCESSOR(
        text=build_prompt(req.instruction),
        images=img,
        return_tensors="pt",
    )
    inputs = _move_inputs_to_device(inputs)

    with torch.no_grad():
        try:
            action = MODEL.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )
        except Exception as e:
            print("[VLA Server] Inference failed:")
            print(traceback.format_exc())
            try:
                norm_stats = getattr(MODEL, "norm_stats", {})
                print(f"[VLA Server] Requested unnorm_key: {unnorm_key}")
                print(f"[VLA Server] Available unnorm_keys: {list(norm_stats.keys())}")
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    if hasattr(action, "cpu"):
        action_np = action.cpu().numpy()
    else:
        action_np = np.array(action)

    action_np = action_np.flatten()[:7]

    latency_ms = (time.monotonic() - t0) * 1000

    return PredictResponse(
        action=action_np.tolist(),
        action_raw=action_np.tolist(),
        latency_ms=round(latency_ms, 1),
    )

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=ARGS.host,
        port=ARGS.port,
        log_level="info",
    )
