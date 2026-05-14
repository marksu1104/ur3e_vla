"""
vla_server.py — OpenVLA inference server (FastAPI).

⚠️  跑在 env_openvla (Python 3.11), 不是 env_isaaclab_ros2.
    這個 server 跑起來後, sim_vla.py 透過 HTTP 呼叫它.

Usage:
    conda activate env_openvla
    python vla_server.py --model-path ~/models/openvla-7b --port 8000

    # 低 VRAM 模式 (quantization):
    python vla_server.py --model-path ~/models/openvla-7b --load-in-4bit

Endpoints:
    GET  /health          — 確認 server 活著 + 模型已載入
    POST /predict         — 主推論端點
    GET  /config          — 查看當前模型設定

POST /predict 格式:
    Request:
        {
            "image_b64": "<base64 encoded JPEG/PNG>",
            "instruction": "pick up the red mug",
            "unnorm_key": "bridge_orig"   // optional, default "bridge_orig"
        }

    Response:
        {
            "action": [dx, dy, dz, dr, dp, dyaw, gripper],  // 7-DoF
            "action_raw": [...],   // 未 unnormalize 的原始輸出
            "latency_ms": 123.4
        }
"""

import argparse
import base64
import io
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

# OpenVLA
from transformers import AutoModelForVision2Seq, AutoProcessor


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Args                                                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="openvla/openvla-7b",
                   help="Local path or HuggingFace model ID")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="4-bit quantization (需要 bitsandbytes, 省約 8GB VRAM)")
    p.add_argument("--load-in-8bit", action="store_true",
                   help="8-bit quantization (省約 4GB VRAM)")
    p.add_argument("--unnorm-key", type=str, default="bridge_orig",
                   help="Action unnormalization key. 用 fine-tuned 模型時改成你的 dataset name")
    return p.parse_args()


ARGS = parse_args()


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Model 載入 (startup 時做一次)                                    ║
# ╚══════════════════════════════════════════════════════════════════╝

MODEL = None
PROCESSOR = None
DEVICE = None


def load_model():
    global MODEL, PROCESSOR, DEVICE

    print(f"[VLA Server] Loading model")
    print(f"[VLA Server] Device: {ARGS.device}")
    t0 = time.monotonic()

    DEVICE = ARGS.device

    PROCESSOR = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", 
        trust_remote_code=True
    )

    # 量化選項
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
        "openvla/openvla-7b",
        **load_kwargs,
    )

    # 只有非量化時才手動 .to(device)
    # 量化模式下 model 已自動 dispatch
    if not ARGS.load_in_4bit and not ARGS.load_in_8bit:
        MODEL = MODEL.to(DEVICE)

    MODEL.eval()

    elapsed = time.monotonic() - t0
    print(f"[VLA Server] Model loaded in {elapsed:.1f}s")

    # Warm-up: 跑一張假圖讓 CUDA kernel 編譯好 (之後 latency 才準)
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


def _warmup():
    dummy_img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
    inputs = PROCESSOR(
        text="pick up the object",
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


# ╔══════════════════════════════════════════════════════════════════╗
# ║  FastAPI app                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="OpenVLA Server", lifespan=lifespan)

# CORS (讓 sim_vla.py 跑在同機器的任意 port 都能存取)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Request / Response schemas                                       ║
# ╚══════════════════════════════════════════════════════════════════╝

class PredictRequest(BaseModel):
    image_b64: str            # base64 encoded image (JPEG or PNG)
    instruction: str          # natural language task instruction
    unnorm_key: Optional[str] = None   # None = 用 server 預設


class PredictResponse(BaseModel):
    action: list[float]       # 7-DoF: [dx, dy, dz, dr, dp, dyaw, gripper]
    action_raw: list[float]   # 未 unnormalize 的原始模型輸出 (debug 用)
    latency_ms: float


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Endpoints                                                        ║
# ╚══════════════════════════════════════════════════════════════════╝

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
    # 看這個模型支援哪些 unnorm_key (dataset 名稱)
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

    # ── 解碼圖片 ───────────────────────────────────────────────────
    try:
        img_bytes = base64.b64decode(req.image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    # OpenVLA 預期 256x256 (或其他 resize, 但以 256 為最佳)
    img = img.resize((256, 256), Image.LANCZOS)

    # ── 準備 inputs ───────────────────────────────────────────────
    unnorm_key = req.unnorm_key or ARGS.unnorm_key
    inputs = PROCESSOR(
        text=req.instruction,
        images=img,
        return_tensors="pt",
    )
    inputs = _move_inputs_to_device(inputs)

    # ── 推論 ──────────────────────────────────────────────────────
    with torch.no_grad():
        try:
            action = MODEL.predict_action(
                **inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    # action shape: (7,) numpy array
    if hasattr(action, "cpu"):
        action_np = action.cpu().numpy()
    else:
        action_np = np.array(action)

    action_np = action_np.flatten()[:7]

    latency_ms = (time.monotonic() - t0) * 1000

    return PredictResponse(
        action=action_np.tolist(),
        action_raw=action_np.tolist(),  # 如果有 raw 分開的話再改
        latency_ms=round(latency_ms, 1),
    )


# ╔══════════════════════════════════════════════════════════════════╗
# ║  Entry point                                                      ║
# ╚══════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=ARGS.host,
        port=ARGS.port,
        log_level="info",
    )