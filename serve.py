"""
FLUX.1-schnell text-to-image endpoint on AWS Trainium (Neuron), PyTorch-native.

Loads black-forest-labs/FLUX.1-schnell via diffusers.FluxPipeline, moves it to the
Neuron device, torch.compile(backend="neuron")s the heavy transformer (the ~12B DiT
backbone), warms it up (first-call NEFF compile), and serves generation over FastAPI.

FLUX.1-schnell is timestep-distilled: run ~4 steps with guidance_scale=0.0.

Env:
  MODEL_NAME       black-forest-labs/FLUX.1-schnell
  HEIGHT, WIDTH    output resolution (default 1024x1024)
  NUM_STEPS        denoising steps (schnell: 4)
  MAX_SEQ_LEN      T5 sequence length (default 256 for schnell)
  DTYPE            bfloat16 (default) | float32
  COMPILE          "1" to torch.compile the transformer (default), "0" to run eager
"""
import os
import time
import base64
import logging
from io import BytesIO
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flux-schnell")

MODEL_NAME = os.environ.get("MODEL_NAME", "black-forest-labs/FLUX.1-schnell")
HEIGHT = int(os.environ.get("HEIGHT", 1024))
WIDTH = int(os.environ.get("WIDTH", 1024))
NUM_STEPS = int(os.environ.get("NUM_STEPS", 4))          # schnell is 4-step distilled
GUIDANCE = float(os.environ.get("GUIDANCE_SCALE", 0.0))  # schnell: guidance-distilled -> 0.0
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 256))
DTYPE = torch.bfloat16 if os.environ.get("DTYPE", "bfloat16") == "bfloat16" else torch.float32
COMPILE = os.environ.get("COMPILE", "1") == "1"
DEVICE = os.environ.get("DEVICE", "neuron")

app = FastAPI(title="FLUX.1-schnell (Neuron)")

pipe = None
_ready = False


def load():
    global pipe, _ready
    from diffusers import FluxPipeline

    logger.info(f"Loading {MODEL_NAME} (dtype={DTYPE}) ...")
    pipe = FluxPipeline.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)

    # Move to Neuron. NOTE: assign-style materialization matters on the newer torch
    # stack (models may build on meta); FluxPipeline.from_pretrained loads real weights,
    # then .to(device) places them on-device.
    logger.info(f"Moving pipeline to {DEVICE} ...")
    pipe.to(DEVICE)

    if COMPILE:
        # Compile the heavy DiT backbone (the transformer) with the Neuron backend.
        # Keep the VAE / text encoders eager first; add them once the transformer compiles.
        logger.info("torch.compile(transformer, backend='neuron') ...")
        pipe.transformer = torch.compile(
            pipe.transformer, backend="neuron", fullgraph=False, dynamic=False
        )

    # Warmup: triggers first-call NEFF compilation (minutes) at the fixed HxW.
    logger.info(f"Warmup generate {WIDTH}x{HEIGHT}, {NUM_STEPS} steps (NEFF compile) ...")
    t0 = time.time()
    _ = pipe(
        "warmup",
        height=HEIGHT, width=WIDTH,
        num_inference_steps=NUM_STEPS, guidance_scale=GUIDANCE,
        max_sequence_length=MAX_SEQ_LEN,
        generator=torch.Generator().manual_seed(0),
    )
    logger.info(f"Warmup done in {time.time()-t0:.1f}s — ready.")
    _ready = True


class GenerateRequest(BaseModel):
    prompt: str
    num_inference_steps: Optional[int] = Field(None, ge=1, le=50)
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    image: str = Field(..., description="Base64-encoded PNG")
    latency_s: float


@app.on_event("startup")
def _startup():
    load()


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME, "compiled": COMPILE}


@app.get("/readiness")
def readiness():
    if not _ready:
        raise HTTPException(status_code=503, detail="model not ready")
    return {"status": "ready"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not _ready:
        raise HTTPException(status_code=503, detail="model not ready")
    steps = req.num_inference_steps or NUM_STEPS
    gen = torch.Generator().manual_seed(req.seed) if req.seed is not None else None
    t0 = time.time()
    out = pipe(
        req.prompt,
        height=HEIGHT, width=WIDTH,
        num_inference_steps=steps, guidance_scale=GUIDANCE,
        max_sequence_length=MAX_SEQ_LEN,
        generator=gen,
    )
    img = out.images[0]
    buf = BytesIO()
    img.save(buf, format="PNG")
    return GenerateResponse(
        image=base64.b64encode(buf.getvalue()).decode(),
        latency_s=round(time.time() - t0, 3),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
