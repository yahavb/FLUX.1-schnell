"""
FLUX.1-schnell text-to-image endpoint on AWS Trainium (Neuron), PyTorch-native.

Recipe mirrors the PROVEN pave-dig-twin-diff text-to-image port (neuron_t2i.py),
which compiles and generates images on trn3/SDK-2.31:

- SINGLE CORE, no tensor-parallelism, no torchrun/collectives. The pave optimization
  log (#15/#16/#17) shows every inter-core split (SP, TP-conv) REGRESSED for this
  workload — data-parallel-per-core wins. flux-schnell at these sizes fits on one core.
- Text encoders (CLIP + T5) run EAGER ON THE HOST (CPU), NOT compiled and NOT on the
  Neuron device. They are cheap + per-prompt and compiling them adds eager-fallback
  noise. Only the heavy compute (DiT transformer + VAE decode) becomes NEFFs.
- The DiT transformer is compiled PER-BLOCK (each transformer_block +
  single_transformer_block), not one fullgraph — a single graph exceeds neuronx-cc's
  ~5M-instruction ceiling -> [F139] (same class the pave VAE decoder hit as NCC_IXTP002,
  fixed there by per-leaf compile).
- The VAE decoder's nearest Upsample2D (neuronx-cc [F139] on aten.upsample_nearest2d)
  is swapped for the NKI 2x-upsample kernel (nki_upsample.py, gated by test_nki_upsample).

schnell is timestep- + guidance-distilled: 4 steps, guidance_scale=0.0.
"""
import os
import time
import base64
import logging
from io import BytesIO
from typing import Optional

import torch

from nki_upsample import upsample_nearest2x

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flux-schnell")

# Per-block compile wraps dozens of blocks; default TorchDynamo cache (8) trips
# FailOnRecompileLimitHit. Raise it (proven pave t2i / neuron_enhancer fix).
try:
    import torch._dynamo as _dynamo
    _dynamo.config.cache_size_limit = max(getattr(_dynamo.config, "cache_size_limit", 8), 128)
    if hasattr(_dynamo.config, "accumulated_cache_size_limit"):
        _dynamo.config.accumulated_cache_size_limit = max(
            getattr(_dynamo.config, "accumulated_cache_size_limit", 256), 4096)
except Exception:
    pass

MODEL_NAME = os.environ.get("MODEL_NAME", "black-forest-labs/FLUX.1-schnell")
HEIGHT = int(os.environ.get("HEIGHT", 480))
WIDTH = int(os.environ.get("WIDTH", 640))
NUM_STEPS = int(os.environ.get("NUM_STEPS", 4))
GUIDANCE = float(os.environ.get("GUIDANCE_SCALE", 0.0))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", 256))
DTYPE = torch.bfloat16 if os.environ.get("DTYPE", "bfloat16") == "bfloat16" else torch.float32

# Single core: pin before any Neuron init (mirrors t2i's NEURON_RT_VISIBLE_CORES=0).
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "1")
os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

pipe = None
device = None
_ready = False


class NkiNearestUpsample(torch.nn.Module):
    """Drop-in replacement for diffusers Upsample2D nearest 2x upsample: applies the NKI
    2x kernel then the original post-upsample conv. Only valid for scale_factor=2/nearest."""
    def __init__(self, orig):
        super().__init__()
        self.conv = getattr(orig, "conv", None)

    def forward(self, hidden_states, output_size=None, *args, **kwargs):
        hidden_states = upsample_nearest2x(hidden_states.contiguous())
        if self.conv is not None:
            hidden_states = self.conv(hidden_states)
        return hidden_states


def _swap_vae_upsamplers(vae):
    """Replace every nearest-mode Upsample2D in the VAE decoder with the NKI version."""
    from diffusers.models.upsampling import Upsample2D
    n = 0
    for parent in vae.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, Upsample2D):
                setattr(parent, name, NkiNearestUpsample(child))
                n += 1
    logger.info(f"Replaced {n} VAE Upsample2D modules with NKI kernel")


def compile_transformer_blocks(transformer):
    """Compile the DiT transformer PER-BLOCK instead of one fullgraph NEFF.

    Flux's transformer as a single graph exceeds neuronx-cc's ~5M-instruction ceiling
    -> [F139] neuronx-cc terminated abnormally. Compiling each transformer_block +
    single_transformer_block separately keeps every NEFF under the ceiling while the
    thin top-level forward glue (embeds, norm_out, proj_out) stays eager. Mirrors the
    proven pave per-leaf-block strategy (_compile_leaf_blocks)."""
    kw = dict(backend="neuron", fullgraph=False, dynamic=False)
    n = 0
    for attr in ("transformer_blocks", "single_transformer_blocks"):
        blocks = getattr(transformer, attr, None)
        if blocks is None:
            continue
        for i in range(len(blocks)):
            blocks[i] = torch.compile(blocks[i], **kw)
            n += 1
    logger.info(f"Compiled {n} transformer blocks per-block (avoids [F139] instruction ceiling)")
    return transformer


def load():
    global pipe, device, _ready
    from diffusers import FluxPipeline

    device = torch.device("neuron:0")
    logger.info(f"Single-core Neuron device={device}")

    logger.info(f"Loading {MODEL_NAME} (dtype={DTYPE}) ...")
    pipe = FluxPipeline.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)

    # VAE nearest-upsample -> NKI kernel BEFORE moving to device ([F139] fix).
    _swap_vae_upsamplers(pipe.vae)

    # DEVICE SPLIT (the t2i lesson): text encoders stay EAGER ON HOST (CPU); only the
    # heavy compute (transformer + VAE) goes to the Neuron device and gets compiled.
    logger.info("Text encoders stay on CPU (eager, host); moving transformer + VAE to device ...")
    pipe.text_encoder.to("cpu").eval().requires_grad_(False)
    pipe.text_encoder_2.to("cpu").eval().requires_grad_(False)
    pipe.transformer.to(device)
    # VAE in fp32, NOT bf16. PAVE port (measured): "the SD VAE overflows fp16/low-precision
    # on Neuron (quality-critical) -> VAE runs fp32; only the UNet/transformer runs bf16."
    # A bf16 VAE decode overflows -> NaN -> BLANK image (observed). Keep the transformer in
    # DTYPE (bf16) for speed; decode in fp32 for correctness.
    pipe.vae.to(device=device, dtype=torch.float32)
    # diffusers calls self.vae.decode(latents) with the RAW bf16 latents (no cast), so an
    # fp32 VAE would dtype-mismatch. Wrap decode to upcast the incoming latent to fp32.
    _orig_decode = pipe.vae.decode
    def _decode_fp32(z, *a, **k):
        return _orig_decode(z.to(torch.float32), *a, **k)
    pipe.vae.decode = _decode_fp32

    # With the text encoders on CPU, diffusers' _execution_device property (derived from
    # module placement) resolves to CPU, so the pipeline allocates the initial latents on
    # CPU and hands them to the on-device transformer -> "input tensor is on cpu, expected
    # neuron" at x_embedder. Force the execution device to neuron so latents are created
    # on-device (the transformer+VAE compute all lives there; text embeds are moved to
    # device in _encode_on_host before pipe() is called).
    type(pipe)._execution_device = property(lambda self: device)

    logger.info("Per-block compile of transformer (backend='neuron') ...")
    pipe.transformer = compile_transformer_blocks(pipe.transformer)

    logger.info(f"Warmup {WIDTH}x{HEIGHT} {NUM_STEPS} steps (NEFF compile) ...")
    t0 = time.time()
    _run("warmup", NUM_STEPS, None)
    logger.info(f"Warmup done in {time.time()-t0:.1f}s")
    _ready = True


def _encode_on_host(prompt):
    """CLIP+T5 encode on the HOST (CPU text encoders), return embeds moved to the
    Neuron device. Passing these into the pipeline makes it SKIP its internal encode
    (which would run the CPU text encoders and hand device-mismatched tensors to the
    on-device transformer)."""
    prompt_embeds, pooled_prompt_embeds, _text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, device=torch.device("cpu"),
        num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN,
    )
    return (prompt_embeds.to(device=device, dtype=DTYPE),
            pooled_prompt_embeds.to(device=device, dtype=DTYPE))


def _run(prompt, steps, seed):
    prompt_embeds, pooled = _encode_on_host(prompt)
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    out = pipe(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled,
               height=HEIGHT, width=WIDTH, num_inference_steps=steps,
               guidance_scale=GUIDANCE, generator=gen)
    return out.images[0]


def run_server():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import uvicorn

    app = FastAPI(title="FLUX.1-schnell (Neuron, single-core)")

    class GenReq(BaseModel):
        prompt: str
        num_inference_steps: Optional[int] = Field(None, ge=1, le=50)
        seed: Optional[int] = None

    class GenResp(BaseModel):
        image: str
        latency_s: float

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.get("/readiness")
    def readiness():
        if not _ready:
            raise HTTPException(status_code=503, detail="not ready")
        return {"status": "ready"}

    @app.post("/generate", response_model=GenResp)
    def generate(req: GenReq):
        if not _ready:
            raise HTTPException(status_code=503, detail="not ready")
        steps = req.num_inference_steps or NUM_STEPS
        t0 = time.time()
        img = _run(req.prompt, steps, req.seed)
        buf = BytesIO(); img.save(buf, format="PNG")
        return GenResp(image=base64.b64encode(buf.getvalue()).decode(),
                       latency_s=round(time.time()-t0, 3))

    logger.info("Starting uvicorn on :8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


def main():
    load()
    run_server()


if __name__ == "__main__":
    main()
