"""
FLUX.1-schnell text-to-image endpoint on AWS Trainium (Neuron), PyTorch-native.

- Loads black-forest-labs/FLUX.1-schnell (diffusers FluxPipeline), whole pipeline on Neuron.
- Tensor-parallel (FFN-only) shards the DiT transformer across the rank's NeuronCores.
- torch.compile(backend="neuron") on the transformer.
- Replaces the VAE decoder's nearest-neighbor Upsample2D (which neuronx-cc cannot
  compile — [F139] on aten.upsample_nearest2d) with a NKI kernel (nki_upsample.py).
- torchrun launches TP_DEGREE ranks; rank 0 serves FastAPI, others run the TP worker loop.

schnell is timestep- + guidance-distilled: 4 steps, guidance_scale=0.0.
"""
import os
import time
import base64
import logging
from io import BytesIO
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, parallelize_module)

from nki_upsample import upsample_nearest2x

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flux-schnell")

# Per-block compilation wraps dozens of distinct transformer blocks; the default
# TorchDynamo cache (8) then trips FailOnRecompileLimitHit during warmup. Raise it,
# mirroring the proven PAVEDigitalTwinDiffusion port (neuron_enhancer.py sets 128).
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

pipe = None
device = None
rank = 0
world_size = 1
_ready = False


class NkiNearestUpsample(torch.nn.Module):
    """Drop-in replacement for diffusers Upsample2D when it does nearest 2x upsample.
    Wraps the original module: applies the NKI 2x upsample kernel, then the original
    module's post-upsample conv (if any). Only valid for scale_factor=2 / mode=nearest."""
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
    logger.info(f"Rank {rank}: replaced {n} VAE Upsample2D modules with NKI kernel")


def compile_transformer_blocks(transformer):
    """Compile the DiT transformer PER-BLOCK instead of one fullgraph NEFF.

    Flux's transformer as a single graph exceeds neuronx-cc's ~5M-instruction ceiling
    -> [F139] neuronx-cc terminated abnormally (same failure class the PAVE port hit as
    NCC_IXTP002 on the SD-turbo VAE decoder, fixed there via per-leaf compile in
    neuron_enhancer._compile_leaf_blocks). Compiling each transformer_block +
    single_transformer_block separately keeps every NEFF under the ceiling while the
    thin top-level forward glue (embeds, norm_out, proj_out) stays eager.
    """
    kw = dict(backend="neuron", fullgraph=False, dynamic=False)
    n = 0
    for attr in ("transformer_blocks", "single_transformer_blocks"):
        blocks = getattr(transformer, attr, None)
        if blocks is None:
            continue
        for i, blk in enumerate(blocks):
            blocks[i] = torch.compile(blk, **kw)
            n += 1
            logger.info(f"  wrapped {attr}[{i}] ({n} total)")
    r = dist.get_rank() if dist.is_initialized() else 0
    logger.info(f"Rank {r}: compiled {n} transformer blocks per-block (avoids [F139] instruction ceiling)")
    return transformer


def shard_transformer(transformer, mesh):
    """FFN-only tensor parallelism (mirrors t5 apply_ffn_tp). Attention stays replicated
    (Colwise on to_q/k/v splits head_dim and breaks the per-head QK RMSNorm). Only the
    MLPs are sharded: Colwise in-proj + Rowwise out-proj so the matmul all-reduces back."""
    def tp(module, plan):
        try:
            parallelize_module(module, mesh, plan)
        except Exception as e:
            logger.warning(f"TP skip {type(module).__name__}: {e}")

    for blk in getattr(transformer, "transformer_blocks", []):
        tp(blk.ff, {"net.0.proj": ColwiseParallel(), "net.2": RowwiseParallel()})
        tp(blk.ff_context, {"net.0.proj": ColwiseParallel(), "net.2": RowwiseParallel()})
    return transformer


def load():
    global pipe, device, rank, world_size, _ready
    from diffusers import FluxPipeline

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"neuron:{rank}")
    logger.info(f"Rank {rank}/{world_size} device={device}")

    logger.info(f"Rank {rank}: loading {MODEL_NAME} (dtype={DTYPE}) ...")
    pipe = FluxPipeline.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)

    # Swap VAE nearest-upsamplers to the NKI kernel BEFORE moving to device.
    _swap_vae_upsamplers(pipe.vae)

    logger.info(f"Rank {rank}: moving pipeline to {device} ...")
    pipe.to(device)

    if world_size > 1:
        logger.info(f"Rank {rank}: TP-sharding transformer over {world_size} devices ...")
        mesh = DeviceMesh("neuron", list(range(world_size)))
        pipe.transformer = shard_transformer(pipe.transformer, mesh)

    logger.info(f"Rank {rank}: per-block compile of transformer (backend='neuron') ...")
    pipe.transformer = compile_transformer_blocks(pipe.transformer)

    logger.info(f"Rank {rank}: warmup {WIDTH}x{HEIGHT} {NUM_STEPS} steps (NEFF compile) ...")
    t0 = time.time()
    _run("warmup", NUM_STEPS, None)
    logger.info(f"Rank {rank}: warmup done in {time.time()-t0:.1f}s")
    dist.barrier()
    _ready = True


def _run(prompt, steps, seed):
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    out = pipe(prompt, height=HEIGHT, width=WIDTH,
               num_inference_steps=steps, guidance_scale=GUIDANCE,
               max_sequence_length=MAX_SEQ_LEN, generator=gen)
    return out.images[0]


def worker_loop():
    logger.info(f"Rank {rank}: worker loop")
    while True:
        sig = torch.zeros(2, dtype=torch.int64, device=device)  # [cmd, steps]
        dist.broadcast(sig, src=0)
        if sig[0].item() == 0:
            break
        _run("worker", int(sig[1].item()) or NUM_STEPS, 0)


def run_server():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import uvicorn

    app = FastAPI(title="FLUX.1-schnell (Neuron, TP)")

    class GenReq(BaseModel):
        prompt: str
        num_inference_steps: Optional[int] = Field(None, ge=1, le=50)
        seed: Optional[int] = None

    class GenResp(BaseModel):
        image: str
        latency_s: float
        tp_degree: int

    @app.get("/health")
    def health():
        return {"status": "healthy", "tp_degree": world_size}

    @app.get("/readiness")
    def readiness():
        if not _ready:
            raise HTTPException(status_code=503, detail="not ready")
        return {"status": "ready", "tp_degree": world_size}

    @app.post("/generate", response_model=GenResp)
    def generate(req: GenReq):
        if not _ready:
            raise HTTPException(status_code=503, detail="not ready")
        steps = req.num_inference_steps or NUM_STEPS
        if world_size > 1:
            dist.broadcast(torch.tensor([1, steps], dtype=torch.int64, device=device), src=0)
        t0 = time.time()
        img = _run(req.prompt, steps, req.seed)
        buf = BytesIO(); img.save(buf, format="PNG")
        return GenResp(image=base64.b64encode(buf.getvalue()).decode(),
                       latency_s=round(time.time()-t0, 3), tp_degree=world_size)

    logger.info("Rank 0: starting uvicorn on :8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)


def main():
    load()
    if rank == 0:
        run_server()
    else:
        worker_loop()


if __name__ == "__main__":
    main()
