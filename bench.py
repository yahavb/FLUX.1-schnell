"""
FLUX.1-schnell 3-run benchmark / bring-up entrypoint (headless, one-shot).

WHY this exists (vs serve.py): serve.py is a FastAPI server run under a Deployment.
When load()/compile fails, the container exits, k8s restarts it, and the pod
RECYCLES — the compile error scrolls past and we never see the root cause. This
script is the Job counterpart: restartPolicy=Never + backoffLimit=0 => it runs
ONCE and stops, so the full compile log survives.

It also does what serve.py does NOT: torch.compile(backend="neuron") on ALL
pipeline components — both text encoders (CLIP + T5), the DiT transformer, and the
VAE decoder (latent->pixel) — so we learn exactly which component/op the compiler
chokes on. serve.py only compiles the transformer.

Every rank runs the IDENTICAL fixed sequence (schnell is guidance/step-distilled,
do_sample effectively off, same shapes every iter) so TP collectives stay in
lockstep without any rank-0 broadcast coordination. Only rank 0 prints the skill's
built=/median= markers and writes the PNG for the mandatory output-correctness gate.

Markers this prints (the neuron-3run-benchmark harness greps for these):
  built=<sec>s                          # [1/3] warmup: NEFF compile-or-load, not timed
  median=<ms>ms throughput=<f> img/s    # [2/3] clean run: THE real latency
"""
import os
import time
import argparse
import logging
import statistics

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

# Reuse the proven pieces from serve.py: VAE NKI-upsample swap, FFN-only TP shard,
# and the env-derived constants (MODEL_NAME/HEIGHT/WIDTH/NUM_STEPS/GUIDANCE/...).
import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flux-bench")

PROMPT = "a photo of an astronaut riding a horse on the moon, high detail"


def _compile(mod, name, rank):
    logger.info(f"Rank {rank}: torch.compile({name}, backend='neuron', dynamic=False) ...")
    return torch.compile(mod, backend="neuron", fullgraph=False, dynamic=False)


def build_pipe():
    from diffusers import FluxPipeline

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"neuron:{rank}")
    logger.info(f"Rank {rank}/{world} device={device}")

    logger.info(f"Rank {rank}: loading {serve.MODEL_NAME} (dtype={serve.DTYPE}) ...")
    pipe = FluxPipeline.from_pretrained(serve.MODEL_NAME, torch_dtype=serve.DTYPE)

    # VAE nearest-upsample -> NKI kernel BEFORE moving to device ([F139] fix).
    serve._swap_vae_upsamplers(pipe.vae)

    logger.info(f"Rank {rank}: moving pipeline to {device} ...")
    pipe.to(device)

    if world > 1:
        logger.info(f"Rank {rank}: TP-sharding transformer (FFN-only) over {world} devices ...")
        mesh = DeviceMesh("neuron", list(range(world)))
        pipe.transformer = serve.shard_transformer(pipe.transformer, mesh)

    # ── Compile ALL components (the point of this Job). Each is lazy: the actual
    #    NEFF compile happens on first forward during warmup, so with
    #    NEURON_LAUNCH_BLOCKING=1 a failure names the exact component/op. ──
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = _compile(pipe.text_encoder, "text_encoder [CLIP]", rank)
    if getattr(pipe, "text_encoder_2", None) is not None:
        pipe.text_encoder_2 = _compile(pipe.text_encoder_2, "text_encoder_2 [T5]", rank)
    # DiT: compile PER-BLOCK, not one fullgraph — the whole-transformer graph exceeds
    # neuronx-cc's ~5M-instruction ceiling -> [F139] (see serve.compile_transformer_blocks).
    logger.info(f"Rank {rank}: per-block compile of transformer [DiT] ...")
    pipe.transformer = serve.compile_transformer_blocks(pipe.transformer)
    # VAE latent->pixel is pipe.vae.decode(z) -> self.decoder(z); compile the decoder.
    pipe.vae.decoder = _compile(pipe.vae.decoder, "vae.decoder [latent->pixel]", rank)

    return pipe, rank, world, device


def run_once(pipe, steps, seed=0):
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    out = pipe(PROMPT, height=serve.HEIGHT, width=serve.WIDTH,
               num_inference_steps=steps, guidance_scale=serve.GUIDANCE,
               max_sequence_length=serve.MAX_SEQ_LEN, generator=gen)
    return out.images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--steps", type=int, default=serve.NUM_STEPS)
    args = ap.parse_args()

    pipe, rank, world, device = build_pipe()

    # ── [1/3] warmup: first full-pipeline run compiles every component's NEFFs. ──
    # HANG GUARD: a TP collective deadlock (graph break straddling all_reduce) would
    # hang forever with no error. Arm a watchdog so warmup fails loudly instead. Set
    # FLUX_WARMUP_TIMEOUT=0 to disable. Only rank 0 needs to print; every rank arms it.
    import signal
    timeout_s = int(os.environ.get("FLUX_WARMUP_TIMEOUT", "1800"))
    if timeout_s > 0:
        def _bail(signum, frame):
            raise TimeoutError(
                f"warmup exceeded {timeout_s}s on rank {rank} — likely a TP collective "
                f"deadlock (per-block compile graph break straddling all_reduce). "
                f"Re-run at TP_DEGREE=1 to confirm.")
        signal.signal(signal.SIGALRM, _bail)
        signal.alarm(timeout_s)

    t0 = time.time()
    img = None
    for _ in range(max(1, args.warmup)):
        print(f"Rank {rank}: warmup forward start ...", flush=True)
        img = run_once(pipe, args.steps)
        print(f"Rank {rank}: warmup forward done ({time.time()-t0:.1f}s)", flush=True)
    dist.barrier()
    if timeout_s > 0:
        signal.alarm(0)
    built = time.time() - t0
    if rank == 0:
        print(f"built={built:.1f}s  (warmup {max(1, args.warmup)} iter; all components compiled)", flush=True)

    # ── [2/3]/[3/3] timed iters: warm cache, same shapes -> real per-image latency. ──
    lat_ms = []
    for _ in range(args.iters):
        t = time.time()
        img = run_once(pipe, args.steps)
        dist.barrier()  # collective forces device sync; output already on host via PIL
        lat_ms.append((time.time() - t) * 1000.0)
    if args.iters and rank == 0:
        med = statistics.median(lat_ms)
        print(f"median={med:.1f}ms throughput={1000.0 / med:.4f} img/s  (iters={args.iters})", flush=True)

    # ── mandatory output-correctness gate: save the image, print a loud banner. ──
    if rank == 0 and img is not None:
        outp = os.environ.get("FLUX_OUT", "/tmp/flux_out.png")
        img.save(outp)
        print("=" * 44, flush=True)
        print("  VALIDATE THE OUTPUT — flux-schnell", flush=True)
        print(f"  saved: {outp}  ({serve.WIDTH}x{serve.HEIGHT}, {args.steps} steps, TP={world})", flush=True)
        print("  pull: kubectl cp <pod>:%s ./flux_out.png && open ./flux_out.png" % outp, flush=True)
        print("=" * 44, flush=True)

    dist.barrier()


if __name__ == "__main__":
    main()
