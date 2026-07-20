"""
FLUX.1-schnell 3-run benchmark / bring-up entrypoint (headless, one-shot, SINGLE CORE).

WHY this exists (vs serve.py): serve.py is a FastAPI server under a Deployment. When
compile fails, the container exits, k8s restarts it, and the pod RECYCLES — the compile
error scrolls past. This is the Job counterpart: restartPolicy=Never + backoffLimit=0, so
it runs ONCE and the full log survives.

Recipe matches serve.py (which mirrors the PROVEN pave-dig-twin-diff t2i port): single
core, text encoders eager on host, transformer per-block + VAE on device. No TP, no
torchrun, no collectives — the pave opt-log showed every inter-core split regressed.

Markers (the neuron-3run-benchmark harness greps for these):
  built=<sec>s                          # [1/3] warmup: NEFF compile-or-load, not timed
  median=<ms>ms throughput=<f> img/s    # [2/3] clean run: THE real latency
"""
import os
import sys
import time
import argparse
import logging
import statistics

import torch

# serve.py holds the shared, proven load recipe (VAE NKI swap, per-block compile,
# host-side prompt encode) and the env-derived constants.
import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flux-bench")

PROMPT = os.environ.get("FLUX_PROMPT", "a photo of an astronaut riding a horse on Mars, high detail")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--steps", type=int, default=serve.NUM_STEPS)
    args = ap.parse_args()

    # Build the pipeline via serve.load() — identical path to the server, so the bench
    # measures exactly what serves: single core, host text encoders, per-block transformer.
    t0 = time.time()
    serve.load()
    built = time.time() - t0
    print(f"built={built:.1f}s  (serve.load: compiled transformer per-block + VAE, warmup ran)", flush=True)

    # serve.load() already ran one warmup generation. Extra warmups if asked.
    for _ in range(max(0, args.warmup - 1)):
        serve._run(PROMPT, args.steps, 0)

    # ── timed iters: warm cache, same shape -> real per-image latency. ──
    lat_ms = []
    img = None
    for i in range(args.iters):
        t = time.time()
        img = serve._run(PROMPT, args.steps, 0)
        dt = (time.time() - t) * 1000.0
        lat_ms.append(dt)
        print(f"  iter {i}: {dt:.1f}ms", flush=True)
    if args.iters:
        med = statistics.median(lat_ms)
        print(f"median={med:.1f}ms throughput={1000.0 / med:.4f} img/s  (iters={args.iters})", flush=True)

    # ── mandatory output-correctness gate: save the image, print a loud banner. ──
    if img is not None:
        outp = os.environ.get("FLUX_OUT", "/tmp/flux_out.png")
        img.save(outp)
        print("=" * 44, flush=True)
        print("  VALIDATE THE OUTPUT — flux-schnell", flush=True)
        print(f"  saved: {outp}  ({serve.WIDTH}x{serve.HEIGHT}, {args.steps} steps, single-core)", flush=True)
        print("  pull: kubectl cp <pod>:%s ./flux_out.png && open ./flux_out.png" % outp, flush=True)
        print("=" * 44, flush=True)

    # ── ORDERED TEARDOWN: fixes the teardown SIGSEGV + "nrt_unload NRT uninitialized" spam.
    # Cause: at interpreter exit the Neuron runtime's atexit handler calls nrt_close()
    # BEFORE Python GCs the ~100 compiled-block objects; their finalizers then call
    # nrt_unload on an already-closed runtime -> use-after-free -> one ERROR per NEFF and
    # ultimately a "Segmentation fault (core dumped)". This all happens AFTER the real
    # work (built=/median=/image already produced), so the fix is to (1) drop refs and GC
    # while NRT is still up (clean unload), then (2) os._exit(0) to terminate immediately,
    # skipping the atexit/finalizer chain that use-after-frees. Deterministic exit 0.
    import gc
    sys.stdout.flush(); sys.stderr.flush()
    serve.pipe = None
    img = None
    gc.collect()
    os._exit(0)


if __name__ == "__main__":
    main()
