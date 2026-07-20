# FLUX.1-schnell on AWS Trainium (PyTorch-native, torch.compile)

Serve [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
as a text→image endpoint on Trainium (trn3), PyTorch-native via `torch.compile(backend="neuron")`.
schnell is timestep+guidance distilled → **4 steps, guidance_scale=0.0**.

## TL;DR — the working configuration

- **Single NeuronCore**, no tensor-parallelism, no `torchrun`, no collectives.
- **Text encoders (CLIP + T5) run EAGER on the host CPU** — cheap, per-prompt; only the
  heavy compute goes to the device.
- **DiT transformer**: compile the **feed-forward submodules only** (`FLUX_COMPILE_MODE=ff_only`).
  **Attention stays EAGER** — compiling flux attention on this SDK produces a NaN latent
  (black image); see [Compilation findings](#compilation-findings).
- **VAE decoder in fp32** (bf16/fp16 overflows → NaN); latent/scheduler accumulation in fp32,
  transformer weights bf16 (mixed precision).
- **VAE nearest-upsample → NKI kernel** (neuronx-cc can't compile `aten.upsample_nearest2d`, [F139]).

## Performance (640×480, 4 steps, single core, trn3 `s-lnc2-trn3`)

| Compile mode | Attention | Image | Clean median | Notes |
|---|---|---|---|---|
| `none` (eager) | eager | ✅ correct | 5015 ms | baseline, no compile |
| **`ff_only`** (default) | eager | ✅ correct | **4770 ms** | **shipping config** — FF compiled, attn eager |
| `leaf` (attn+FF) | compiled | ❌ NaN/black | 4305 ms | compiling attn corrupts rotary |
| `per_block` (whole block) | compiled | ❌ NaN/black | 3533 ms | fastest but broken |

- Cold compile (`ff_only`, empty NEFF cache): **~290 s** warmup. Warm reload (cache hit): **~20 s**.
- Profiler overhead is ~2–5 % (clean vs profiled median), so the clean number is trustworthy.
- **The fast modes are the broken ones.** Attention is the dominant cost and must stay eager,
  so `ff_only` is only ~5 % faster than full eager. Closing that gap requires fixing the
  neuronx-cc attention miscompile (below) — a compiler-level issue, not a config change.

## Compilation findings (why the config is what it is)

Established by bisecting the compile boundary and probing the latent (`nan`/range) at each stage
(embeds → pre-decode latent → output image). Embeds were always finite; the fault localized to
the transformer denoise path:

1. **Whole-transformer compile → `[F139]` neuronx-cc terminated abnormally.** The flux DiT as a
   single graph exceeds the compiler's ~5M-instruction ceiling. Must compile at a finer boundary.
2. **Per-block / attn+FF compile → fully-NaN latent (`min=inf max=-inf`), black image.** Compiling
   a whole block — or just the **attention** submodule — corrupts the numerics: the rotary (cos/sin)
   embedding is applied *inside* attention, and the compiled attention NEFF mishandles it on this SDK.
3. **FF-only compile (`ff_only`) → finite latent, correct image.** Compiling only the feed-forward
   submodules (leaving attention + rotary + residual adds eager) keeps every NEFF under the ceiling
   AND preserves the numerics. This is the shipping config.
4. **Mixed precision required.** Full fp32 OOMs the 38.6 GB HBM (transformer+VAE ≈ 38.4 GB). Keep
   transformer weights bf16; run the latent/scheduler accumulation in fp32 (flux is flow-matching —
   bf16 velocity accumulation overflows → NaN). VAE decode in fp32 (overflow/quality). Achieved by
   feeding the pipeline fp32 `prompt_embeds` (→ fp32 latents) + a transformer forward-pre-hook that
   casts inputs back to bf16 at the model boundary.

**Open item:** attention compile producing NaN is a neuronx-cc bug (flux rotary-in-attention on
SDK 2.31). Fixing it would unlock the fast `per_block`/`leaf` path (~3.5 s). Until then, attention
runs eager.

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `FLUX_COMPILE_MODE` | `ff_only` | `ff_only` (ship) / `none` (eager) / `leaf`,`per_block` (broken — attn compiled) |
| `DTYPE` | `bfloat16` | transformer weight dtype (latent/scheduler forced fp32 in code) |
| `HEIGHT`/`WIDTH` | `480`/`640` | output size (NEFFs are shape-specific) |
| `NUM_STEPS` | `4` | schnell single-shot recipe |
| `GUIDANCE_SCALE` | `0.0` | schnell is guidance-distilled |
| `MAX_SEQ_LEN` | `256` | T5 prompt length |
| `MODEL_NAME` | `black-forest-labs/FLUX.1-schnell` | HF model id |

## Files
- `serve.py` — single-core FluxPipeline on Neuron: text encoders eager on host, FF submodules
  torch.compiled (attn eager), VAE fp32 + NKI upsampler, `/generate` `/health` `/readiness` on :8000.
- `bench.py` — headless one-shot benchmark (`serve.load()` + timed iters); prints `built=`/`median=`
  and saves the output PNG. Used with the neuron-3run-benchmark skill.
- `nki_upsample.py` — NKI kernel: nearest 2× upsample. Replaces `F.interpolate(mode="nearest")`
  (neuronx-cc [F139]).
- `test_nki_upsample.py` — kernel accuracy proof (neuron vs CPU `F.interpolate`, exact match).
- `flux-deploy.yaml` — k8s Deployment + NodePort Service (serves `python3 serve.py`).
- `flux-bench-job.yaml` — k8s Job (runs the 3-run benchmark once; survives compile failures).
- `requirements.txt` — runtime deps (torch/neuron/nki come from the SDK 2.31 base image).

## Validate the kernel first
```bash
python3 test_nki_upsample.py     # expect max|Δ|=0.0, ALL PASS
```

## Deploy (k8s)
```bash
kubectl apply -f flux-deploy.yaml
kubectl rollout status deployment/flux
# cold start ~5-6 min (deps + model download + FF compile); readiness gates traffic until warm.

kubectl get svc flux             # get the NodePort
curl -X POST http://<node>:<nodeport>/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"an astronaut riding a horse on Mars"}' | jq -r .image | base64 -d > out.png
```

## Benchmark
```bash
kubectl apply -f flux-bench-job.yaml
kubectl logs -f job/flux-bench 2>&1 | grep -E "PROBE|built=|median=|SAVED IMAGE"
# output PNG persists to the S3-backed PVC: /var/mdl/flux/runs/<pod>_<ts>.png
# set RUN_PROFILER=1 to enable the per-NEFF profiler stage (slow; off by default).
```
