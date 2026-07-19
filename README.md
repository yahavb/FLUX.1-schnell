# FLUX.1-schnell on AWS Trainium (PyTorch-native, torch.compile)

Serve [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
as a text→image endpoint on Trainium (trn3), PyTorch-native via `torch.compile(backend="neuron")`.
schnell is timestep+guidance distilled → **4 steps, guidance_scale=0.0**.

## Files
- `serve.py` — FluxPipeline on Neuron; FFN-only tensor-parallel transformer (torchrun,
  rank 0 serves FastAPI); VAE nearest-upsampler replaced by a NKI kernel; `/generate` on :8000.
- `nki_upsample.py` — NKI 0.4.0 kernel: nearest 2x upsample. Replaces
  `F.interpolate(mode="nearest")`, which neuronx-cc can't compile ([F139]).
- `test_nki_upsample.py` — accuracy proof (kernel on neuron vs CPU `F.interpolate`, exact match).
- `requirements.txt` — runtime deps (torch/neuron/nki come from the base image).

## Validate the kernel first
```bash
python3 test_nki_upsample.py     # expect max|Δ|=0.0, ALL PASS
```

## Deploy (k8s): see flux-schnell-deploy.yaml in the cluster repo — it clones this repo.
