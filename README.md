# FLUX.1-schnell on AWS Trainium (PyTorch-native, torch.compile)

Serve [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
as a text→image endpoint on Trainium, using **PyTorch-native `torch.compile(backend="neuron")`**
(no `neuronx_distributed_inference`). Runs on the SDK 2.31 `native-pytorch` DLC image.

FLUX.1-schnell is **timestep- and guidance-distilled**: generate in **4 steps** with
**guidance_scale=0.0** (no CFG).

## Components
- `serve.py` — loads `FluxPipeline`, moves to `neuron`, `torch.compile`s the DiT transformer,
  warms up (first-call NEFF compile), serves `/generate` (FastAPI, port 8000).
- `requirements.txt` — runtime deps installed at container start via `uv` (torch/neuron come
  from the base image, not pinned here).
- `flux-schnell-deploy.yaml` — k8s Service + Deployment: trn3 node, `s-lnc2-trn3` claim
  (1 device), SDK 2.31 image, HF weights cached on the S3-backed PVC.

## Endpoints
- `GET /health` — liveness/startup
- `GET /readiness` — 200 once the model is compiled + warmed
- `POST /generate` — `{"prompt": "...", "num_inference_steps": 4, "seed": 0}` → `{image: <base64 png>, latency_s}`

## Deploy
```bash
kubectl apply -f flux-schnell-deploy.yaml
kubectl get pods -l app=flux-schnell -w        # wait for 1/1 (first compile takes minutes)
kubectl port-forward svc/flux-schnell 8000:8000
curl -s localhost:8000/generate -H 'content-type: application/json' \
  -d '{"prompt":"a fox in a forest, cinematic"}' | python3 -c 'import sys,json,base64;open("out.png","wb").write(base64.b64decode(json.load(sys.stdin)["image"]))'
```

## Notes / open items
- **First bring-up may surface torch.compile-on-Neuron issues in the Flux graph** (dynamic
  shapes, unsupported ops). Start with `COMPILE=1` on the transformer only; if a compile error
  appears, set `COMPILE=0` (eager) to isolate, then reintroduce compilation piece by piece.
- Single device (schnell bf16 ~24GB fits in 144GB HBM). If you need TP/CP across devices,
  switch the claim to `m-lnc2-trn3` and add device-mesh sharding in `serve.py`.
- `HEIGHT/WIDTH` are fixed per pod (compiled shape). Change env + restart to recompile.
