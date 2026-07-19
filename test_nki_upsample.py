"""
ACCURACY PROOF for the NKI nearest-2x upsample kernel.

Nearest upsample is a pure copy (each pixel repeated 2x2), so the kernel output must
match torch.nn.functional.interpolate(mode="nearest") EXACTLY (max|Δ| == 0.0).

Per skill + rolling_forcing/verify_*.py convention: reference computed on CPU, kernel
run on the neuron device, result moved back to CPU for comparison.

Run in the SDK 2.31 native-pytorch container on a trn3 node:
    python3 test_nki_upsample.py
"""
import torch
import torch.nn.functional as F

from nki_upsample import upsample_nearest_2x_fwd


def check(N, C, H, W, dtype=torch.bfloat16):
    x_cpu = torch.randn(N, C, H, W, dtype=dtype)

    # CPU reference (exact — nearest is a copy).
    ref = F.interpolate(x_cpu.float(), scale_factor=2, mode="nearest").to(dtype)

    # NKI kernel on neuron.
    out = upsample_nearest_2x_fwd(x_cpu.to("neuron")).to("cpu")

    shape_ok = tuple(out.shape) == tuple(ref.shape)
    max_abs = (out.float() - ref.float()).abs().max().item() if shape_ok else float("nan")
    ok = shape_ok and max_abs == 0.0
    print(f"  {N}x{C}x{H}x{W} {dtype}: out={tuple(out.shape)} ref={tuple(ref.shape)} "
          f"max|Δ|={max_abs:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    torch.manual_seed(0)
    # Representative Flux VAE decoder upsample shapes (480x640 output).
    cases = [(2, 64, 16, 16), (1, 512, 60, 80), (1, 256, 120, 160), (1, 128, 240, 320)]
    results = [check(*c) for c in cases]
    print("ALL PASS" if all(results) else "SOME FAILED")
