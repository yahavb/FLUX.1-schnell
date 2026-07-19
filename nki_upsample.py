"""
NKI nearest-neighbor 2x spatial upsample (upsample_nearest2d, scale_factor=2).

Replaces F.interpolate(mode="nearest") in the diffusers Flux VAE decoder, which
neuronx-cc cannot compile ([F139] crash on aten.upsample_nearest2d).

NKI 0.4.0 API (import nki / nisa.dma_copy / nisa.tensor_copy / .ap) — matching the
KaenaNeuronKernelLibrary convention (the older nl.load/nl.mgrid Beta-1 style does not
compile on SDK 2.30+).

Algorithm (NCHW, scale 2, nearest = pure copy):
  Flatten to rows R = N*C*H, cols W. Output is [2R, 2W]. Output rows 2r and 2r+1 both
  equal the W-upsampled input row r — because nc*2H + 2i = 2*(nc*H + i) = 2r, so the
  H-doubling maps input row r cleanly to output rows 2r/2r+1 with no H-boundary
  wraparound. W-upsample duplicates each column into even/odd positions.

`.ap(pattern=[[stride_elems, num], ...], offset=elems)` describes strided access; used
here to scatter columns (stride 2) and rows (stride 2*W2) so a single copy fans out.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def upsample_nearest_2x_fwd(src_arr):
    """Nearest 2x upsample of an NCHW tensor. src_arr: HBM (N,C,H,W). Returns HBM (N,C,2H,2W)."""
    N, C, H, W = src_arr.shape
    R = N * C * H
    W2 = W * 2

    in_flat = src_arr.reshape((R, W))
    output = nl.ndarray((N, C, H * 2, W * 2), dtype=src_arr.dtype, buffer=nl.shared_hbm)
    out_flat = output.reshape((R * 2, W2))

    TILE_P = 128
    n_tiles = (R + TILE_P - 1) // TILE_P
    for t in nl.affine_range(n_tiles):
        r0 = t * TILE_P
        p = min(TILE_P, R - r0)

        # Load p input rows [p, W].
        in_tile = nl.ndarray((p, W), dtype=src_arr.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=in_tile, src=in_flat[r0:r0 + p, 0:W])

        # W-upsample -> [p, 2W]: duplicate each column into even (2j) and odd (2j+1) slots.
        wide = nl.ndarray((p, W2), dtype=src_arr.dtype, buffer=nl.sbuf)
        # even output cols 0,2,4,...: dst stride 2 in the free dim.
        nisa.tensor_copy(dst=wide.ap(pattern=[[W2, p], [2, W]]), src=in_tile)
        # odd output cols 1,3,5,...: same, shifted by 1 element.
        nisa.tensor_copy(dst=wide.ap(pattern=[[W2, p], [2, W]], offset=1), src=in_tile)

        # H-upsample: write `wide` to output rows 2*(r0+k) and 2*(r0+k)+1.
        # out_flat rows are contiguous blocks of W2; partition stride is 2*W2 (skip a row).
        nisa.dma_copy(dst=out_flat.ap(pattern=[[2 * W2, p], [1, W2]], offset=2 * r0 * W2),
                      src=wide)
        nisa.dma_copy(dst=out_flat.ap(pattern=[[2 * W2, p], [1, W2]], offset=(2 * r0 + 1) * W2),
                      src=wide)

    return output


def upsample_nearest2x(x):
    """torch-callable nearest 2x upsample. x: NCHW tensor on the neuron device."""
    return upsample_nearest_2x_fwd(x.contiguous())
