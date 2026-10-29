# Copyright © 2025 Apple Inc.
# Ported from adityak74's MLX-LM PR #1870 at
# 3b04b9f1992fb83c4210228f7eb1c9dc72827e49. Algorithm unchanged;
# type annotations modernised for this repository.
# MIT licence: third_party/mlx_lm_LICENSE.txt.


import mlx.core as mx


def _unit_tri_inv(m: mx.array) -> mx.array:
    """Inverse of a unit lower-triangular matrix, built from matmuls.

    ``mx.linalg.tri_inv`` is CPU-only and has no VJP, so it cannot appear in a
    training graph. The 2x2 block identity

        [A 0; C B]^-1 = [A^-1 0; -B^-1 C A^-1, B^-1]

    gives the inverse in log2(n) levels of batched matmuls, which autodiff
    handles natively and which parallelizes over the batch.
    """
    n = m.shape[-1]
    if n == 1:
        return mx.ones_like(m)
    h = n // 2
    a, c, b = m[..., :h, :h], m[..., h:, :h], m[..., h:, h:]
    a_inv, b_inv = _unit_tri_inv(a), _unit_tri_inv(b)
    lower = -(b_inv @ c @ a_inv)
    zeros = mx.zeros_like(mx.swapaxes(lower, -1, -2))
    return mx.concatenate(
        [
            mx.concatenate([a_inv, zeros], axis=-1),
            mx.concatenate([lower, b_inv], axis=-1),
        ],
        axis=-2,
    )


def gated_delta_chunkwise(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array | None = None,
    mask: mx.array | None = None,
    chunk_size: int = 64,
) -> tuple[mx.array, mx.array]:
    """Chunkwise-parallel gated delta rule, for training.

    Equivalent to :func:`gated_delta_ops` but splits the sequence into chunks of
    ``chunk_size`` and evaluates each chunk with matmuls, so autograd retains
    O(T / chunk_size) states instead of one per timestep.

    Writing the recurrence as ``S_t = g_t S_{t-1} + u_t k_t^T`` with
    ``u_t = b_t (v_t - g_t S_{t-1} k_t)`` telescopes within a chunk::

        S_j = G_j S_0 + sum_{i<=j} (G_j / G_i) u_i k_i^T,  G_j = prod_{m<=j} g_m

    and the ``u_i`` satisfy a unit lower-triangular system solved once per
    chunk. Quantities are carried as ratios ``G_j / G_i`` (bounded by 1 for
    i <= j) rather than as ``G_j``, which underflows on strongly decaying heads
    and sends the intermediate writes to infinity.

    Scalar gating only; vectorized gating and masks fall back to the caller.

    Shapes match :func:`gated_delta_ops`.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    out_dtype = q.dtype
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    if (repeat_factor := Hv // Hk) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    # [B, T, H, D] -> [B, H, T, D]; accumulate in fp32 as the kernels do
    q, k, v = (mx.swapaxes(x, 1, 2).astype(mx.float32) for x in (q, k, v))
    g = mx.swapaxes(g, 1, 2).astype(mx.float32)
    beta = mx.swapaxes(beta, 1, 2).astype(mx.float32)

    pad = (-T) % chunk_size
    if pad:
        # padding must neither decay the state (g = 1) nor write to it (beta = 0)
        widths = [(0, 0), (0, 0), (0, pad), (0, 0)]
        q, k, v = (mx.pad(x, widths) for x in (q, k, v))
        g = mx.pad(g, widths[:3], constant_values=1.0)
        beta = mx.pad(beta, widths[:3])

    T_pad = q.shape[2]
    n_chunks = T_pad // chunk_size
    shape = (B, Hv, n_chunks, chunk_size)
    q, k, v = (x.reshape(*shape, -1) for x in (q, k, v))
    g, beta = g.reshape(shape), beta.reshape(shape)

    cumulative = mx.cumsum(mx.log(mx.maximum(g, 1e-30)), axis=-1)
    causal = mx.tril(mx.ones((chunk_size, chunk_size), dtype=mx.bool_))
    strict = mx.tril(mx.ones((chunk_size, chunk_size), dtype=mx.bool_), -1)
    eye = mx.eye(chunk_size, dtype=mx.float32)

    # ratios[j, i] = G_j / G_i for i <= j, zero above the diagonal
    deltas = cumulative[..., :, None] - cumulative[..., None, :]
    ratios = mx.where(causal, mx.exp(mx.where(causal, deltas, -1e30)), 0.0)
    gammas = mx.exp(cumulative)

    ys = []
    s = state.astype(mx.float32)
    for c in range(n_chunks):
        q_c, k_c, v_c = q[:, :, c], k[:, :, c], v[:, :, c]
        beta_c, gamma_c, ratio_c = beta[:, :, c], gammas[:, :, c], ratios[:, :, c]
        k_t = mx.swapaxes(k_c, -1, -2)

        system = eye + mx.where(strict, beta_c[..., :, None] * (k_c @ k_t) * ratio_c, 0.0)
        u = _unit_tri_inv(system) @ (
            beta_c[..., None] * (v_c - gamma_c[..., None] * (k_c @ mx.swapaxes(s, -1, -2)))
        )

        attn = mx.where(causal, (q_c @ k_t) * ratio_c, 0.0)
        ys.append(gamma_c[..., None] * (q_c @ mx.swapaxes(s, -1, -2)) + attn @ u)

        tail = ratio_c[..., -1, :]  # G_C / G_i
        s = gamma_c[..., -1, None, None] * s + mx.swapaxes(tail[..., None] * u, -1, -2) @ k_c

    y = mx.stack(ys, axis=2).reshape(B, Hv, T_pad, Dv)
    if pad:
        y = y[:, :, :T]
    return mx.swapaxes(y, 1, 2).astype(out_dtype), s


