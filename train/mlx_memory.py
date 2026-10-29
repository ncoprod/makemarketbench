"""Checkpoint MLX's training-only gated-delta recurrence in short chunks.

The upstream fallback unrolls every token while retaining large recurrent states
for backward. Rematerialising chunks preserves the recurrence and its gradients.
Inference continues to use the upstream Metal kernel.
"""


def checkpoint_gated_delta(chunk_size=64):
    import mlx.core as mx
    from mlx_lm.models import gated_delta

    from makemarket.provenance import file_sha256

    original = gated_delta.gated_delta_ops

    def chunked(q, k, v, g, beta, state=None, mask=None):
        outputs = []
        for start in range(0, q.shape[1], chunk_size):
            stop = start + chunk_size
            output, state = mx.checkpoint(original)(
                q[:, start:stop], k[:, start:stop], v[:, start:stop],
                g[:, start:stop], beta[:, start:stop], state,
                None if mask is None else mask[:, start:stop])
            outputs.append(output)
        return mx.concatenate(outputs, axis=1), state

    gated_delta.gated_delta_ops = chunked
    return {"chunk_size": chunk_size, "source_sha256": file_sha256(__file__)}
