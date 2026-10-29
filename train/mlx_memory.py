"""Bound MLX's training-only gated-delta recurrence memory.

Use the upstream chunk-parallel training patch for scalar, unmasked gates.
Checkpoint the reference recurrence for other cases. Inference is unchanged.
"""


def enable_gated_delta_training(chunk_size=64):
    import mlx.core as mx
    from gated_delta_chunkwise import gated_delta_chunkwise
    from mlx_lm.models import gated_delta

    from makemarket.provenance import file_sha256

    original = gated_delta.gated_delta_ops

    def chunked(q, k, v, g, beta, state=None, mask=None):
        if mask is None and g.ndim == 3:
            return gated_delta_chunkwise(q, k, v, g, beta, state, chunk_size=chunk_size)
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
    return {"chunk_size": chunk_size, "source_sha256": file_sha256(__file__),
            "implementation": "parallel scalar/unmasked; checkpointed sequential otherwise",
            "port_sha256": file_sha256(gated_delta_chunkwise.__code__.co_filename)}
