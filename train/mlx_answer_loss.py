"""Avoid projecting masked prompt positions through Qwen3.5's large output head."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.trainer import iterate_batches


def answer_batches(*args, width=128, pad_training_to=0, **kwargs):
    for batch, lengths in iterate_batches(*args, **kwargs):
        first_retained_step = mx.maximum(1, lengths[:, 1] - width + 1)
        if mx.any(lengths[:, 0] < first_retained_step).item():
            raise ValueError("Answer window would discard supervised tokens; increase its width")
        if kwargs.get("loop") and pad_training_to:
            if batch.shape[1] > pad_training_to:
                raise ValueError("Training batch exceeds its fixed padding length")
            batch = mx.pad(batch, [(0, 0), (0, pad_training_to - batch.shape[1])])
        yield batch, lengths


def answer_loss(model, batch, lengths, width=128):
    inputs = batch[:, :-1]
    width = min(width, inputs.shape[1])
    end = mx.minimum(lengths[:, 1], inputs.shape[1])
    steps = mx.maximum(1, end - width + 1)[:, None] + mx.arange(width)
    targets = mx.take_along_axis(batch, steps, axis=1)
    text = model.language_model
    hidden = mx.take_along_axis(text.model(inputs), (steps - 1)[..., None], axis=1)
    logits = (text.model.embed_tokens.as_linear(hidden) if text.args.tie_word_embeddings
              else text.lm_head(hidden))
    mask = (steps >= lengths[:, 0:1]) & (steps <= lengths[:, 1:])
    losses = nn.losses.cross_entropy(logits, targets) * mask
    tokens = mask.sum()
    return losses.astype(mx.float32).sum() / tokens, tokens
