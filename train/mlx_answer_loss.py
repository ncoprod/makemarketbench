"""Avoid projecting masked prompt positions through Qwen3.5's large output head."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.tuner.trainer import iterate_batches


def answer_batches(*args, width=128, **kwargs):
    for batch, lengths in iterate_batches(*args, **kwargs):
        first_retained_step = max(1, batch.shape[1] - width)
        if mx.any(lengths[:, 0] < first_retained_step).item():
            raise ValueError("Answer window would discard supervised tokens; increase its width")
        yield batch, lengths


def answer_loss(model, batch, lengths, width=128):
    inputs, targets = batch[:, :-1], batch[:, 1:][:, -width:]
    text = model.language_model
    hidden = text.model(inputs)[:, -width:]
    logits = (text.model.embed_tokens.as_linear(hidden) if text.args.tie_word_embeddings
              else text.lm_head(hidden))
    steps = mx.arange(1, inputs.shape[1] + 1)[-width:]
    mask = (steps >= lengths[:, 0:1]) & (steps <= lengths[:, 1:])
    losses = nn.losses.cross_entropy(logits, targets) * mask
    tokens = mask.sum()
    return losses.astype(mx.float32).sum() / tokens, tokens
