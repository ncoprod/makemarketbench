import json
from argparse import Namespace

import pytest

from makemarket.provenance import file_sha256
from train.final_candidate import evaluate_last_dagger


def test_retained_validation_is_not_rewritten_and_rejects_changed_inputs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'train/runs/qwen3.5-4b-dagger'
    source.mkdir(parents=True)
    checkpoint = source / '0000500_adapters.safetensors'
    checkpoint.write_bytes(b'frozen adapter')
    training = tmp_path / 'results/training/qwen35_dagger.json'
    training.parent.mkdir(parents=True)
    training.write_text(json.dumps({'status': 'validation_plateau'}))
    validation = tmp_path / 'results/validation/qwen35-dagger-last'
    validation.mkdir(parents=True)
    data = validation / 'leaderboard_dice.json'
    data.write_bytes(b'frozen validation')
    manifest = validation / 'cohorts.json'
    manifest.write_text(json.dumps({'adapter_sha256': file_sha256(checkpoint),
                                    'provenance': {'model_revision': 'fixed'},
                                    'input_sha256': {str(data): file_sha256(data)}}))
    before = {p: p.read_bytes() for p in (manifest, data)}
    evaluate_last_dagger(Namespace(revision='fixed'))
    assert all(p.read_bytes() == content for p, content in before.items())
    data.write_bytes(b'changed validation')
    with pytest.raises(AssertionError, match='Retained validation changed'):
        evaluate_last_dagger(Namespace(revision='fixed'))
