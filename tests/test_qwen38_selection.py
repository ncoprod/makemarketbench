import json
import sys

import pytest

from makemarket.provenance import file_sha256
from scripts import select_qwen38


@pytest.mark.parametrize('batch_size', [1, 4])
def test_newest_selection_validates_adapter_identity_and_freezes_ties(monkeypatch, tmp_path, batch_size):
    runtime_path = tmp_path / 'runtime.json'
    runtime = {'seed': 0, 'model_revision': '8aa5f05d26b7205477066e1449e0af13f762a299'}
    runtime_path.write_text(json.dumps(runtime))
    for method, directory in [('sft', 'warmstart_adapter'), ('grpo', 'lora_adapter')]:
        adapter = tmp_path / directory / 'adapter_model.safetensors'
        adapter.parent.mkdir()
        adapter.write_bytes(method.encode())
        family = 'qwen3.8-27b-batch4' if batch_size == 4 else 'qwen3.8-27b'
        label = f'{family}-{method}'
        root = tmp_path / label / 'results'
        root.mkdir(parents=True)
        for game in ('dice', 'cards'):
            record = {'seeds': list(range(80000, 80100)),
                      'provenance': {'config': {'revision': '8aa5f05d26b7205477066e1449e0af13f762a299',
                                                'adapter_sha256': file_sha256(adapter),
                                                'eval_batch_size': batch_size, 'seed': 0}},
                      'rows': [{'player': label, 'episodes': 100, 'pct_of_bayes': 80 if game == 'dice' else 5}]}
            (root / f'leaderboard_{game}.json').write_text(json.dumps(record))
    out = tmp_path / 'selection.json'
    monkeypatch.setattr(sys, 'argv', ['select_qwen38.py', '--training-root', str(tmp_path),
                                     '--validation-root', str(tmp_path), '--test-root', str(tmp_path / 'test'),
                                     '--out', str(out), '--batch-size', str(batch_size)])
    adapter.write_bytes(b'wrong adapter')
    with pytest.raises(AssertionError):
        select_qwen38.main()
    assert not out.exists()
    adapter.write_bytes(b'grpo')
    validation_path = root / 'leaderboard_cards.json'
    validation = json.loads(validation_path.read_text())
    validation['provenance']['config']['eval_batch_size'] = 4 if batch_size == 1 else 1
    validation_path.write_text(json.dumps(validation))
    with pytest.raises(AssertionError, match='Wrong registered inference variant'):
        select_qwen38.main()
    assert not out.exists()
    validation['provenance']['config']['eval_batch_size'] = batch_size
    validation_path.write_text(json.dumps(validation))
    runtime['seed'] = 1
    runtime_path.write_text(json.dumps(runtime))
    with pytest.raises(AssertionError, match='primary training seed'):
        select_qwen38.main()
    assert not out.exists()
    runtime['seed'] = 0
    runtime_path.write_text(json.dumps(runtime))
    select_qwen38.main()
    frozen = out.read_bytes()
    assert json.loads(frozen)['method'] == 'sft'
    assert json.loads(frozen)['final_candidate_family'] == (11 if batch_size == 4 else 9)
    assert json.loads(frozen)['training_runtime_sha256'] == file_sha256(runtime_path)
    with pytest.raises(AssertionError, match='already frozen'):
        select_qwen38.main()
    assert out.read_bytes() == frozen
