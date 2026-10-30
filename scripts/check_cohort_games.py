"""Check completed cohorts against the canonical seeded games without scoring."""

import argparse
import hashlib
import json
from pathlib import Path

from makemarket.games import make_game
from makemarket.provenance import file_sha256, receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--out', default='results/final/game_comparability.json')
    args = ap.parse_args()
    record = {'provenance': receipt(vars(args)), 'cohorts': {}, 'incomplete_cohorts': [],
              'source_sha256': file_sha256(__file__),
              'game_source_sha256': {str(p): file_sha256(p) for p in sorted(Path('src/makemarket/games').glob('*.py'))},
              'checks': ['200 distinct seeds 200-399', 'six rounds per episode',
                         'hidden value and public reveals match canonical seeded game'],
              'interpretation': 'Structural comparability only; no model scores inspected.'}
    games = {name: make_game(name) for name in ('dice', 'cards')}
    paths = set()
    for source in map(Path, args.inputs):
        assert source.exists(), f"Missing cohort input: {source}"
        paths.update([source] if source.is_file() else source.rglob('*.jsonl'))
    for path in sorted(paths):
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        if sorted(r['seed'] for r in rows) != list(range(200, 400)):
            record['incomplete_cohorts'].append(str(path))
            continue
        for row in rows:
            canonical = games[row['game']].episode(row['seed'])
            assert len(row['rounds']) == canonical.n_rounds == 6, (path, row['seed'])
            assert row['value'] == canonical.value, (path, row['seed'], 'hidden value')
            assert [r['t'] for r in row['rounds']] == list(range(6)), (path, row['seed'], 'round indices')
            assert [r['reveal'] for r in row['rounds']] == canonical.reveals, (path, row['seed'], 'reveals')
        record['cohorts'][str(path)] = {'episodes': len(rows), 'game': rows[0]['game'],
                                      'player': rows[0]['player'], 'sha256': hashlib.sha256(data).hexdigest()}
    assert record['cohorts'], 'No complete held-out cohorts found'
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + '\n')
    print(f"Canonical game checks pass for {len(record['cohorts'])} complete cohorts.")


if __name__ == '__main__':
    main()
