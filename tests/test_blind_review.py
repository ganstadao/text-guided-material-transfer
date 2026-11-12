import csv
import json
from PIL import Image
import pytest
from evaluation.prepare_blind_review import prepare
from evaluation.summarize_ratings import summarize


def test_blind_package_has_no_method_labels_or_fabricated_ratings(tmp_path):
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('pair_id\np01\n')
    for method in 'ABCD':
        directory = tmp_path / 'runs' / method / 'p01'
        directory.mkdir(parents=True)
        artifacts = {}
        for key in ('target', 'exemplar', 'target_mask', 'result'):
            Image.new('RGB', (16, 16), 'red').save(directory / f'{key}.png')
            artifacts[key] = f'{key}.png'
        (directory / 'run.json').write_text(json.dumps(dict(config={'method': method}, artifacts=artifacts)))
    package = prepare(manifest, tmp_path / 'runs', tmp_path / 'blind')
    rows = list(csv.DictReader((package / 'ratings.csv').open(encoding='utf-8-sig')))
    assert len(rows) == 4 and len({r['item_id'] for r in rows}) == 4
    assert all('group' not in r and not r['material_match'] for r in rows)
    assert not (package / 'mapping.csv').exists()
    mapping = tmp_path / 'blind/private/mapping.csv'
    summary = tmp_path / 'scores.json'
    assert summarize([package / 'ratings.csv'], mapping, summary) == 1
    assert json.loads(summary.read_text())['groups']['A']['shape']['mean'] is None
    with pytest.raises(FileExistsError):
        prepare(manifest, tmp_path / 'runs', tmp_path / 'blind')
