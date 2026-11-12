"""Experiment protocol checks using synthetic images, never model outputs."""
import csv
from dataclasses import replace
import json

from PIL import Image
import pytest

from experiments.run_ablation import execute, prepare
from experiments.summarize_results import summarize
from material_transfer.config import TransferConfig
from material_transfer.records import read_manifest, save_run


@pytest.fixture
def experiment(tmp_path):
    for name in ('target', 'exemplar', 'mask', 'depth'):
        Image.new('RGB', (16, 16), 'white').save(tmp_path / f'{name}.png')
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('pair_id,target,exemplar,mask,depth,material_prompt,seed\n'
                        'p01,target.png,exemplar.png,mask.png,depth.png,ceramic,17\n')
    return manifest, tmp_path / 'out'


def fake_runs(plan, output):
    sample = plan['samples'][0]
    for job in plan['jobs']:
        method = job['method']
        artifacts = {k: Image.new('RGB', (16, 16), 'white') for k in
                     ('result', 'target', 'exemplar', 'target_mask', 'depth', 'init', 'reference')}
        if method in 'BD':
            artifacts['reference_mask'] = Image.new('L', (16, 16), 255)
        if method in 'CD':
            artifacts['attention_mask'] = Image.new('L', (16, 16), 150)
        cfg = replace(TransferConfig(**plan['configs'][method]), seed=job['seed'])
        save_run(output / job['directory'] / sample['pair_id'], artifacts, cfg,
                 dict(segmentation_applied=method in 'BD', attention_applied=method in 'CD'),
                 {k: v for k, v in sample.items() if k != 'sha256'})


def test_preparation_pairs_seeds_and_preserves_notes(experiment):
    manifest, output = experiment
    plan = prepare(manifest, output, [42, 123])
    assert len(plan['jobs']) == 8
    assert read_manifest(output / 'prepared/seed_123.csv')[0].seed == 123
    settings = [{k: v for k, v in cfg.items() if k != 'method'} for cfg in plan['configs'].values()]
    assert all(cfg == settings[0] for cfg in settings)
    notes = output / 'observations.csv'
    notes.write_text('manual observations', encoding='utf-8')
    prepare(manifest, output, [42, 123])
    assert notes.read_text(encoding='utf-8') == 'manual observations'
    with pytest.raises(ValueError, match='differs'):
        prepare(manifest, output, [7])


def test_fixed_inputs_required_before_writing(experiment):
    manifest, output = experiment
    manifest.write_text(manifest.read_text().replace('mask.png,depth.png', 'mask.png,'))
    with pytest.raises(ValueError, match='fixed mask and depth'):
        prepare(manifest, output, [42])
    assert not output.exists()


def test_pending_is_not_complete(experiment):
    manifest, output = experiment
    prepare(manifest, output, [42])
    assert summarize(output) == dict(complete=0, pending=4, invalid=0)
    with Image.open(output / 'comparisons/p01_seed_42.png') as image:
        assert image.size == (1536, 290)


def test_complete_and_refuse_overwrite(experiment):
    manifest, output = experiment
    plan = prepare(manifest, output, [42])
    fake_runs(plan, output)
    assert summarize(output) == dict(complete=4, pending=0, invalid=0)
    with pytest.raises(FileExistsError):
        execute(plan, output)


def test_resume_skips_only_valid_completed_pairs(experiment, monkeypatch):
    manifest, output = experiment
    manifest.write_text(manifest.read_text() + 'p02,target.png,exemplar.png,mask.png,depth.png,ceramic,17\n')
    plan = prepare(manifest, output, [42])
    fake_runs(plan, output)  # Only p01 has been generated.
    calls = []
    import subprocess
    original_run = subprocess.run
    def run(command, **kwargs):
        if '--manifest' not in command:
            return original_run(command, **kwargs)
        pending = read_manifest(command[command.index('--manifest') + 1])
        calls.append([p.pair_id for p in pending])
    monkeypatch.setattr('experiments.run_ablation.subprocess.run', run)
    execute(plan, output, resume=True)
    assert calls == [['p02']] * 4


def test_resume_refuses_mixed_source_code(experiment):
    manifest, output = experiment
    plan = prepare(manifest, output, [42])
    fake_runs(plan, output)
    path = output / 'seed_42/A/p01/run.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    record['package_source_sha256'] = 'different-code'
    path.write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(ValueError, match='incompatible'):
        execute(plan, output, resume=True)


@pytest.mark.parametrize('change', ['config', 'flags', 'missing_artifact', 'escaped_artifact'])
def test_invalid_runs_are_not_accepted(experiment, change):
    manifest, output = experiment
    plan = prepare(manifest, output, [42])
    fake_runs(plan, output)
    directory = output / 'seed_42/D/p01'
    path = directory / 'run.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    if change == 'config':
        record['config']['seed'] = 99
    elif change == 'flags':
        record['timings']['attention_applied'] = False
    elif change == 'missing_artifact':
        (directory / record['artifacts']['attention_mask']).unlink()
    else:
        record['artifacts']['result'] = str(manifest.parent / 'target.png')
    path.write_text(json.dumps(record), encoding='utf-8')
    assert summarize(output) == dict(complete=3, pending=0, invalid=1)


def test_shared_preprocessing_mismatch(experiment):
    manifest, output = experiment
    plan = prepare(manifest, output, [42])
    fake_runs(plan, output)
    Image.new('RGB', (16, 16), 'black').save(output / 'seed_42/B/p01/init.png')
    assert summarize(output) == dict(complete=0, pending=0, invalid=4)


def test_input_modified_after_plan(experiment):
    manifest, output = experiment
    plan = prepare(manifest, output, [42])
    fake_runs(plan, output)
    Image.new('RGB', (16, 16), 'black').save(manifest.parent / 'target.png')
    assert summarize(output) == dict(complete=0, pending=0, invalid=4)
