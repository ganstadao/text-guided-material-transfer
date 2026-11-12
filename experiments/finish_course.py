"""Validate frozen inputs and complete machine evaluation + a blank blind-review pack."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finish(data, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    frozen = json.loads((data / 'frozen_inputs.json').read_text(encoding='utf-8'))
    for path, expected in frozen['source_inputs_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Source image changed: {path}')
    for path, expected in frozen['artifacts_sha256'].items():
        if digest(data / path) != expected:
            raise ValueError(f'Frozen mask/depth/GT changed: {path}')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1',
        HF_HOME=str(ROOT / 'models/hf-cache'), TORCH_HOME=str(ROOT / 'models/torch-cache'))
    def run(*args):
        subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=env, check=True)
    run('experiments/summarize_results.py', output, '--require-complete')
    run('evaluation/compute_metrics.py', '--manifest', data / 'manifest.csv', '--outputs', output / 'seed_42',
        '--gt-masks', data / 'gt_by_pair', '--out', output / 'metrics.csv', '--lpips',
        '--clip-model', ROOT / 'models/local/clip-vit-base-patch32')
    run('evaluation/summarize_ablation.py', '--metrics', output / 'metrics.csv',
        '--strata', data / 'strata.json', '--output', output / 'evaluation')
    blind = output / 'blind_review'
    if not blind.exists():
        run('evaluation/prepare_blind_review.py', '--manifest', data / 'manifest.csv',
            '--outputs', output / 'seed_42', '--output', blind)
    else:
        with (blind / 'private/mapping.csv').open(encoding='utf-8-sig', newline='') as stream:
            for item in csv.DictReader(stream):
                source = output / 'seed_42' / item['group'] / item['pair_id'] / 'result.png'
                if digest(source) != item['result_sha256']:
                    raise ValueError('Existing blind package differs from source results')
    report = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    human_path = output / 'human_ratings.json'
    human = json.loads(human_path.read_text(encoding='utf-8')) if human_path.exists() else {}
    report.update(automatic_evaluation_complete=True,
        human_ratings_complete=human.get('completed', 0) >= report['complete'] and human.get('incomplete', 1) == 0,
        independent_blind_review_complete=human.get('independent_blind_review_complete', False),
        rating_provenance=human.get('rating_provenance', []),
        reference_gt_status=json.loads((data / 'gt_by_pair/annotations.json').read_text())['status'],
        rater_package=str(blind / 'rater_package'), private_mapping=str(blind / 'private/mapping.csv'))
    (output / 'completion.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Generation, machine metrics, paired summaries and blank blind-review package are complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/course_fixed_v1')
    parser.add_argument('--output', default='outputs/course-1024-v1')
    args = parser.parse_args()
    finish(args.data, args.output)
