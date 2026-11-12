"""Prepare fixed-input A/B/C/D experiments; --execute enables inference."""
import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from material_transfer.config import TransferConfig
from material_transfer.images import depth_image, load_mask, load_rgb
from material_transfer.records import read_manifest


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(manifest, output, seeds, config=None):
    """Validate inputs before writing anything. Never load model weights."""
    output = Path(output).resolve()
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s >= 2**63 for s in seeds):
        raise ValueError('Provide distinct seeds in [0, 2**63)')
    cfg = config or TransferConfig()
    samples = []
    for pair in read_manifest(manifest):
        if pair.mask is None or pair.depth is None:
            raise ValueError(f'{pair.pair_id}: fixed mask and depth are required for ablation')
        if not pair.material_prompt.strip():
            raise ValueError(f'{pair.pair_id}: material_prompt is required for B/C/D')
        target = load_rgb(pair.target)
        load_rgb(pair.exemplar)
        load_mask(pair.mask, target.size)
        depth_image(pair.depth, target.size)
        sample = asdict(pair)
        sample.pop('seed')  # Experiment seeds replace manifest seeds in all four groups.
        for key in ('target', 'exemplar', 'mask', 'depth'):
            sample[key] = str(sample[key])
        sample['sha256'] = {key: file_hash(sample[key]) for key in ('target', 'exemplar', 'mask', 'depth')}
        samples.append(sample)
    configs = {m: asdict(replace(cfg, method=m)) for m in 'ABCD'}
    jobs = [dict(seed=s, method=m, directory=f'seed_{s}/{m}') for s in seeds for m in 'ABCD']
    plan = dict(schema_version=1, configs=configs, seeds=seeds, samples=samples, jobs=jobs)
    plan_path = output / 'experiment_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
        raise ValueError('Existing experiment plan differs; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')
    prepared = output / 'prepared'
    prepared.mkdir(exist_ok=True)
    for method, values in configs.items():
        (prepared / f'{method}.json').write_text(json.dumps(values, indent=2), encoding='utf-8')
    fields = ['pair_id', 'target', 'exemplar', 'mask', 'depth', 'material_prompt', 'seed']
    for seed in seeds:
        with (prepared / f'seed_{seed}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({**{k: sample[k] for k in fields if k != 'seed'}, 'seed': seed} for sample in samples)
    review = output / 'observations.csv'
    if not review.exists():
        with review.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['pair_id', 'seed', 'reference_type', 'B_vs_A', 'C_vs_A',
                             'D_vs_B', 'D_vs_C', 'material_match', 'background_structure', 'failure_notes'])
            writer.writerows([sample['pair_id'], seed] + [''] * 8 for sample in samples for seed in seeds)
    return plan


def execute(plan, output, resume=False):
    output = Path(output).resolve()
    for job in plan['jobs']:
        for sample in plan['samples']:
            if not resume and (output / job['directory'] / sample['pair_id'] / 'run.json').exists():
                raise FileExistsError('Completed runs exist; use a new output directory')
    from material_transfer.records import source_state
    from experiments.summarize_results import inspect_run
    source_hash = source_state()['package_source_sha256']
    for job in plan['jobs']:
        pending = []
        for sample in plan['samples']:
            directory = output / job['directory'] / sample['pair_id']
            if resume and (directory / 'run.json').exists():
                checked, fingerprint = inspect_run(directory, sample, job, plan)
                if checked['status'] != 'complete' or fingerprint[1] != source_hash:
                    raise ValueError(f'Cannot resume incompatible completed run: {directory}')
            else:
                pending.append(sample)
        if not pending:
            print(f"Already complete: seed={job['seed']} method={job['method']}", flush=True)
            continue
        manifest = output / 'prepared' / f"pending_seed_{job['seed']}_{job['method']}.csv"
        fields = ['pair_id', 'target', 'exemplar', 'mask', 'depth', 'material_prompt', 'seed']
        with manifest.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({**{k: sample[k] for k in fields if k != 'seed'}, 'seed': job['seed']} for sample in pending)
        command = [sys.executable, '-m', 'material_transfer', 'batch',
                   '--config', str(output / 'prepared' / f"{job['method']}.json"),
                   '--manifest', str(manifest),
                   '--output', str(output / job['directory'])]
        print(f"Running seed={job['seed']} method={job['method']}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--config', type=Path, help='Shared settings; method is replaced by A/B/C/D')
    parser.add_argument('--execute', action='store_true', help='Run real inference after preparation')
    parser.add_argument('--resume', action='store_true', help='Skip only validated completed runs with the same source fingerprint')
    args = parser.parse_args()
    plan = prepare(args.manifest, args.output, args.seeds,
                   TransferConfig.load(args.config) if args.config else None)
    print(f"Prepared {len(plan['samples'])} pairs x {len(plan['seeds'])} seeds x 4 methods")
    print(f"Plan: {args.output.resolve() / 'experiment_plan.json'}")
    if args.execute:
        execute(plan, args.output, args.resume)
    else:
        print('Preparation only: no models loaded and no generated results.')


if __name__ == '__main__':
    main()
