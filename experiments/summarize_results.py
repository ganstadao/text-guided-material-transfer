"""Summarize planned runs; missing outputs are explicitly marked pending."""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

try:
    from .run_ablation import file_hash
except ImportError:
    from run_ablation import file_hash


def inspect_run(directory, sample, job, plan):
    record_path = directory / 'run.json'
    if not record_path.exists():
        return {'status': 'pending', 'detail': 'run.json missing'}, None
    try:
        record = json.loads(record_path.read_text(encoding='utf-8'))
        expected = {**plan['configs'][job['method']], 'seed': job['seed']}
        if record['config'] != expected:
            raise ValueError('Configuration differs from experiment plan')
        for key in ('pair_id', 'target', 'exemplar', 'mask', 'depth', 'material_prompt'):
            if record['inputs'][key] != sample[key]:
                raise ValueError(f'Input differs from plan: {key}')
        for key, digest in sample['sha256'].items():
            if file_hash(sample[key]) != digest:
                raise ValueError(f'Original input changed after preparation: {key}')
        artifacts = record['artifacts']
        required = ['result', 'target', 'exemplar', 'target_mask', 'depth', 'init', 'reference']
        if job['method'] in 'BD':
            required.append('reference_mask')
        if job['method'] in 'CD':
            required.append('attention_mask')
        shared = {}
        for key in required:
            path = (directory / artifacts[key]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError('Artifact path escapes run directory')
            with Image.open(path) as image:
                image.verify()
            if key in ('target', 'exemplar', 'target_mask', 'depth', 'init'):
                shared[key] = file_hash(path)
        flags = record['timings']
        if flags['segmentation_applied'] != (job['method'] in 'BD') or flags['attention_applied'] != (job['method'] in 'CD'):
            raise ValueError('Actual component flags differ from method')
        source = record['package_source_sha256']
        if not source:
            raise ValueError('Missing package source fingerprint')
        return dict(status='complete', detail='', generation_seconds=flags.get('generation_seconds'),
                    peak_cuda_bytes=flags.get('peak_cuda_bytes'),
                    result=str(directory / artifacts['result'])), (shared, source)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return {'status': 'invalid', 'detail': str(exc)}, None


def summarize(output):
    output = Path(output).resolve()
    plan = json.loads((output / 'experiment_plan.json').read_text(encoding='utf-8'))
    rows, comparisons = [], output / 'comparisons'
    comparisons.mkdir(exist_ok=True)
    for seed in plan['seeds']:
        for sample in plan['samples']:
            group, fingerprints = [], []
            for method in 'ABCD':
                job = next(j for j in plan['jobs'] if j['seed'] == seed and j['method'] == method)
                values, fingerprint = inspect_run(output / job['directory'] / sample['pair_id'], sample, job, plan)
                row = dict(pair_id=sample['pair_id'], seed=seed, method=method, **values)
                group.append(row)
                if fingerprint is not None:
                    fingerprints.append(fingerprint)
            if any(f != fingerprints[0] for f in fingerprints[1:]):
                for row in group:
                    if row['status'] == 'complete':
                        row.update(status='invalid', detail='Shared artifacts or package source differ across methods')
            rows.extend(group)
            canvas = Image.new('RGB', (6 * 256, 290), 'white')
            draw = ImageDraw.Draw(canvas)
            panels = [('Target', sample['target']), ('Reference', sample['exemplar'])]
            panels += [(r['method'] + ': ' + r['status'], r.get('result') if r['status'] == 'complete' else None) for r in group]
            for i, (label, path) in enumerate(panels):
                x = i * 256
                draw.text((x + 8, 8), label, fill='black')
                if path and Path(path).is_file():
                    with Image.open(path) as image:
                        tile = ImageOps.contain(image.convert('RGB'), (256, 256))
                    canvas.paste(tile, (x + (256 - tile.width) // 2, 30 + (256 - tile.height) // 2))
                else:
                    draw.rectangle((x + 8, 38, x + 248, 278), outline='gray')
                    draw.text((x + 24, 145), 'No validated result', fill='gray')
            canvas.save(comparisons / f"{sample['pair_id']}_seed_{seed}.png")
    fields = ['pair_id', 'seed', 'method', 'status', 'detail', 'generation_seconds', 'peak_cuda_bytes', 'result']
    with (output / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = {status: sum(r['status'] == status for r in rows) for status in ('complete', 'pending', 'invalid')}
    (output / 'summary.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(counts)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    counts = summarize(args.output)
    return int(bool(counts['invalid'] or (args.require_complete and counts['pending'])))


if __name__ == '__main__':
    raise SystemExit(main())
