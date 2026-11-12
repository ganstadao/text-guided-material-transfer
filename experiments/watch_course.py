"""Watch this local batch and finish its authorized evaluation when all runs exist."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]


def progress(output):
    output = Path(output).resolve()
    plan = json.loads((output / 'experiment_plan.json').read_text(encoding='utf-8'))
    groups = {m: 0 for m in 'ABCD'}
    invalid = []
    for job in plan['jobs']:
        for sample in plan['samples']:
            path = output / job['directory'] / sample['pair_id'] / 'run.json'
            if path.exists():
                try:
                    json.loads(path.read_text(encoding='utf-8'))
                    groups[job['method']] += 1
                except (ValueError, OSError):
                    invalid.append(str(path))
    active = False
    for process in psutil.process_iter(['name', 'cmdline']):
        try:
            args = process.info['cmdline'] or []
            if not (process.info['name'] or '').lower().startswith('python'):
                continue
            text = ' '.join(args).replace('\\', '/').lower()
            if output.name.lower() in text and ('run_ablation.py' in text or '-m material_transfer batch' in text):
                active = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return dict(completed=sum(groups.values()), expected=len(plan['jobs']) * len(plan['samples']),
        groups=groups, generator_active=active, invalid_records=invalid,
        updated_utc=datetime.now(timezone.utc).isoformat(), stage='generation')


def watch(data, output, interval=45, finish=False):
    output = Path(output).resolve()
    while True:
        state = progress(output)
        if state['completed'] == state['expected']:
            state['stage'] = 'metrics_and_blind_package' if finish else 'generation_complete'
        elif not state['generator_active']:
            state['stage'] = 'stopped_incomplete'
        (output / 'progress.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
        print(json.dumps(state), flush=True)
        if state['completed'] == state['expected']:
            if finish:
                subprocess.run([sys.executable, str(ROOT / 'experiments/finish_course.py'),
                    '--data', str(Path(data).resolve()), '--output', str(output)], cwd=ROOT, check=True)
                state['stage'] = 'automatic_complete'
                (output / 'progress.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
                print('AUTOMATIC_COURSE_WORK_COMPLETE', flush=True)
            return 0
        if not state['generator_active']:
            return 1
        time.sleep(interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/course_fixed_v1')
    parser.add_argument('--output', default='outputs/course-1024-v1')
    parser.add_argument('--interval', type=int, default=45)
    parser.add_argument('--finish', action='store_true')
    args = parser.parse_args()
    raise SystemExit(watch(args.data, args.output, args.interval, args.finish))
