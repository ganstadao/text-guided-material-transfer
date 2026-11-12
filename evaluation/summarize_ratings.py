"""Join real human ratings to the private mapping and summarize valid scores."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def summarize(ratings, mapping, output):
    with Path(mapping).open(encoding='utf-8-sig', newline='') as stream:
        lookup = {r['item_id']: r for r in csv.DictReader(stream)}
    valid, pending, seen = [], 0, set()
    for path in ratings:
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                axes = ('material_match', 'shape', 'naturalness')
                if not all(row.get(k, '').strip() for k in ('rater',) + axes):
                    pending += 1
                    continue
                key = row['rater'], row['item_id']
                if key in seen or row['item_id'] not in lookup:
                    raise ValueError(f'Duplicate or unknown rating: {key}')
                seen.add(key)
                source = lookup[row['item_id']]
                if row['pair_id'] != source['pair_id']:
                    raise ValueError('Rating pair does not match mapping')
                scores = {k: int(row[k]) for k in axes}
                if any(v < 1 or v > 5 for v in scores.values()):
                    raise ValueError('Ratings must be integers in 1..5')
                valid.append(dict(rater=row['rater'], pair_id=row['pair_id'], group=source['group'], **scores))
    result = dict(completed=len(valid), incomplete=pending, raters=sorted({r['rater'] for r in valid}), groups={})
    provenance = []
    for rating_path in ratings:
        path = Path(rating_path).with_suffix('.provenance.json')
        if path.exists():
            provenance.append(json.loads(path.read_text(encoding='utf-8')))
    result['rating_provenance'] = provenance
    result['independent_blind_review_complete'] = bool(valid) and len(provenance) == len(ratings) and all(
        p.get('independent_blind_review_complete') is True for p in provenance)
    for group in 'ABCD':
        rows = [r for r in valid if r['group'] == group]
        result['groups'][group] = {k: dict(n=len(rows), mean=statistics.mean(r[k] for r in rows) if rows else None)
            for k in ('material_match', 'shape', 'naturalness')}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Completed ratings: {len(valid)}; incomplete rows: {pending}")
    return 0 if valid else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ratings', nargs='+', required=True, type=Path)
    parser.add_argument('--mapping', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(summarize(args.ratings, args.mapping, args.out))
