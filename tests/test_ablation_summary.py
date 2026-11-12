import csv
import json
import pytest
from evaluation.summarize_ablation import summarize


def test_paired_summary_uses_candidate_minus_same_pair_baseline(tmp_path):
    metrics = tmp_path / 'metrics.csv'
    rows = [dict(pair_id='p01', group=g, status='complete', bg_mae=value,
                 gt_status='assistant_draft') for g, value in zip('ABCD', (0.1, 0.05, 0.2, 0.15))]
    with metrics.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    strata = tmp_path / 'strata.json'
    strata.write_text(json.dumps([dict(pair_id='p01', target_id='t01', reference_id='e01',
        target_context='simple', reference_context='texture_closeup')]))
    result = summarize(metrics, strata, tmp_path / 'report')
    assert result['pairs'] == 1 and result['unique_targets'] == 1
    assert result['comparisons']['B-A']['bg_mae']['mean_delta'] == pytest.approx(-0.05)
    assert result['groups']['A']['ref_iou']['mean'] is None
    with metrics.open('a', newline='') as stream:
        csv.DictWriter(stream, fieldnames=list(rows[0])).writerow(rows[0])
    with pytest.raises(ValueError, match='duplicate'):
        summarize(metrics, strata, tmp_path / 'report')
