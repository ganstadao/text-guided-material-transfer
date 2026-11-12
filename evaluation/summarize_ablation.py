"""Summarize complete paired A/B/C/D metrics without claiming significance."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

METRICS = ('bg_mae', 'bg_ssim', 'bg_lpips', 'ref_iou', 'ref_dice', 'clip_sim')
COMPARISONS = (('B', 'A'), ('C', 'A'), ('D', 'B'), ('D', 'C'))


def number(row, field):
    try:
        value = float(row.get(field, 'nan'))
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def summarize(metrics, strata, output):
    with Path(metrics).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    index = {}
    for row in rows:
        key = row['pair_id'], row['group']
        if key in index or row['status'] != 'complete':
            raise ValueError('Metrics contain duplicate or incomplete runs')
        index[key] = row
    pairs = sorted({r['pair_id'] for r in rows})
    if not pairs or any((p, g) not in index for p in pairs for g in 'ABCD'):
        raise ValueError('Every pair needs all four methods')
    labels = {r['pair_id']: r for r in json.loads(Path(strata).read_text(encoding='utf-8'))}
    if not set(pairs) <= labels.keys():
        raise ValueError('Missing sample strata')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(pairs=len(pairs), runs=len(rows),
        unique_targets=len({labels[p]['target_id'] for p in pairs}),
        unique_references=len({labels[p]['reference_id'] for p in pairs}),
        reference_gt_statuses=sorted({r.get('gt_status', '') for r in rows}), groups={}, strata={}, comparisons={})
    def aggregates(selected):
        result = {}
        for metric in METRICS:
            values = [v for r in selected if (v := number(r, metric)) is not None]
            result[metric] = dict(n=len(values), mean=statistics.mean(values) if values else None,
                                 median=statistics.median(values) if values else None)
        return result
    for group in 'ABCD':
        summary['groups'][group] = aggregates([r for r in rows if r['group'] == group])
    for dimension in ('target_context', 'reference_context'):
        summary['strata'][dimension] = {}
        for value in sorted({labels[p][dimension] for p in pairs}):
            summary['strata'][dimension][value] = {g: aggregates([r for r in rows
                if r['group'] == g and labels[r['pair_id']][dimension] == value]) for g in 'ABCD'}
    deltas = []
    for candidate, baseline in COMPARISONS:
        key = f'{candidate}-{baseline}'
        summary['comparisons'][key] = {}
        for metric in METRICS:
            values = []
            for pair in pairs:
                a, b = number(index[pair, candidate], metric), number(index[pair, baseline], metric)
                if a is not None and b is not None:
                    values.append(a - b)
                    deltas.append(dict(pair_id=pair, comparison=key, metric=metric, delta=a - b))
            summary['comparisons'][key][metric] = dict(n=len(values), mean_delta=statistics.mean(values) if values else None)
    (output / 'ablation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output / 'paired_deltas.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['pair_id', 'comparison', 'metric', 'delta'])
        writer.writeheader()
        writer.writerows(deltas)
    def fmt(value):
        return f'{value:.4f}' if value is not None else 'N/A'
    lines = ['# 四组首轮结果', '',
        f"{len(pairs)} 对组合、{len(rows)} 次生成，来自 {summary['unique_targets']} 个目标和 {summary['unique_references']} 个参考。",
        '组合共享目标/参考，不视作独立同分布样本；单 seed 不作显著性或泛化结论。', '',
        '参考 GT 状态：' + ', '.join(summary['reference_gt_statuses']) + '。草稿标注的 IoU/Dice 是暂定值。', '',
        '| 方法 | 背景 MAE ↓ | 背景 SSIM ↑ | 背景 LPIPS ↓ | 参考 IoU ↑ | CLIP 图像相似度 |',
        '|---|---:|---:|---:|---:|---:|']
    for group in 'ABCD':
        stats = summary['groups'][group]
        lines.append('| ' + group + ' | ' + ' | '.join(fmt(stats[k]['mean'])
            for k in ('bg_mae', 'bg_ssim', 'bg_lpips', 'ref_iou', 'clip_sim')) + ' |')
    lines += ['', '## 成对变化', '', '数值定义为候选方法减基线。MAE/LPIPS 为负代表背景误差降低，SSIM 为正代表背景相似度提高。', '',
              '| 对比 | Δ背景 MAE | Δ背景 SSIM | Δ背景 LPIPS | ΔCLIP |', '|---|---:|---:|---:|---:|']
    for key, stats in summary['comparisons'].items():
        lines.append('| ' + key + ' | ' + ' | '.join(fmt(stats[k]['mean_delta'])
            for k in ('bg_mae', 'bg_ssim', 'bg_lpips', 'clip_sim')) + ' |')
    lines += ['', '## 解释范围', '',
        '- 背景指标衡量保留程度，不能替代材质质量；CLIP 相似度会混入形状和物体语义。',
        '- IoU/Dice 只评价参考分割，不等同于最终迁移效果；同一参考重复使用，独立参考只有四张。',
        '- 本轮参考为三个材质特写和一个孤立物体，不把它们声称为多材质杂乱参考组。',
        '- 材质符合度、形状保留和自然程度仍需独立人工盲评；未生成或预填人工分数。',
        '- 目标 mask 与参考区域标注为助手草稿，尚待独立复核。分层数值见 ablation_summary.json。', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--strata', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    summarize(args.metrics, args.strata, args.output)
