# 评测与盲评（成员 2 用）

## 自动指标

先等四组输出生成到 `outputs/A~D/{pair_id}/`，再运行：

```bash
python evaluation/compute_metrics.py \
  --manifest data/manifest.csv --outputs outputs \
  --groups A B C D --gt-masks data/masks_exemplar_gt \
  --out evaluation/results.csv
```

- 背景保留：目标 mask 外区域的 MAE、SSIM（SSIM 越大越好，MAE 越小越好），四组都有
- 参考图定位：B/D 的 `reference_mask` 对人工 GT 的 IoU、Dice，只评价分割，不等于生成质量
- `--lpips` 加背景 LPIPS（需 `pip install lpips`）；`--clip-model openai/clip-vit-base-patch32` 加输出物体 crop 与参考图的 CLIP 相似度（只作参考，CLIP 会混入物体语义，不能单独当材质正确性证据）
- 不报告全图 PSNR/LPIPS：没有材质迁移后的真值图

## 人工盲评

- 对每个 pair 随机打乱 A/B/C/D 顺序后重命名文件（例如 `p001_1.png` ~ `p001_4.png`），另存一份 pair_id 到顺序的映射，评分者看不到组名
- 三个维度各打 1–5 分：材质符合度（换上去的材质对不对）、物体形状保留、自然程度
- 评分写进 `evaluation/ratings_template.csv`（列：rater, pair_id, group, material_match, shape, naturalness），填完把 group 换回真实组名
- 报告里写清评分人数、样本数、评分尺度与分歧
