# 评测与盲评

当前端到端协议见 [COURSE_RUN.md](../experiments/COURSE_RUN.md)。
背景 mask 已统一为 bool，target/mask 会对齐到生成尺寸；缺失或错误结果会显式记录并返回非零状态。

## 当前冻结数据的评分命令

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:TORCH_HOME = "$PWD/models/torch-cache"
$env:HF_HUB_OFFLINE = '1'
.venv/Scripts/python.exe experiments/finish_course.py --data data/course_fixed_v1 --output outputs/course-1024-v1
```

该命令先检查冻结数据 SHA256 与 96 次运行，再计算全部机器指标、成对变化及分层统计，最后生成空白盲评包。
CLIP 权重目录为 models/local/clip-vit-base-patch32，LPIPS 使用 lpips 0.1.4 与本地 AlexNet 权重。

### 指标定义

- bg_mae：将原 target 以 Lanczos 对齐 result 尺寸，mask 以最近邻对齐，仅在 mask 外统计 RGB 绝对误差。
- bg_ssim：Rec709 亮度、sigma=1.5 的高斯窗口；排除窗口触及编辑前景的背景点。
- bg_lpips：spatial AlexNet LPIPS，两图前景同时置为相同中性灰，再仅对背景距离图取均值。不是原先的全图 LPIPS。
- ref_iou/ref_dice：参考 mask 与独立区域标注比较；当前标注为 assistant_draft，分数是暂定值，不是人工复核后的金标准。
- clip_sim：目标编辑区域包围框 crop 与参考图的 CLIP 图像嵌入余弦相似度，仅为代理指标，不能单独证明材质正确。

`metrics.csv` 的 status/gt_status 保留缺失、错误和标注状态；`metrics.summary.json` 记录协议、依赖版本与有效数量。
`--allow-missing` 用于显式生成部分结果表；默认有缺失运行即返回非零。
`--require-reviewed-gt` 会拒绝没有 reviewed 标记的 GT，当前草稿应先独立复核再使用此选项。

## 匿名评分包

`outputs/course-1024-v1/blind_review/rater_package/` 是可交给评分者的目录，含匿名结果、目标、参考、mask、评分说明与空白 ratings.csv。
`blind_review/private/mapping.csv` 只由实验负责人保存，不向评分者提供。
评分表没有预填或自动生成的人类分数。

收到真实评分后运行：

```shell
.venv/Scripts/python.exe evaluation/summarize_ratings.py --ratings rater1.csv rater2.csv --mapping outputs/course-1024-v1/blind_review/private/mapping.csv --out outputs/course-1024-v1/human_ratings.json
```

空白评分不生成均值；重复条目、未知 item_id、非 1–5 整数会报错。

## 原始接口与建议

### 自动指标

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

### 人工盲评

- 对每个 pair 随机打乱 A/B/C/D 顺序后重命名文件（例如 `p001_1.png` ~ `p001_4.png`），另存一份 pair_id 到顺序的映射，评分者看不到组名
- 三个维度各打 1–5 分：材质符合度（换上去的材质对不对）、物体形状保留、自然程度
- 评分写进 `evaluation/ratings_template.csv`（列：rater, pair_id, group, material_match, shape, naturalness），填完把 group 换回真实组名
- 报告里写清评分人数、样本数、评分尺度与分歧
