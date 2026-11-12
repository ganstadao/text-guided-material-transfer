# 24 对课程数据的本地实验协议

本轮使用 jerryluo-data 提交的 6 张目标图和 4 张参考图，共 24 个组合。
原始 data/manifest.csv 保留成员的 42–65 种子；本轮冻结版统一 seed=42，明确覆盖原始种子。
四组共 96 次生成，采用 1024×1024、30 步（strength=0.9999 时实际 29 个时间步）。

## 数据与标注

- 固定数据：data/course_fixed_v1/manifest.csv；6 个目标各复用一份 mask、一份预计算 DPT 深度。
- 标注说明：experiments/annotations/course_v1.json；mask/参考区域均为助手草稿，已经目视检查，但尚未独立人工复核。
- t01/t02/t03：整椅、整沙发、整桌；t04：未被衣物遮挡的可见沙发皮面；t05：整张扶手椅；t06：无遮挡木质桌面与抽屉面板。
- 用户已确认 t06 的目标是桌面；这不等同于对每个 mask 像素或参考 GT 的独立复核。
- 参考 e01/e02/e03 为纹理特写，e04 为孤立石块；本轮不称它们为杂乱、多材质参考组。
- 参考 GT 独立于 CLIPSeg：纹理图整幅标记，石块用单独的轮廓/GrabCut 草稿；每个 pair 复制对应参考标注。
- 冻结信息与文件 SHA256：data/course_fixed_v1/frozen_inputs.json。已冻结目录禁止原地重新生成。

预览：data/course_fixed_v1/previews/target_masks.png、reference_masks.png。
若要修改像素标注，应编辑标注 JSON 并生成 course_fixed_v2，不能在正在运行的 v1 上修改输入。
原始图像的具体来源仍以成员提交记录为准，报告使用前应由数据提供者补充来源细节。

## 重建固定输入

需要完整 Intel/dpt-hybrid-midas 模型目录 models/local/dpt 和 OpenCV prep 依赖。
首次可先用 --masks-only 查看预览，再准备深度和冻结清单：

```shell
.venv/Scripts/python.exe scripts/prepare_course_data.py --output data/course_fixed_v2 --masks-only
.venv/Scripts/python.exe scripts/prepare_course_data.py --output data/course_fixed_v2
```

## 运行与断点续跑

```powershell
./experiments/run_local.ps1 -Manifest data/course_fixed_v1/manifest.csv -Config experiments/configs/local_6gb_1024.json -Output outputs/course-1024-v1 -Resume
```

沿用已经验证的逐层 CPU 卸载、VAE 512 分块、UNet 10 份无损分片、CPU 单线程。
resume 只跳过图片完整、配置/输入与计划相符、且包代码指纹与当前一致的结果。
剩余样本生成临时清单后交给核心批量入口，因此不会覆盖完整结果。
改动生成包源码后请使用新输出目录；不要混合不同实现的结果。

## 评分与交付

全部生成后执行：

```shell
.venv/Scripts/python.exe experiments/finish_course.py --data data/course_fixed_v1 --output outputs/course-1024-v1
```

- summary.csv/json：96 次运行完整性与耗时/显存。
- comparisons/：24 张目标/参考/A/B/C/D 对比图。
- metrics.csv 与 metrics.summary.json：机器指标、协议、标注状态、依赖。
- evaluation/RESULTS.md：分组均值、B−A/C−A/D−B/D−C 的成对变化。
- evaluation/ablation_summary.json：按目标背景与参考类型的分层汇总。
- evaluation/paired_deltas.csv：逐 pair 的变化。
- blind_review/rater_package：匿名结果和空白评分表；private/mapping.csv 单独保管。
- completion.json：自动环节完成情况，以及人工评分与 GT 复核仍待完成的状态。

背景分数不衡量材质正确性；CLIP 图像相似度包含语义因素。
24 个组合共享 6 个目标和 4 个参考，不能当成 24 个独立来源样本；当前只有一个 seed，不作显著性结论。
真实盲评分数只能由评分者填写，不由程序补齐。草稿 GT 的分割分数暂不作为正式金标准结果。
