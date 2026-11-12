# 消融实验工作区

24 对课程数据的当前完整运行、评分及盲评协议见 [COURSE_RUN.md](COURSE_RUN.md)。

本目录新增正式实验编排工具，生成逻辑统一调用 `material_transfer`。
原有 `CLIP.py`、`CLIPSeg.py`、`CustomIPAdapterXL_test.py` 和 `legacy/` 是历史探针，
部分依赖旧 API、示例路径；不要把它们当作当前消融入口。

## 第一轮要回答的问题

| 方法 | 参考图分割 | 特征融合 | 主要对比 |
|---|---|---|---|
| A | 关 | 关 | ZeST-style 基线 |
| B | 开 | 关 | B−A：分割的贡献 |
| C | 关 | 开 | C−A：融合的贡献 |
| D | 开 | 开 | D−C：有融合时分割的贡献；D−B：有分割时融合的贡献 |

四组固定目标图、参考图、材质描述、目标 mask、深度、seed 和其余配置。
`material_prompt` 只用于参考图分割/融合；SDXL 正负提示词使用共同配置。
第一轮不调整阈值、IP scale 或 ControlNet scale；先找出组件在哪些样本有效或失败。

## 先跑一个功能示例

所有命令在仓库根目录运行，适用于 PowerShell 与 bash。

```shell
python scripts/prepare_smoke.py
python experiments/run_ablation.py --manifest data/smoke/manifest.csv --output outputs/pilot-smoke --seeds 42
python experiments/summarize_results.py outputs/pilot-smoke
```

默认仅检查输入、生成计划和观察表，不加载模型。没有生成结果时汇总显示 pending；
对比图显示输入与明确标记的空位。
官方示例只有南瓜/釉面杯这一对，目标 mask 是近似多边形，只用于流程检查。

在按根 README 配置好依赖及权重、有足够显存的机器上执行真实推理：

```shell
python experiments/run_ablation.py --manifest data/smoke/manifest.csv --output outputs/pilot-smoke --seeds 42 --execute
python experiments/summarize_results.py outputs/pilot-smoke --require-complete
```

`--execute` 启动四个独立批量进程，每组进程内部复用模型，组间释放模型。
所有组开始前检查已完成目录；不覆盖已有结果，不自动跳过已完成组。中途失败后保留结果，
可先汇总；重跑完整实验请换输出目录。`--config path.json` 可指定共同模型路径及参数，
脚本只替换 method；命令行 seeds 覆盖 CSV 中的 seed，以保证各组配对。

## 再准备 6 对真实样本

### 本机低显存功能验证

`configs/local_6gb.json` 使用 512×512、10 步、逐层 CPU 卸载和 VAE 分块。
CLIPSeg/DPT 预处理在 CPU 执行，融合分支使用实际执行设备。
VAE 保留在 GPU，以支持 SDXL 编解码期间的精度切换，其余大模型逐层卸载。
该配置与默认 1024×1024、30 步不同，结果用于本机功能验证；四组共用相同设置。
卸载会增加主内存需求及 CPU/GPU 传输成本，6GB 显存不能单独保证成功。

模型完整目录分别放在仓库根目录下的 `models/local/sdxl`、`controlnet`、`ip-adapter`、`clipseg`。
IP-Adapter 目录保留 `models/image_encoder/` 和 `sdxl_models/` 子目录。
本机独立环境是 `.venv`，Windows 使用如下命令（从仓库根目录运行）：

```powershell
./experiments/run_local.ps1 -Output outputs/local-6gb-rerun
```

启动脚本设置 F 盘项目内缓存、离线加载和 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。
本机首次使用默认线程数加载时曾出现 `torch_cpu.dll` 原生崩溃；限制线程数后成功进入去噪。
也可在 PowerShell 设置这两个线程环境变量后，直接执行 `run_ablation.py --execute`。
添加 `-PrepareOnly` 可只准备计划。已完成的目录拒绝覆盖，重跑选择新的输出路径。

1024×1024、30 步的本地配置为 `configs/local_6gb_1024.json`。
它额外指定 `vae_tile_size=512`：SDXL 的默认 VAE 分块阈值为 1024，
仅启用 tiling 不会拆分恰好 1024 的图片；本机曾在这一情况下解码显存不足。
sample/latent 两个空间的分块尺寸同步设置为 512/64，四组共用这一设置：

```powershell
./experiments/run_local.ps1 -Config experiments/configs/local_6gb_1024.json -Output outputs/local-1024-30-rerun
```

保留 strength=0.9999 时，30 步配置实际执行 29 个去噪时间步。
VAE 分块涉及重叠区融合，可能影响局部像素；正式比较应保持所有方法采用同一分块设置。

本机还遇到过大 UNet Safetensors 文件加载时的 Windows 原生崩溃，诊断栈位于
`safetensors.load_file → torch.storage.__getitem__`。本地权重已无损拆成 10 份，
Diffusers 根据 index 自动加载分片，原始文件保留。
首次需要拆分其他同结构本地权重时可运行：

```shell
.venv/Scripts/python.exe scripts/shard_local_unet.py models/local/sdxl/unet
```

已有 index 时无需再次运行。工具逐片核对原始/输出 payload 哈希，保存原始文件 SHA256 和分片记录
至 `models/local/sdxl/unet/shard_provenance.fp16.json`，不修改张量数值。

### 六对样本试验

使用 `manifests/pilot_template.csv` 作为填写模板。路径是待替换占位符，不能直接运行。
建议 3 对背景干净的参考图、3 对背景复杂的参考图，覆盖不同材质。
不要把重复同一目标或参考的组合解释成独立、多样的 6 个样本。

1. 每对准备目标图、参考图、人工确认的目标 mask、固定深度图。
2. 填写材质描述（英文短语）和唯一 pair_id；CSV 相对路径以 CSV 所在目录为基准。
3. 深度可提前用现有 DPT 路径计算，但四组必须读取同一份文件。
4. 把清单保存到 `data/pilot/manifest.csv`，先用 seed 42 运行。
5. 流程确认后用 42、123、2026 三个 seed 重跑到新目录，共 72 次生成。

```shell
python experiments/run_ablation.py --manifest data/pilot/manifest.csv --output outputs/pilot-six --seeds 42 123 2026
python experiments/run_ablation.py --manifest data/pilot/manifest.csv --output outputs/pilot-six --seeds 42 123 2026 --execute
python experiments/summarize_results.py outputs/pilot-six --require-complete
```

## 交付物与如何观察

- `experiment_plan.json`：共同配置、各次任务、输入文件 SHA256。
- `prepared/`：四组配置与每个 seed 的绝对路径清单。
- `seed_N/A…D/pair_id/`：核心引擎保存的真实输出和 run.json。
- `summary.csv/json`：complete/pending/invalid、耗时和显存；不表示图像质量评分。
- `comparisons/`：每对每个 seed 的目标、参考、A/B/C/D 拼图，保留宽高比。
- `observations.csv`：人工填写的对比观察表；重复准备和汇总不会覆盖人工记录。

观察时同时查看 reference_mask、attention_mask 和最终结果。记录材质颜色、纹理、反光
是否接近参考，背景是否变化，目标轮廓是否破坏，分割是否漏选或误选。
每个对比写“更好/相近/更差 + 具体现象”，保留失败案例，先不做没有定义的总分。
注意力热图不能单独证明定位正确；背景像素误差也不能代表材质迁移质量。

汇总检查配置、输入、实际组件开关、必需图片，以及各方法共享中间图和包源码指纹的一致性。
原始输入应保留到汇总完成；移动到服务器后应重新准备计划，避免绝对路径失效。
准备后不要修改输入或生成配置；修改实验设置使用新的输出目录。

## 分工建议

| 工作 | 交付 |
|---|---|
| 数据成员 | 固定清单、确认过的 mask/depth、参考背景类型 |
| 分割成员 | B−A / D−C 的 reference_mask 与成功失败案例 |
| 融合成员 | C−A / D−B 的输出与描述敏感性观察 |
| 实验负责人 | 统一计划、汇总表、拼图与观察结论 |

第一轮用于发现问题；正式质量结论需要更完整的数据、预先定义的评价协议及重复实验。
