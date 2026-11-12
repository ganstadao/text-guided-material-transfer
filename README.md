# Text-guided Material Transfer

以 ZeST 为基础的材质迁移研究代码。输入为**目标图、材质参考图、材质描述**；不是纯文本生成。
目前整理了统一 API、四组开关、批量清单与结果记录。**已在 NVIDIA A40 上完成四组真模型端到端验证**，并验证自动 mask／深度路径。当前是可运行的研究基础设施，不声称与官方 ZeST 数值等价，也不预设改动有效。见 [阶段总结](docs/PROJECT_STATUS.md)。

## 安装

使用 Python 3.10–3.12。以下命令在仓库根目录运行；可使用自己的 conda 或 virtualenv 环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,ui,auto-mask]'
export HF_HOME="$PWD/models/hf-cache"
export U2NET_HOME="$PWD/models/rembg"
python -m material_transfer doctor
python -m pytest -q
```

`doctor` 显示当前 Python、PyTorch 和 CUDA 状态；没有可见 GPU 时返回状态码 1。完整 SDXL 推理需要支持 CUDA 的 NVIDIA GPU；CPU 可运行数据检查与轻量测试，或使用 `device=cpu`、`dtype=float32` 配置进行较慢的推理。

`pyproject.toml` 定义核心依赖和可选的 `ui`、`auto-mask`、`dev` 依赖。`requirements-lock.txt` 记录已验证的 Linux x86_64 / Python 3.10 / CUDA 12.4 环境；其他平台优先按 `pyproject.toml` 安装与硬件兼容的环境。

## 模型权重

根据 [ZeST 官方 README](https://github.com/ttchengab/zest_code#download-models) 的 IP-Adapter、SDXL、depth ControlNet、DPT 路线整理。当前使用 Diffusers 原生 IP-Adapter loader，**无需 clone `ip_adapter/` 或 DPT 源码**。

| 用途 | 默认模型 / 文件 |
|---|---|
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` |
| Depth ControlNet | `diffusers/controlnet-depth-sdxl-1.0` |
| CLIP ViT-H 图像编码器 | `h94/IP-Adapter/models/image_encoder/` |
| 基础 IP-Adapter SDXL ViT-H | `h94/IP-Adapter/sdxl_models/ip-adapter_sdxl_vit-h.safetensors` |
| 参考图区域定位（B/D） | `CIDAS/clipseg-rd64-refined` |
| 自动深度（可选） | `Intel/dpt-hybrid-midas` |

建议权重集中存放，避免复制多个模型仓库：

```bash
export HF_HOME=/path/to/shared/model-cache
python scripts/download_models.py --config configs/D.json --with-depth
```

此命令会下载大模型，先检查该目录空间。已有权重时无需重下：复制配置 JSON 并将 `base_model`、`controlnet` 改为本地模型目录，将 `adapter_repo` 改为包含 `models/image_encoder/` 和 `sdxl_models/` 的本地 IP-Adapter 根目录。本地模型没有 fp16 变体时加 `"model_variant": null`。旧版只有 `.bin` 时设置 `adapter_weight` 为 `ip-adapter_sdxl_vit-h.bin`。离线可加 `"local_files_only": true`。本地目录必须是完整模型目录，不能只有单个权重文件。

优先使用提前固定的目标 mask 和深度图。自动深度使用 Transformers DPT hybrid，预处理可能与原 DPT 脚本不同。严格复现实验请按官方 README 预计算深度，并让四组共享同一份文件。支持官方 uint16 depth PNG，也支持 uint8 PNG。

缺省目标 mask 时使用可选 rembg（`python -m pip install -e '.[auto-mask]'`）；mask 从 RGBA **alpha 通道**提取。正式实验推荐上传人工确认的 mask，白色表示待编辑目标，黑色表示保留背景。

## 四组运行

| 配置 | 参考图 CLIPSeg | 自定义特征融合 |
|---|---:|---:|
| `configs/A.json` | 关 | 关 |
| `configs/B.json` | 开 | 关 |
| `configs/C.json` | 关 | 开 |
| `configs/D.json` | 开 | 开 |

```bash
python -m material_transfer run \
  --config configs/A.json \
  --target data/target.png --exemplar data/material.png \
  --mask data/target_mask.png --depth data/depth.png \
  --material-prompt 'woven fabric' --output outputs/A/p001
```

对 B/C/D 使用相同输入，分别替换配置和输出路径。`material-prompt` 仅供参考图分割／融合使用，SDXL 正负提示词从配置读取并在四组固定。目标 mask 必须与目标图尺寸相同；模型输入默认统一缩放为 1024×1024，非正方形输入会拉伸。正式评测应提前固定预处理规则。

B/D 将参考 mask 外区域置白；空分割会明确报错，避免悄悄退回基线。C/D 复用一个 SDXL transformer block 的预训练权重，作用于 CLIP patch tokens，再以材质文本注意力 mask 加权 CLS 表示。该分支是**未训练对齐的探索方法**，不是已经验证的材质特征提取器。实现修复了多头 reshape、短语子词选择、重复加载 UNet 和向扩散管线传入未知参数的问题。注意力图显示原始权重，不按每图 min-max 拉伸。

A 为 ZeST-style 基线：保留前景去色、深度 ControlNet、IP-Adapter 条件，但更换了 adapter loader 并修正 mask 提取，因此未验证前不得称为官方逐像素复现。见 [实现差异](docs/implementation.md)。

## 批量清单与结果接口

CSV 路径相对于 **CSV 所在目录**，列如下，`seed` 可选。`mask`、`depth` 列必须存在但可为空，空值触发自动处理。

```csv
pair_id,target,exemplar,mask,depth,material_prompt,seed
p001,targets/apple.png,exemplars/cloth.png,masks/apple.png,depths/apple.png,woven fabric,42
```

```bash
python run_batch.py --config configs/D.json --manifest data/manifest.csv --output outputs/D --validate-only
python run_batch.py --config configs/D.json --manifest data/manifest.csv --output outputs/D
```

`--validate-only` 检查清单、图片、mask 尺寸与描述，不下载权重。批量只加载一次模型，同一 seed 在四组保持一致；已完成的目录会拒绝覆盖。协作者可共同维护清单、配置与结果协议，不需要修改生成逻辑。

每个输出目录包含 `result.png`、`target.png`、`exemplar.png`、`target_mask.png`、`depth.png`、`init.png`、`reference.png` 和 `run.json`；B/D 增加 `reference_mask.png`，C/D 增加 `attention_mask.png`。`run.json` 包含配置、实际组件开关、输入路径、随机种子、依赖版本、Git commit/dirty 状态、包源码 SHA256、预处理／生成时间及 CUDA 峰值显存。计时排除主干加载，但包含按需加载的预处理模型，正式性能实验应先 warm up 并独立报告加载成本。

后续 evaluation 从 `run.json` 的 `artifacts` 读取，不依赖 UI。当前尚未实现指标计算和正式评测数据集。

Python 调用：

```python
from material_transfer import TransferConfig
from material_transfer.pipeline import MaterialTransfer
from material_transfer.images import load_rgb, load_mask, depth_image

cfg = TransferConfig.load('configs/D.json')
engine = MaterialTransfer(cfg)
target = load_rgb('data/target.png')
artifacts, metadata = engine.run(
    target, load_rgb('data/material.png'), 'woven fabric',
    load_mask('data/target_mask.png', target.size),
    depth_image('data/depth.png', target.size),
)
artifacts['result'].save('result.png')
```

同一 engine 串行使用，不要跨方法修改已加载 engine；改变方法／模型路径应新建 engine。

## 交互界面

```bash
python demo_gradio.py --config configs/D.json --port 7860
```

UI 使用相同 API，单请求串行执行并保存结果。启动后在浏览器打开 `http://127.0.0.1:7860`。默认只绑定本机回环地址；远程使用时可通过 SSH 或编辑器转发相应端口。

## 复核示例

下载 ZeST 官方示例并准备近似目标 mask：

```bash
python scripts/prepare_smoke.py
for method in A B C D; do
  python run_batch.py --config "configs/${method}.json" \
    --manifest data/smoke/manifest.csv --output "outputs/smoke/${method}"
done
python scripts/make_smoke_grid.py outputs/smoke
```

对比图保存到 `outputs/smoke/comparison.png`。示例 mask 只用于功能检查，不是评测真值；模型、示例数据和生成结果不随源码分发。已有完整结果的目录会拒绝覆盖，重跑时请使用新的输出目录。

## 文件结构

```text
material_transfer/     配置、图像处理、分割／深度、特征融合、生成、CLI、结果协议
configs/               四组配置（其他参数见 TransferConfig）
tests/                 轻量数据流与张量／文件协议验证，不下载权重
scripts/               模型下载工具
experiments/           四组消融编排、结果汇总与历史独立探针（见该目录 README）
run_batch.py           批量 CLI 兼容入口
demo_gradio.py         统一 API 的可选 UI
```

旧的 `CustomIPAdapterXL` API 已由统一引擎替代；原代码保存在 `experiments/legacy/`，也可通过 Git 查看。上游 MIT 版权声明保留在 `LICENSE`。上游工作见 [ZeST](https://github.com/ttchengab/zest_code)、[IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)；使用基础方法时请引用 [ZeST 论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00144.pdf)。
