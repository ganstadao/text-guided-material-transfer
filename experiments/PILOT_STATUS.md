# 本地首轮实验记录

日期：2026-11-11。

## 已完成

- 拉取当前仓库到本地工作区。
- 使用原有 `scripts/prepare_smoke.py` 下载 ZeST 官方南瓜目标图、釉面杯参考图、深度图，生成近似目标 mask。
- 为这 1 对功能样例准备 seed=42 的 A/B/C/D 四组计划，并完成四组真实生成。
- 下载完整 SDXL、Depth ControlNet、IP-Adapter、CLIP 图像编码器和 CLIPSeg 权重，逐个校验官方 SHA256。
- 新增实验编排、完整性检查、CSV 汇总、对比拼图、人工观察表和六对样本清单模板。
- 加入 CPU 卸载与 VAE 分块配置；融合分支使用实际执行设备，分割和自动深度可在 CPU 运行。
- 全部 22 项测试通过，`pip check` 和 `git diff --check` 通过。

## 当前生成状态

真实结果位于 `outputs/local-6gb-smoke/`：complete=4、pending=0、invalid=0。
四组 target/exemplar/target_mask/depth/init 的文件哈希和包代码指纹一致，seed 均为 42。
四张输出互不相同，均为 512×512；B/D 有 reference_mask，C/D 有 attention_mask。
对比图：`outputs/local-6gb-smoke/comparisons/pumpkin_glaze_seed_42.png`。

配置：512×512、steps=10、strength=0.9999，实际执行 9 个去噪时间步，
sequential CPU offload、VAE tiling。VAE 驻留 GPU，支持 SDXL 编解码时的精度切换。

| 方法 | 预处理秒 | 生成秒 | PyTorch 峰值分配显存 GiB |
|---|---:|---:|---:|
| A | 0.03 | 66.29 | 1.337 |
| B | 3.71 | 68.50 | 1.337 |
| C | 4.21 | 52.38 | 1.402 |
| D | 8.36 | 50.75 | 1.402 |

计时排除主干加载。单次计时受缓存、CPU 卸载和 Windows 内存调度影响，不能据此判断 C/D 更快。
显存数字为 PyTorch 的 max_memory_allocated，不是整机显存总占用。
旧 `outputs/pilot-smoke/` 仍是最初准备阶段的 pending 预览；以 local-6gb-smoke 为本轮结果。

## 本机环境

- Python 3.10.9。
- NVIDIA GeForce RTX 3060 Laptop GPU，显存 6GB；CUDA 可见。
- 独立 `.venv`：torch 2.5.1+cu124、torchvision 0.20.1+cu124、transformers 4.46.3、diffusers 0.32.2、accelerate 1.2.1。
- 完整依赖记录：`outputs/local-6gb-smoke/environment.txt`。
- 全局 Python 环境保持原有版本；安装临时文件、缓存、权重和输出均放在 F 盘项目目录。
- 本机约 16GB 主内存，通过 CPU 卸载完成了当前样例。

首次默认 CPU 线程设置加载时遇到 torch_cpu.dll / 0xc0000005 原生崩溃。
CUDA 分配和卷积检查通过；设置 OMP_NUM_THREADS=1、MKL_NUM_THREADS=1 后四组均成功。
这是本机实测的规避方式，尚未进一步证明底层崩溃根因。
本地启动脚本 `experiments/run_local.ps1` 已包含线程设置和离线加载环境。

## 初步观察

四组均呈现青绿色釉面外观，帘幕、桌面与南瓜柄大体保留。
A/C 的沟槽较明显，局部有棕色斑点；B/D 表面更平滑，B 的沟槽与局部形状变化较明显。
这些只是单例的目视现象，不代表 B/D 更优，也不能证明融合模块有效。
参考图分割和注意力图应结合输出一起检查。

## 剩余工作

1. 1024×1024、30 步的本机四组验证已完成，见 [DEFAULT_1024_STATUS.md](DEFAULT_1024_STATUS.md)。新增显式 VAE 512 分块和 UNet 无损权重分片后成功。
2. 如需开展正式消融，再扩展样本与多个 seed，固定人工确认的 mask/depth 和评价协议。
3. 当前官方示例及近似 mask 仅用于功能检查，不能用这一对样本得出模块有效性结论。

执行命令、实验问题与分工见 [README.md](README.md)。
模型、下载数据、计划和输出按仓库现有规则保留在被 Git 忽略的目录中；代码尚未提交或推送。
