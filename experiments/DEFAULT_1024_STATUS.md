# 本地 1024×1024 / 30 步验收

记录日期：2026-11-12。状态：A/B/C/D 四组真实生成完成，汇总为 complete=4、pending=0、invalid=0。

## 输入与配置

- ZeST 官方南瓜目标图、釉面杯参考图、固定近似目标 mask 和固定深度图；材质描述 glazed ceramic。
- seed=42，width=height=1024，steps=30，strength=0.9999，实际执行 29 个去噪时间步。
- NVIDIA RTX 3060 Laptop GPU / 6GB，约 16GB 主内存，Windows，独立 `.venv`。
- torch 2.5.1+cu124、diffusers 0.32.2、transformers 4.46.3、accelerate 1.2.1。
- sequential CPU offload，VAE 驻留 GPU，vae_tiling=true，vae_tile_size=512；对应 latent tile=64。
- OMP_NUM_THREADS=1、MKL_NUM_THREADS=1，所有模型从 F 盘离线加载。

## 结果

| 方法 | 预处理秒 | 生成秒 | PyTorch 峰值分配显存 GiB |
|---|---:|---:|---:|
| A | 0.02 | 138.29 | 1.43 |
| B | 3.56 | 148.16 | 1.43 |
| C | 4.19 | 133.30 | 1.50 |
| D | 7.70 | 129.45 | 1.50 |

计时排除主干加载；单次时间受缓存、卸载与系统调度影响，不作方法性能排名。
显存为 torch.cuda.max_memory_allocated，不含整机全部显存使用。

输出目录：`outputs/local-1024-30-tiled512/`。
各组原图位于 `seed_42/{A,B,C,D}/pumpkin_glaze/result.png`，均为真实 1024×1024 输出。
对比图位于 `comparisons/pumpkin_glaze_seed_42.png`，汇总为 `summary.csv` 与 `summary.json`。
配置、输入哈希、实际组件开关和依赖见每次运行的 run.json；完整环境见 environment.txt。

一致性检查：四组共享 target/exemplar/target_mask/depth/init 文件哈希、seed 和包代码指纹。
B/D 的 reference_mask、C/D 的 attention_mask 均存在。已目视检查对比图。
全部 27 项回归测试通过，其中包括真实小型 VAE 分块尺寸测试与无损权重分片测试。

## 本机遇到的问题与处理

1. 初次 1024 去噪完成，但 VAE 解码显存不足。仅 enable_vae_tiling 不能保证实际分块：
   此 VAE 默认阈值为 1024，输入恰为 1024 时不会分块。新增 vae_tile_size，明确同步设置 sample/latent 的 512/64 阈值后成功。
2. 单线程设置下仍出现过 Safetensors 加载的原生访问异常，栈位于 torch.storage.__getitem__。
   使用 `scripts/shard_local_unet.py` 将 UNet 无损拆为 10 份约 512MiB 的分片。
   原始权重保留，1680 个张量的原始 payload 全部逐片校验，原文件 SHA256 与官方记录一致。
   provenance 位于 `models/local/sdxl/unet/shard_provenance.fp16.json`。分片后四组均成功加载。

原始 UNet SHA256：`83e012a805b84c7ca28e5646747c90a243c65c8ba4f070e2d7ddc9d74661e139`。
Windows 大文件 mmap 路径有[相似的访问异常报告](https://github.com/Comfy-Org/ComfyUI/issues/15424)；
本轮依据本机栈和分片后的实际运行记录处理，未进一步证明操作系统层面的根因。

首次无显式分块的失败计划保留在 `outputs/local-1024-30-smoke/`，该目录不含完成结果。

## 复跑

从仓库根目录运行，并使用新的输出目录：

```powershell
./experiments/run_local.ps1 -Config experiments/configs/local_6gb_1024.json -Output outputs/local-1024-30-rerun
```

VAE 分块会在重叠区域融合，可能改变部分像素；正式四组比较应统一此设置。
目前是单样本、单 seed 的运行验收，不据此判断分割或融合组件有效性，亦不代表与官方 ZeST 数值等价。
