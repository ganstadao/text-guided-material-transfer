# 实现与验证边界

对照来源：
- ZeST README: https://github.com/ttchengab/zest_code
- 官方 batch: https://github.com/ttchengab/zest_code/blob/main/run_batch.py
- IP-Adapter: https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py
- Diffusers 0.32.2 loader: https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/loaders/ip_adapter.py

与旧仓库比较：

1. 运行入口不在 import 时创建模型、启动 UI 或修改 HF_ENDPOINT。
2. 使用固定版本 Diffusers 原生 IP-Adapter，不依赖遗漏的 ip_adapter 目录；基础 ViT-H 四 token checkpoint，不使用 Plus checkpoint。
3. 目标 mask 从 alpha 读取；参考 mask 和目标 mask 分开保存。CLIPSeg 默认使用原图尺寸的阈值二值 mask，不沿用旧脚本的膨胀／腐蚀／模糊。该预处理变化需要在报告中说明。
4. 自定义融合保留旧 prototype 的 SDXL transformer block + 两个 text encoder last hidden states 拼接 + sigmoid attention mask + CLS pooling 路线。块在安装 IP attention processor 前复制，否则材质文本末四 token 会被误当作 IP tokens。只复制一个 block，避免每张图加载整个 UNet。
5. 注意力分数按真实多头拆分计算并均值聚合，短语取全部非特殊 token。仍然保留原方法对 CLS 乘 mask 的行为；patch 热图是诊断信息，不能据此证明材质局部匹配。
6. 自动深度采用 Transformers DPT hybrid，不保证与原 DPT transform 完全相同；正式四组实验使用同一预计算 depth/mask。
7. 默认 guidance_scale=5、strength=0.9999、steps=30、ControlNet=0.9、IP scale=1、seed=42。SDXL pipeline 默认 scheduler 保持加载配置；主干加载耗时不计入生成时间。

轻量测试仅证明配置、mask/depth 处理、attention 张量计算、四组调度和结果协议符合预期。记录型假管线测试不证明 SDXL/ControlNet/IP-Adapter 真实兼容性或图像质量。真模型验收已在 NVIDIA A40 上完成：同一官方示例四组生成，mask/depth/init 与代码指纹一致，中间参考 mask、attention 图、输出和记录完整。另验证了 rembg 自动目标 mask + DPT hybrid 自动深度 + A 组生成。9 项测试、pip check 和 Gradio HTTP/config/API schema 检查通过。

这些检查只证明已测输入上的运行与数据流，不证明组件效果提升。验证范围与后续工作见 [PROJECT_STATUS.md](PROJECT_STATUS.md)。
