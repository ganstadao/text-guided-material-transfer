# 小组数据与评价更新检查

检查对象：[jerryluo-data 分支](https://github.com/ganstadao/text-guided-material-transfer/tree/jerryluo-data)，
提交 `5b30c323d5c4e0afefb3afca9261167146e14d86`，提交时间 2026-11-12 00:13（UTC+8）。
获取远端时 main 仍为 `9f735a0`，新提交尚未进入 main。
本次只 fetch，并在 `.cache/review-jerryluo-5b30c32` 建立隔离 worktree 检查，未合并到本地工作代码。

## 新增内容

- 6 张目标图、4 张参考图，均为 512×512 PNG。
- 6×4=24 个组合的 manifest，材质 wood/metal/fabric/stone，seed 按行从 42 到 65。
- 背景 MAE/SSIM、参考分割 IoU/Dice，以及可选 LPIPS/CLIP 的评价脚本。
- 人工盲评方案与评分表表头。未新增训练或生成方法，也未修改核心 pipeline。

24 行清单通过现有核心 CLI 的 validate-only。组合跨方法公平性应按相同 pair_id 配对；
按此清单做四组共 96 次生成。24 个组合共享 6 个目标和 4 个参考，并非 24 个独立来源样本。

## 必须修正的评价问题

### P1：二值 mask 被再次按 0–255 使用

[compute_metrics.py:148–149](https://github.com/ganstadao/text-guided-material-transfer/blob/5b30c323d5c4e0afefb3afca9261167146e14d86/evaluation/compute_metrics.py#L148-L149)
将 mask 二值化为 0/1 后使用 `mask < 128`，所有像素都会成为背景。
[object_crop:103–106](https://github.com/ganstadao/text-guided-material-transfer/blob/5b30c323d5c4e0afefb3afca9261167146e14d86/evaluation/compute_metrics.py#L103-L106)
又使用 `mask > 127`，因此裁剪找不到目标，返回全图。

已复现：4×4 图中只有中心 2×2 前景变化，背景应为 12 像素、背景 MAE 应为 0；
当前得到背景 16 像素、MAE=0.25，目标 crop 仍为 4×4，预期是 2×2。
应统一使用布尔/0–1 mask，背景用 mask==0，目标范围用 mask>0。

### P1：默认结果与原图尺寸不同会中断评分

[compute_metrics.py:146–151](https://github.com/ganstadao/text-guided-material-transfer/blob/5b30c323d5c4e0afefb3afca9261167146e14d86/evaluation/compute_metrics.py#L146-L151)
直接相减 result 和 target；核心引擎保存的 target 是输入原尺寸，result 默认为 1024×1024。
此次新增 target 全部为 512×512，因此按 README 默认配置生成后评分会触发广播错误。

已复现：1024×1024×3 与 512×512×3 计算 background_mae 抛出 ValueError。
评价前应把原图、目标 mask 按预先约定的生成预处理规则对齐到 result 尺寸，
例如 RGB 使用相同的缩放方式，mask 使用最近邻；不能仅修函数阈值而忽略空间对齐。

### P2：可选“背景 LPIPS”实际计算全图

[compute_metrics.py:152–153](https://github.com/ganstadao/text-guided-material-transfer/blob/5b30c323d5c4e0afefb3afca9261167146e14d86/evaluation/compute_metrics.py#L152-L153)
调用 lpips_score(result, target)，函数内也没有背景 mask。
该值包括预期发生材质改变的前景，不能命名为背景保留指标。
应明确实现并记录背景区域评价协议，或改名为全图 LPIPS 并说明其含义。

## 数据还需要补齐的部分

1. manifest 的 mask/depth 全空。核心 CLI 会触发自动处理，但本地固定输入消融入口会明确拒绝；
   已实测首行报 `p001: fixed mask and depth are required for ablation`。
   建议为 6 张目标各预计算并人工确认 1 张 mask、1 张 depth，再让 24 行复用对应文件。
2. 分支没有 masks_exemplar_gt 文件。当前 B/D 的 IoU/Dice 将保持 NaN，不能据此报告分割效果。
   按现有脚本，需要给使用 e03/e04 的 12 个 pair_id 准备 GT 文件；也可改成按参考图查找 GT，从而只维护两张。
3. e03 是布料特写，e04 是纯背景的孤立石块，与 README 所说“多种材质/物体的杂乱参考图”不符。
   它们可以作为材质参考，但若要研究参考分割抵抗干扰的作用，应补充真实多材质参考，或修正分组命名。
   目标图背景复杂与参考图复杂是两个不同变量，不能互相替代。
4. t06 同时包含电脑、风扇、书本等，待编辑对象不明确；t04 的沙发上有衣物。
   应先约定具体编辑区域，再确认目标 mask，避免把自动抠图结果直接当标注。

## 与本地流程衔接

- 数据和评价代码主要是新增文件，与本地低显存引擎改动互补，但本次尚未合并。
- 本地 run_ablation 的 --seeds 会覆盖清单每行的 seed；若采用成员清单的 42–65，需明确保留逐行种子的运行协议。
- 评价本地四组输出时，--outputs 应指向 `outputs/某实验/seed_42` 这一层，才能找到 A/B/C/D。
- 先修评价逻辑并补齐固定输入，再跑新的样本；不需要重写生成 pipeline。
- 当前检查不修改远端、不发送消息，也未替成员提交修复。
