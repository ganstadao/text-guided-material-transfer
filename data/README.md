# 数据集要求与来源（成员 2 用）

清单 `manifest.csv` 已按 6 张目标图 × 4 张材质参考图生成 24 对，文件名按行引用。
图片按下面的约定放进对应目录，然后用 `--validate-only` 检查，**检查通过后清单不再改动**。

## 图片要求

**目标图 `targets/t01.png` ~ `t06.png`**
- 单张主体突出的照片，分辨率至少 512×512，建议直接裁成正方形（非正方形输入会被拉伸到 1024×1024）
- 背景：t01、t02、t03 用干净背景；t04、t05、t06 用多物体、遮挡或杂乱背景
- 物体类别至少 3 类（模板默认：杯、南瓜、椅子、鞋、灯/花瓶、包），类别可以换，但要先改 README 备注再定稿
- 避免：水印、文字、过暗或过曝

**目标 mask（清单 mask 列已留空，自动抠图）**
- 仓库会用 rembg 自动生成 mask（白 = 要换材质的物体，黑 = 背景），四组共享同一输入时结果一致
- 若要人工指定：灰度 PNG 放进 `masks/t01.png` ~ `t06.png`，与目标图**像素尺寸完全一致**，再在 `manifest.csv` 填回路径
- 自动抠图结果建议人工检查、修边缘后再定稿

**纯材质参考图 `exemplars/e01.png`（wood）、`e02.png`（metal）**
- 材质铺满整张图、正面视角、光照均匀，无明显阴影或反光爆炸
- 建议从 DTD 选取后放大/裁成 ≥512×512

**杂乱材质参考图 `exemplars/e03.png`（fabric）、`e04.png`（stone）**
- 真实照片，画面里同时存在多种材质/物体
- 文字提示（`fabric` / `stone`）必须能唯一对应其中一种材质区域
- 为这两张图各标注一张目标材质区域 GT mask，存 `masks_exemplar_gt/{pair_id}.png`（涉及 e03/e04 的 12 个 pair_id：p003、p004、p007、p008、p011、p012、p015、p016、p019、p020、p023、p024），用来检查 CLIPSeg 的 IoU
- 同一参考图被多个 pair 使用，GT mask 只需按 pair_id 复制同一张

**通用**
- 图片用 PNG 或高质量 JPEG；mask 一律 PNG，避免 JPEG 压缩边缘
- 每张图来源记进 `data/sources.md`（来源、文件、许可），报告要写

## 现成数据集

- **DTD（纯纹理首选）**：Oxford VGG，47 类约 5640 张，每类 120 张，尺寸 300~640。类别是感知属性词（banded/blotchy/braided/.../woven/zigzagged），**没有 wooden/metallic/fabric/stone 文件夹**，需按属性词挑（wood→fibrous/grooved/veined，metal→crystalline/perforated/meshed，stone→marbled/porous/pitted）。https://robots.ox.ac.uk/~vgg/data/dtd/
- **OpenSurfaces（杂乱参考图首选）**：康奈尔的真实照片 + 材质区域标注，可直接参考它的材质分割。https://vision.cs.cornell.edu/opensurfaces/ （网站不稳时改用自己拍的照片）
- **COCO**：目标物体照片，找目标图候选。https://cocodataset.org/
- **ADE20K**：场景照片 + 物体分割标注，可作杂乱参考图补充。https://groups.csail.mit.edu/vision/datasets/ADE20K/
- **自己手机拍**：杂乱多材质场景最省事且无许可问题，推荐 e03/e04 用实拍

所有数据仅课程内部使用；提交 ZIP 不包含图片数据，报告里写明来源与选择规则。

## 检查命令

```bash
python run_batch.py --config configs/A.json --manifest data/manifest.csv --output outputs/A --validate-only
```

`depth` 列留空时会自动用 DPT 生成深度图；同一输入与 seed 下结果确定，四组共享一致。若之后改为预计算深度，把 PNG 放进 `depths/` 并在清单里填上路径。
