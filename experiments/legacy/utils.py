
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import torch

# 输入注意力层（以算出注意力分数即token和image位置相关性）和image_embeds[1,257,1280]
def get_soft_mask(attn_layer, clip_image_embeds, text_features, tokenizer1,inputs1, text_prompt):
    # 1. 先计算跨注意力层的 query/key/value（为get_attention_scores准备输入）
    attn2=attn_layer.attn2
    # query（Q）：图像特征→通过to_q投影得到
    q = attn2.to_q(clip_image_embeds)  # shape: [1, 257, 2048]（图像特征）→ [1, 257, 2048]（Q）
    # key（K）和value（V）：文本特征→通过to_k/to_v投影得到
    k = attn2.to_k(text_features)      # shape: [1, 77, 2048]（文本特征）→ [1, 77, 2048]（K）
    v = attn2.to_v(text_features)      # shape: [1, 77, 2048]（文本特征）→ [1, 77, 2048]（V）

    # 2. 调用get_attention_scores计算注意力权重（核心步骤）
    # 第三个参数是attention_mask，这里不需要，传None即可
    attn_weights = attn2.get_attention_scores(q, k, attention_mask=None)
    print("通过get_attention_scores获取的权重 shape:", attn_weights.shape)
    # 正常输出应为：[1, 32, 257, 77]（batch=1，32个头，257个图像token，77个文本token）

    # 3. 后续生成软mask（和之前逻辑一致）
    # 找到文本关键词（如“copper”）对应的token索引
    text_tokens = tokenizer1.convert_ids_to_tokens(inputs1.input_ids[0])
    target_token_idx_list = [i for i, token in enumerate(text_tokens) if text_prompt in token.lower()]

    if len(target_token_idx_list) == 0:
        print(f"警告：未找到关键词'copper'，默认取第1个有效token（跳过起始符）")
        target_token_idx = 1  # 默认取第1个非起始符token（起始符通常是0索引）
    else:
        target_token_idx = target_token_idx_list[0]  # 取第一个匹配的索引

    # 处理权重生成软mask
    attn_weights = attn_weights.squeeze(0)  # [257, 77]（去掉batch维度）
    # 对所有注意力头取平均，聚焦目标文本token
    soft_mask = attn_weights[:, target_token_idx]  # [257]（图像token对“copper”的关注度）
    # 扩展维度以匹配图像特征
    soft_mask = soft_mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, clip_image_embeds.shape[-1])  # [1,257,1280]
    # 增强mask对比度（让关注区域更突出）
    soft_mask = torch.sigmoid(soft_mask * 5)

    return soft_mask

def upsample_mask(mask, target_size=(512, 512)):
    # 将torch tensor转为numpy数组
    mask_np = mask.cpu().detach().numpy()
    # 转为PIL图像格式（需先扩展为3通道，再转灰度）
    mask_pil = transforms.ToPILImage()(torch.tensor(mask_np).unsqueeze(0)).convert("L")
    # 上采样到目标尺寸（如512×512，与生成图尺寸一致）
    mask_upsampled = mask_pil.resize(target_size, Image.BILINEAR)
    # 转回numpy数组（值范围0~1）
    mask_upsampled_np = np.array(mask_upsampled) / 255.0

    return mask_upsampled_np

def get_heatmap(soft_mask, pil_image):

    # ===================== 1. 处理软mask：去掉CLS token + 调整形状 =====================
    # soft_mask初始形状：[1,257,2048]（batch=1，257个token，2048维特征）
    # 第一步：取mask的“关注度权重”（每个token的平均关注度，忽略特征维度）
    mask_token = soft_mask.squeeze(0)  # [257,2048]（去掉batch维度）
    mask_token = mask_token.mean(dim=-1)  # [257]（对特征维度取平均，得到每个token的关注度）

    # 第二步：去掉CLS token（第0个token是全局特征，不是图像patch）
    # 剩余196个token对应 CLIP的14×14 patch（16*16=256）
    mask_patch = mask_token[1:]  # [256]（去掉CLS token，保留图像patch token）
    mask_patch = mask_patch.reshape(16, 16)  # [16,16]（还原为patch网格）
    print("处理后patch级mask shape:", mask_patch.shape)  # 输出 [16,16]

    # ===================== 2. 上采样：将14×14 mask放大到原图尺寸 =====================
    # 定义上采样函数（用双线性插值放大，更平滑）

    # 上采样到目标尺寸（建议与你生成图像的尺寸一致，如512×512/1024×1024）
    target_size = (512, 512)  # 可根据你的需求修改
    mask_upsampled = upsample_mask(mask_patch, target_size)
    print("上采样后像素级mask shape:", mask_upsampled.shape)  # 输出 [512,512]


    # 上采样到目标尺寸（建议与你生成图像的尺寸一致，如512×512/1024×1024）
    target_size = (512, 512)  # 可根据你的需求修改
    mask_upsampled = upsample_mask(mask_patch, target_size)
    print("上采样后像素级mask shape:", mask_upsampled.shape)  # 输出 [512,512]


    # ===================== 3. 读取原图并预处理（与mask尺寸对齐） =====================
    # 读取你输入的原始图像（pil_image是你传入的材质参考图）
    # 若pil_image是列表，取第一张（单图场景）
    original_img = pil_image[0] if isinstance(pil_image, list) else pil_image
    # 将原图缩放到与mask一致的尺寸（target_size）
    original_img_resized = original_img.resize(target_size, Image.BILINEAR)
    # 转为numpy数组（用于绘图）
    original_img_np = np.array(original_img_resized)


    # ===================== 4. 绘制热力图（与原图叠加） =====================
    plt.figure(figsize=(10, 5))  # 设置画布大小

    # 子图1：原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(original_img_resized)
    plt.title("Original Image")
    plt.axis("off")  # 隐藏坐标轴

    # 子图2：原图+软mask热力图叠加
    plt.subplot(1, 2, 2)
    # 先画原图
    plt.imshow(original_img_np)
    # 再画热力图，alpha控制透明度（0.5表示半透明，既看得到原图也看得到热力图）
    heatmap = plt.imshow(mask_upsampled, cmap="jet", alpha=0.5, extent=[0, target_size[0], target_size[1], 0])
    # 添加颜色条（表示关注度高低）
    cbar = plt.colorbar(heatmap)
    cbar.set_label("Attention Score (High → Red)")
    plt.title("Soft Mask Heatmap (Overlay)")
    plt.axis("off")

    # ===================== 5. 保存或显示热力图 =====================
    # 保存热力图（可修改保存路径）
    save_path = "./soft_mask_heatmap.png"
    plt.tight_layout()  # 调整子图间距
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # dpi=300保证高清
    plt.close()  # 关闭画布，避免内存占用

    print(f"软mask热力图已保存至：{save_path}")
