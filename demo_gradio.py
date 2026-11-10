import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import gradio as gr
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from rembg import remove # 基于（如 U²-Net）自动分割主体移除背景，生成透明PNG
from PIL import Image
from ip_adapter import IPAdapterXL
from CustomIPAdapterXL import CustomIPAdapterXL
from ip_adapter.utils import register_cross_attention_hook, get_net_attn_map, attnmaps2images
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from datetime import datetime

import glob
import torch
import cv2
import argparse
     

import DPT.util.io

from torchvision.transforms import Compose

from DPT.dpt.models import DPTDepthModel
from DPT.dpt.midas_net import MidasNet_large
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

"""
Get ZeST Ready
"""

base_model_path = "models/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
controlnet_path = "models/controlnet-depth-sdxl-1.0"
device = "cuda"
torch.cuda.empty_cache()

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=True)
clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

def preprocess_material(material_image, text_prompt=None, threshold=0.5):
    """
    文本引导的材质图预处理
    :param material_image: PIL.Image 原始材质图
    :param text_prompt: str 文本提示（如"diamond pattern"）
    :param threshold: float 分割阈值(0-1)
    :return: PIL.Image 预处理后的材质图
    """
    # 无文本提示时直接返回原图
    if not text_prompt:
        return material_image.resize((1024, 1024))
    
    try:
        # 确保输入图像为RGB格式
        if material_image.mode != 'RGB':
            material_image = material_image.convert('RGB')
            
        print(f"Input material image mode: {material_image.mode}, size: {material_image.size}")
        
        # 图像输入是(1,3,352,352)的tensor
        image_inputs = processor.image_processor( 
            images=[material_image],
            return_tensors="pt"
        ).to(device)
        
        # 文本输出是(1,77)的tensor
        text_inputs = processor.tokenizer(
            text=[text_prompt],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        inputs = {
            **text_inputs,
            **image_inputs
        }
        
        # 运行分割模型
        with torch.no_grad():
            outputs = clipseg_model(**inputs)
        
        
        # 获取分割掩码
        mask = torch.sigmoid(outputs.logits).cpu().numpy() # 理应是(1,352,352)
        mask = (mask > threshold).astype(np.uint8) * 255
        # (1, 352, 352)
        
        # 后处理优化掩码
        mask = mask.squeeze(0)  # 移除批次维度
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)  # 膨胀
        mask = cv2.erode(mask, kernel, iterations=1)   # 腐蚀
        mask = cv2.GaussianBlur(mask, (5, 5), 0)       # 平滑
        # (352, 352)
        
        # 应用掩码到原图
        mask_img = Image.fromarray(mask).convert("L")
        mask_img = mask_img.resize(material_image.size, Image.BICUBIC) # 由于composite方法要求shape一样，调整到相同大小并平滑化
        
        # 保存掩码
        mask_dir = "./demo_assets/CLIPSeg_masks"
        os.makedirs(mask_dir, exist_ok=True)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_path = os.path.join(mask_dir, f"mask_{text_prompt}_{threshold}.png")
        mask_img.save(mask_path)
        print(f"CLIPSeg mask saved to: {mask_path}")
        
        result = Image.composite(
            material_image, 
            Image.new("RGB", material_image.size, (255, 255, 255)),  # 白色背景
            mask_img
        )
        
        # 得到mask是(352,352)
        return result.resize((1024, 1024))
    
    except Exception as e:
        print(f"预处理失败: {e}")
        return material_image.resize((1024, 1024))

# load SDXL pipeline
# 加载深度控制的 ControlNet，用于约束生成图像的几何结构
controlnet = ControlNetModel.from_pretrained(
    controlnet_path, 
    variant="fp16",  # 半精度加载
    use_safetensors=True, # 安全权重格式
    torch_dtype=torch.float16
).to(device)
# 支持基于掩码的局部修复（Inpainting）和全局生成
pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet, # 注入ControlNet 实现深度条件控制
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant="fp16",
    add_watermarker=False, # 禁用水印
).to(device)
pipe.unet = register_cross_attention_hook(pipe.unet) # 可视化交叉注意力图（调试生成过程时使用）

# 将图像条件（如材料纹理）注入到 SDXL 中，实现图像驱动的生成
ip_model = CustomIPAdapterXL(pipe, image_encoder_path, ip_ckpt, device) # 接受输入：SDXL 管道，CLIP 图像编码器路径，IP-Adapter 预训练权重


# DPT 架构->得到深度图
model_path = "DPT/weights/dpt_hybrid-midas-501f0c75.pt"
net_w = net_h = 384
model = DPTDepthModel(
    path=model_path,
    backbone="vitb_rn50_384",
    non_negative=True,
    enable_attention_hooks=False,
)
normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# 用DPT架构的 transforms 将原图缩放到384*384
transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,  # 避免拉伸变形
                ensure_multiple_of=32,  #  确保下采样时不会出现小数（很多卷积下采样五次32倍）
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,  # 双三次插值
            ),
            normalization, # 像素值归一化到[-1,1]
            PrepareForNet(), # 张量化
        ]
    )

model.eval()

def greet(input_image, material_exemplar, text_prompt, threshold):
    
    """
    主函数：实现材质转移
    :param input_image: PIL.Image 输入图像
    :param material_exemplar: PIL.Image 材质示例
    :param text_prompt: str 文本提示
    :param threshold: float CLIPSeg 分割阈值（仅预处理方案使用）
    :param scheme: str 选择方案（"预处理方案" 或 "特征融合方案"）
    :return: PIL.Image 生成的图像
    """
    
    img = np.array(input_image)  # [XX,XX,3] numpy数组

    img_input = transform({"image": img})["image"] # [3,384,384] numpy数组

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).unsqueeze(0) # 张量化->[1,C,H,W]

        # if optimize == True and device == torch.device("cuda"):
        #     sample = sample.to(memory_format=torch.channels_last)
        #     sample = sample.half()

        # 深度图生成
        prediction = model.forward(sample) # 深度预测 -> (N,H,W) (1,384,W)/(1,H,384) H/W根据元图片比例调整至合适的32倍数
        # 插值还原深度图尺寸
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1), #添加通道C维度 (1,1,H,W)
                size=img.shape[:2], # 调整到输入图像的原始尺寸 (H,W)
                mode="bicubic", # 双三次插值（平滑）
                align_corners=False,
            )
            .squeeze() #确保移除了批次和通道维度 -> (H,W)
            .cpu()
            .numpy() 
        )
    
    # 深度图归一化
    depth_min = prediction.min() # 最小深度值
    depth_max = prediction.max() # 最小深度值
    bits = 2
    max_val = (2 ** (8 * bits)) - 1 # 65535（16位）


    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (prediction - depth_min) / (depth_max - depth_min) # 归一化到 [0, max_val]
    else:
        out = np.zeros(prediction.shape, dtype=prediction.dtype) # 全零（无效情况）
    
    # 转化为8位图像并调整尺寸
    out = (out / 256).astype('uint8') # 16位 → 8位
    depth_map = Image.fromarray(out).resize((1024, 1024)) # 调整到固定尺寸(1024,1024)并转化为PIL L mode
    # depth_map.save("./demo_assets/depths/depth_output.png")
        
    
    """
    Process foreground decolored image
    """
    rm_bg = remove(input_image) # 使用 rembg 移除背景 -> 透明背景RGBA格式
    # 得到白色前景黑色背景图
    target_mask = rm_bg.convert("RGB").point(lambda x: 0 if x < 1 else 255).convert('L').convert('RGB') # 丢弃透明度通道,将RGBA转为RGB并二值化操作->L_>RGB
    # 背景灰度化
    mask_target_img = ImageChops.lighter(input_image, target_mask) # 逐像素取两输入图像更亮值，即保持前景为白色 背景变为原始输入颜色
    invert_target_mask = ImageChops.invert(target_mask) # 反转掩码 即白色背景黑色前景
    gray_target_image = input_image.convert('L').convert('RGB') # 原图转为灰度图（再转RGB）
    gray_target_image = ImageEnhance.Brightness(gray_target_image) # 亮度调整器
    factor = 1.0  # 亮度系数可调（1为原图）
    gray_target_image = gray_target_image.enhance(factor) # 调整亮度
    # 前景灰度化
    grayscale_img = ImageChops.darker(gray_target_image, target_mask) # 原图灰度图和前景图取更暗，即保留前景灰度，背景全黑
    img_black_mask = ImageChops.darker(input_image, invert_target_mask) # 原图和反转掩码前景图取更暗，即前景全黑，背景保留原图（这里用input或者mask_target都可）
    grayscale_init_img = ImageChops.lighter(img_black_mask, grayscale_img) # 上俩图取更亮，得到原图背景和灰度前景
    init_img = grayscale_init_img
    # 得到保留了光照信息和背景原始信息的输入图
    
    """
    Process material exemplar and resize all images
    """
    # 预处理材质示例图像
    # processed_material = preprocess_material(material_exemplar, text_prompt, threshold)
    
    # 最终输入用到的三个图resize:前景图、材料图、I_init(保留光照和背景的灰度图)
    ip_image = material_exemplar
    init_img = init_img.resize((1024,1024))
    mask = target_mask.resize((1024, 1024))
    
    # 修复扩散模型（四个输入）
    num_samples = 1
    images = ip_model.generate(
        pil_image=ip_image,  # 材料图（风格参考）
        text_prompt=text_prompt,  # 传递文本提示用于第二阶段融合
        image=init_img,  # I_init（光照/背景保留）
        control_image=depth_map,  # 深度图（控制几何结构）
        mask_image=mask,  # 前景图（限定编辑区域）
        controlnet_conditioning_scale=0.9, # ControlNet 权重（0.9表示强控制）
        num_samples=num_samples, # 生成1张结果
        num_inference_steps=30, # 扩散模型推理步数（越多越精细）
        seed=42 # 随机种子（固定可复现）
    )
    
    # 保存生成图像
    output_dir = "demo_assets/outputs"
    os.makedirs(output_dir, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"output_{text_prompt}_{threshold}.png")
    images[0].save(output_path)
    print(f"outputs image saved to: {output_path}")
    
    return images[0]

input_image = gr.Image(type="pil", label="Input Image")
input_image2 = gr.Image(type="pil", label="Material Exemplar")
text_prompt_input = gr.Textbox(label="Text Prompt for Material Selection (optional)", placeholder="Enter text to select specific part of material, e.g., 'diamond pattern'")
# threshold_input = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.3, label="Threshold")

demo = gr.Interface(
    fn=greet,
    inputs=[input_image, input_image2, text_prompt_input],
    title="ZeST: Zero-Shot Material Transfer from a Single Image",
    description="Upload two images -- input image and material exemplar. Optionally provide a text prompt to select specific parts of the material exemplar. ZeST extracts the material from the exemplar and casts it onto the input image following the original lighting cues.",
    outputs=["image"],
    allow_flagging='never'
)

demo.launch()
