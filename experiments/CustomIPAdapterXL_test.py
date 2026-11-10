import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from CustomIPAdapterXL import CustomIPAdapterXL
from ip_adapter import IPAdapterXL
from diffusers import StableDiffusionXLPipeline, ControlNetModel
from PIL import Image

# 测试CustomIPAdapterXL

base_model_path = "models/stable-diffusion-xl-base-1.0"
image_encoder_path = "models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl_vit-h.bin"
controlnet_path = "models/controlnet-depth-sdxl-1.0"

device="cuda"
num_samples=1

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    use_safetensors=True,
    torch_dtype=torch.float16,
    variant="fp16",
    add_watermarker=False, # 禁用水印    
).to(device)

ip_model=CustomIPAdapterXL(pipe,image_encoder_path,ip_ckpt,device)

ip_image=Image.open("./demo_assets/material_exemplars/glass wood metal.jpg")
text_prompt="copper"

images=ip_model.generate(
    pil_image=ip_image,  # 材料图（风格参考）
    text_prompt=text_prompt,  # 传递文本提示用于第二阶段融合
    # image=init_img,  # I_init（光照/背景保留）
    # control_image=depth_map,  # 深度图（控制几何结构）
    # mask_image=mask,  # 前景图（限定编辑区域）
    # controlnet_conditioning_scale=0.9, # ControlNet 权重（0.9表示强控制）
    num_samples=num_samples, # 生成1张结果
    num_inference_steps=30, # 扩散模型推理步数（越多越精细）
    # seed=42 # 随机种子（固定可复现）
)

images[0].save("./test.png")
