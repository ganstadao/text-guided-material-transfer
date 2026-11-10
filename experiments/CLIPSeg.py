from PIL import Image
import requests
import os
import torch
import numpy as np
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
url = "https://farm4.staticflickr.com/3487/3925656789_1b64654c91_z.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 加载CLIPSeg模型
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation, CLIPSegModel

device="cuda"
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined",use_fast=True)
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

print(f"Input image mode: {image.mode}, size: {image.size}")

# 文本提示定位目标物体
prompts = "player"
# 分别处理文本和图像
text_inputs = processor.tokenizer(
    text=[prompts],
    padding="max_length",
    truncation=True,
    return_tensors="pt"
).to(device)

image_inputs = processor.image_processor(
    images=[image],
    return_tensors="pt"
).to(device)

print(f"image_inputs keys: {image_inputs.keys()}, pixel_values shape: {image_inputs['pixel_values'].shape}")
print(f"text_inputs keys: {text_inputs.keys()}, input_ids shape: {text_inputs['input_ids'].shape}")

# 合并结果
inputs = {
    **text_inputs,
    **image_inputs
}

outputs = model(**inputs)
preds = outputs.logits # torch.Size([4, 1, 352, 352]) [-10,4]

print(preds.shape)

'''for i in range(len(prompts)):
    output_img=torch.sigmoid(preds.detach()[i][0]).numpy() * 255
    output_img=output_img.astype(np.uint8)
    output_img=Image.fromarray(output_img).convert('RGB')
    output_img.save(f"output_img{i}.png")'''