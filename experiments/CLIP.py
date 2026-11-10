
# 特征融合研究
import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor
from PIL import Image

# 目标是融合图像材质特征和文本定位信息特征

device="cuda"
prompt="CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision."

# CLIP IMAGE ENCODER
image_processor= CLIPImageProcessor.from_pretrained(
    "models/image_encoder",
    variant="fp16",  # 半精度加载
    use_safetensors=True, # 安全权重格式
    torch_dtype=torch.float16                                          
)
vision_model = CLIPVisionModel.from_pretrained("models/image_encoder").to(device)

# CLIP TEXT ENCODER
tokenizer = CLIPTokenizer.from_pretrained("models/image_encoder")
text_model = CLIPTextModel.from_pretrained("models/image_encoder").to(device)

image = Image.open(dir)
inputs = image_processor(
    image=image, return_tensors='pt'
).to(device)

with torch.no_grad():
        image_features = vision_model(**inputs).last_hidden_state
        # 获取图像的池化特征（通常使用[CLS]标记）
        image_pooled = vision_model(**inputs).pooler_output
    
print("图像编码结果形状:", image_features.shape)
print("图像池化特征形状:", image_pooled.shape)
print("图像编码结果示例:", image_features[0, 0, :5])  # 打印第一个token的前5个值
print("图像池化特征示例:", image_pooled[0, :5])  # 打印池化特征的前5个值


'''text=tokenizer(
    text=prompt,
    padding="max_length",
    max_length=77,
    return_tensors="pt"
)

print(text)'''


