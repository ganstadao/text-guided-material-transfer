

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import os
from typing import List

import torch
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModel
from ip_adapter import IPAdapterXL

from ip_adapter.utils import is_torch2_available, get_generator
from diffusers import UNet2DConditionModel
from utils import get_heatmap, upsample_mask, get_soft_mask

class CustomIPAdapterXL(IPAdapterXL):

    """Custom SDXL"""

    def get_image_text_embeds(self, pil_image=None, text_prompt=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values # -> ([1, 3, 224, 224])
            # 这里用的是CLIPVisionModelWithProjection类
            vision_outputs = self.image_encoder.vision_model(clip_image.to(self.device, dtype=torch.float16))

            clip_image_embeds = vision_outputs.last_hidden_state.to(dtype=torch.float16) # -> ([1, 257, 1280])
            print("original image embeds shape: ", clip_image_embeds.shape)

            if text_prompt:
                print("text prompt exists, input position feature...")
                # Text encoder
                tokenizer1 = CLIPTokenizer.from_pretrained("models/stable-diffusion-xl-base-1.0",subfolder="tokenizer")
                text_encoder1 = CLIPTextModel.from_pretrained("models/stable-diffusion-xl-base-1.0",subfolder="text_encoder",variant="fp16").to(self.device)

                tokenizer2 = CLIPTokenizer.from_pretrained("models/stable-diffusion-xl-base-1.0",subfolder="tokenizer_2")
                text_encoder2 = CLIPTextModel.from_pretrained("models/stable-diffusion-xl-base-1.0",subfolder="text_encoder_2",variant="fp16").to(self.device)

                def encode_text(prompt, tokenizer, encoder):
                    inputs = tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        output = encoder(**inputs).last_hidden_state  # [1,77, dim]
                    return output, inputs

                # 生成768维和1280维特征
                text_feat1, inputs1 = encode_text(text_prompt, tokenizer1, text_encoder1)  # [1,77,768]
                text_feat2, inputs2 = encode_text(text_prompt, tokenizer2, text_encoder2)  # [1,77,1280]

                # 拼接成2048维（SDXL UNet要求的文本特征维度）
                text_features = torch.cat([text_feat1, text_feat2], dim=-1)  # [1,77, 768+1280=2048]

                text_features = text_features.to(dtype=torch.float16)
                print("text feature shape: ", text_features.shape) # [1,77,2048]

                print("start cross attention...")
                unet = UNet2DConditionModel.from_pretrained("models/stable-diffusion-xl-base-1.0",variant="fp16",subfolder="unet").to("cuda").half()
                attn_layer = unet.down_blocks[2].attentions[0].transformer_blocks[0]

                # 做image_embeds和text_prompt的交叉注意力
                new_outputs = attn_layer(
                    hidden_states=clip_image_embeds,
                    encoder_hidden_states=text_features
                ) # -> [1,257,1280]

                #通过注意力分数来计算对应图像编码的软mask
                soft_mask = get_soft_mask(attn_layer, clip_image_embeds, text_features, tokenizer1, inputs1, text_prompt)

                # 4. 应用软mask约束材质迁移区域
                new_outputs = new_outputs * soft_mask  # 只在高关注区域强化文本引导的材质特征

                # 可视化软mask热力图
                get_heatmap(soft_mask, pil_image)

                # 这里需严格按照原始代码结构经过pooler得到新的pooler_output
                pooler = self.image_encoder.vision_model.post_layernorm  # 这是一个包含线性层的模块
                pooled_output=pooler(new_outputs)[:,0,:]
                # 手动得到新的pooler_output


            else:
                # 是否有定位text prompt的区别就在于得到的pooler_output不同
                pooled_output=vision_outputs.pooler_output

            pooled_output = pooled_output.to(device=self.device, dtype=torch.float16)
            print("pooler_output shape: ", pooled_output.shape)

            # 调用投影模型[1,1280]->[1,1024]
            localized_image = self.image_encoder.visual_projection(pooled_output).to(device=self.device,dtype=torch.float16)

        else:
            localized_image = clip_image_embeds # 如果没有text_prompt，使用pooled embeds

        # 投影 pooled to [1,4,2048]
        print("input projection model...")
        image_prompt_embeds = self.image_proj_model(localized_image)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(localized_image))
        print("output pooled image embeds shape: ",image_prompt_embeds.shape)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs, # 从中接受text_prompt定位文本
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        print("start image embedding...")
        # 在源代码基础上加入定位信息融入，输入输出shape不变
        text_prompt = kwargs.get('text_prompt', None)
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_text_embeds(
            pil_image,text_prompt
        )

        # 后续操作不变
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            print("start prompt embedding...")
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            print("prompt embeds shape: ",prompt_embeds.shape)
            print("start cat...")
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
            print("final pipe embeds: ",prompt_embeds.shape)


        self.generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images
