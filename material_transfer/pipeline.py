"""Lazy model ownership and a reusable, serial inference API."""
import copy
import time
from PIL import Image
import numpy as np
from .config import TransferConfig
from .images import automatic_mask, decolorize


class MaterialTransfer:
    def __init__(self, config: TransferConfig):
        self.config = config
        self.pipe = None
        self.fusion_block = None
        self.segmenter = None
        self.depth_estimator = None

    def load(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline
        from transformers import CLIPVisionModelWithProjection
        cfg = self.config
        if cfg.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not visible in this session; run on an allocated GPU host")
        dtype = getattr(torch, cfg.dtype)
        common = dict(torch_dtype=dtype, local_files_only=cfg.local_files_only)
        controlnet = ControlNetModel.from_pretrained(cfg.controlnet, variant=cfg.model_variant, **common)
        encoder = CLIPVisionModelWithProjection.from_pretrained(
            cfg.adapter_repo, subfolder=cfg.image_encoder_subfolder, **common)
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            cfg.base_model, controlnet=controlnet, image_encoder=encoder,
            add_watermarker=False, variant=cfg.model_variant, **common)
        # Copy before installing IP processors, which would otherwise consume the
        # last four material-text tokens as IP tokens in the fusion branch.
        if cfg.fuse_attention:
            self.fusion_block = copy.deepcopy(
                pipe.unet.down_blocks[2].attentions[0].transformer_blocks[0]
            ).to(device=cfg.device, dtype=dtype).eval()
        pipe.load_ip_adapter(cfg.adapter_repo, subfolder=cfg.adapter_subfolder,
                             weight_name=cfg.adapter_weight, image_encoder_folder=None,
                             local_files_only=cfg.local_files_only)
        pipe.set_ip_adapter_scale(cfg.adapter_scale)
        pipe.to(cfg.device)
        for model in (pipe.unet, pipe.vae, pipe.image_encoder, pipe.text_encoder, pipe.text_encoder_2, pipe.controlnet):
            model.eval()
        pipe.vae.enable_slicing()
        self.pipe = pipe

    def reference(self, image, phrase):
        cfg = self.config
        if not cfg.segment_reference:
            return image, None
        import torch
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        if self.segmenter is None:
            kw = dict(local_files_only=cfg.local_files_only)
            self.segmenter = (
                CLIPSegProcessor.from_pretrained(cfg.clipseg_model, **kw),
                CLIPSegForImageSegmentation.from_pretrained(cfg.clipseg_model, **kw).to(cfg.device).eval())
        processor, model = self.segmenter
        inputs = dict(processor.tokenizer([phrase], padding=True, truncation=True, return_tensors="pt"))
        inputs.update(processor.image_processor(images=[image], return_tensors="pt"))
        inputs = {name: value.to(cfg.device) for name, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs, interpolate_pos_encoding=True).logits
            logits = logits.reshape(1, 1, *logits.shape[-2:])
            probs = torch.nn.functional.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False).sigmoid()[0, 0]
        mask = Image.fromarray((probs.cpu().numpy() > cfg.segmentation_threshold).astype("uint8") * 255)
        if mask.getbbox() is None:
            raise ValueError("CLIPSeg returned an empty reference mask; inspect phrase/threshold")
        return Image.composite(image, Image.new("RGB", image.size, "white"), mask), mask

    def depth(self, target):
        import torch
        from transformers import DPTImageProcessor, DPTForDepthEstimation
        cfg = self.config
        if self.depth_estimator is None:
            kw = dict(local_files_only=cfg.local_files_only)
            self.depth_estimator = (
                DPTImageProcessor.from_pretrained(cfg.depth_model, **kw),
                DPTForDepthEstimation.from_pretrained(cfg.depth_model, **kw).to(cfg.device).eval())
        processor, model = self.depth_estimator
        inputs = processor(images=target, return_tensors="pt").to(cfg.device)
        with torch.inference_mode():
            prediction = model(**inputs).predicted_depth
            prediction = torch.nn.functional.interpolate(prediction[:, None], size=(target.height, target.width), mode="bicubic", align_corners=False)[0, 0]
        array = prediction.float().cpu().numpy()
        span = np.ptp(array)
        array = (array - array.min()) / span if span > 1e-8 else np.zeros_like(array)
        return Image.fromarray((array * 255).astype("uint8")).convert("RGB")

    def run(self, target, exemplar, material_prompt="", target_mask=None, depth=None):
        import torch
        cfg = self.config
        if (cfg.segment_reference or cfg.fuse_attention) and not material_prompt.strip():
            raise ValueError("Methods B/C/D require a material_prompt")
        self.load()
        cuda = cfg.device.startswith("cuda")
        if cuda:
            torch.cuda.synchronize(cfg.device)
            torch.cuda.reset_peak_memory_stats(cfg.device)
        start = time.perf_counter()
        target, exemplar = target.convert("RGB"), exemplar.convert("RGB")
        if target_mask is None:
            target_mask = automatic_mask(target)
        if target_mask.size != target.size:
            raise ValueError("Target mask must have the same size as the target")
        if target_mask.convert("L").getbbox() is None:
            raise ValueError("Target mask is empty")
        target_mask = target_mask.convert("L")
        depth = depth if depth is not None else self.depth(target)
        size = (cfg.width, cfg.height)
        init = decolorize(target, target_mask).resize(size, Image.Resampling.LANCZOS)
        mask = target_mask.resize(size, Image.Resampling.NEAREST)
        control = depth.convert("RGB").resize(size, Image.Resampling.BICUBIC)
        reference, ref_mask = self.reference(exemplar, material_prompt)
        attn_map = None
        with torch.inference_mode():
            if cfg.fuse_attention:
                from .attention import fuse_reference
                embedding, attn_map = fuse_reference(self.pipe, reference, material_prompt, self.fusion_block)
                # Diffusers ImageProjection expects unprojected pooled CLIP embeddings.
                embeddings = embedding[:, None, :]
                if cfg.guidance_scale > 1:
                    embeddings = torch.cat([torch.zeros_like(embeddings), embeddings])
                conditioning = dict(ip_adapter_image_embeds=[embeddings])
            else:
                conditioning = dict(ip_adapter_image=reference)
            if cuda:
                torch.cuda.synchronize(cfg.device)
            prepared = time.perf_counter()
            image = self.pipe(
                prompt=cfg.prompt, negative_prompt=cfg.negative_prompt,
                image=init, mask_image=mask, control_image=control,
                height=cfg.height, width=cfg.width, strength=cfg.strength,
                guidance_scale=cfg.guidance_scale,
                controlnet_conditioning_scale=cfg.controlnet_scale,
                num_inference_steps=cfg.steps,
                generator=torch.Generator(device=cfg.device).manual_seed(cfg.seed),
                **conditioning).images[0]
        if cuda:
            torch.cuda.synchronize(cfg.device)
        elapsed = time.perf_counter()
        artifacts = dict(result=image, target=target, exemplar=exemplar,
                         target_mask=target_mask, depth=control, init=init, reference=reference)
        if ref_mask is not None:
            artifacts["reference_mask"] = ref_mask
        if attn_map is not None:
            artifacts["attention_mask"] = attn_map
        metadata = dict(preprocess_seconds=prepared-start, generation_seconds=elapsed-prepared,
                        peak_cuda_bytes=torch.cuda.max_memory_allocated(cfg.device) if cuda else None,
                        segmentation_applied=cfg.segment_reference, attention_applied=cfg.fuse_attention)
        return artifacts, metadata
