from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class TransferConfig:
    method: str = "A"
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    model_variant: str | None = "fp16"
    controlnet: str = "diffusers/controlnet-depth-sdxl-1.0"
    adapter_repo: str = "h94/IP-Adapter"
    adapter_subfolder: str = "sdxl_models"
    adapter_weight: str = "ip-adapter_sdxl_vit-h.safetensors"
    image_encoder_subfolder: str = "models/image_encoder"
    clipseg_model: str = "CIDAS/clipseg-rd64-refined"
    depth_model: str = "Intel/dpt-hybrid-midas"
    device: str = "cuda"
    dtype: str = "float16"
    local_files_only: bool = False
    width: int = 1024
    height: int = 1024
    steps: int = 30
    seed: int = 42
    strength: float = 0.9999
    guidance_scale: float = 5.0
    adapter_scale: float = 1.0
    controlnet_scale: float = 0.9
    segmentation_threshold: float = 0.5
    prompt: str = "best quality, high quality"
    negative_prompt: str = "monochrome, lowres, bad anatomy, worst quality, low quality"

    def __post_init__(self):
        if self.method not in "ABCD" or len(self.method) != 1:
            raise ValueError("method must be A, B, C or D")
        if self.device != "cpu" and self.device != "cuda" and not self.device.startswith("cuda:"):
            raise ValueError("device must be cpu, cuda or cuda:N")
        if self.dtype not in {"float16", "float32"}:
            raise ValueError("dtype must be float16 or float32")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU inference requires dtype=float32")
        if min(self.width, self.height) < 64 or self.width % 8 or self.height % 8:
            raise ValueError("width and height must be multiples of 8 and >= 64")
        if self.steps < 1 or not 0 < self.strength <= 1 or int(self.steps * self.strength) < 1:
            raise ValueError("steps * strength must provide at least one denoising step")
        if not 0 < self.segmentation_threshold < 1:
            raise ValueError("segmentation_threshold must be between 0 and 1")
        if min(self.guidance_scale, self.adapter_scale, self.controlnet_scale) < 0:
            raise ValueError("conditioning scales must be nonnegative")

    @property
    def segment_reference(self):
        return self.method in {"B", "D"}

    @property
    def fuse_attention(self):
        return self.method in {"C", "D"}

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
