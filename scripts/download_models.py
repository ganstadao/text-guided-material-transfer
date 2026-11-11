"""Fetch only weights needed by the selected config into the shared HF cache."""
import argparse
from huggingface_hub import snapshot_download
from material_transfer.config import TransferConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/D.json")
    parser.add_argument("--with-depth", action="store_true")
    args = parser.parse_args()
    cfg = TransferConfig.load(args.config)
    for repo in (cfg.base_model, cfg.controlnet):
        snapshot_download(repo_id=repo, allow_patterns=["*.json", "*.txt", "*.model",
            f"*{cfg.model_variant}*.safetensors" if cfg.model_variant else "*.safetensors"])
    snapshot_download(cfg.adapter_repo, allow_patterns=[
        f"{cfg.image_encoder_subfolder}/*", f"{cfg.adapter_subfolder}/{cfg.adapter_weight}"])
    if cfg.segment_reference:
        snapshot_download(cfg.clipseg_model)
    if args.with_depth:
        snapshot_download(cfg.depth_model)


if __name__ == "__main__":
    main()
