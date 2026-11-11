import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
from pathlib import Path
import sys
from .config import TransferConfig


def doctor():
    import torch
    print(json.dumps(dict(python=sys.executable, torch=torch.__version__,
                          cuda_available=torch.cuda.is_available(),
                          cuda_runtime=torch.version.cuda,
                          gpu_count=torch.cuda.device_count(),
                          diffusers=importlib.metadata.version("diffusers")), indent=2))
    return 0 if torch.cuda.is_available() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="ZeST-style material transfer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report active Python and GPU availability")
    for command in ("run", "batch"):
        run = sub.add_parser(command)
        run.add_argument("--config", default="configs/A.json")
        run.add_argument("--output", required=True, type=Path)
        if command == "run":
            run.add_argument("--target", required=True, type=Path)
            run.add_argument("--exemplar", required=True, type=Path)
            run.add_argument("--mask", type=Path)
            run.add_argument("--depth", type=Path)
            run.add_argument("--material-prompt", default="")
        else:
            run.add_argument("--manifest", required=True, type=Path)
            run.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    cfg = TransferConfig.load(args.config)
    from .records import Pair, read_manifest, save_run
    from .images import load_rgb, load_mask, depth_image
    from .pipeline import MaterialTransfer
    pairs = read_manifest(args.manifest) if args.command == "batch" else [
        Pair("single", args.target, args.exemplar, args.mask, args.depth, args.material_prompt)]
    # Validate every input before any downloads/GPU allocation, even in run mode.
    for pair in pairs:
        target = load_rgb(pair.target)
        load_rgb(pair.exemplar)
        if pair.mask:
            load_mask(pair.mask, target.size)
        if pair.depth:
            depth_image(pair.depth, target.size)
        if cfg.method != "A" and not pair.material_prompt.strip():
            raise ValueError(f"{pair.pair_id}: B/C/D require material_prompt")
        directory = args.output / pair.pair_id if args.command == "batch" else args.output
        if (directory / "run.json").exists():
            raise FileExistsError(f"Run already exists: {directory}")
    if getattr(args, "validate_only", False):
        print(f"Validated {len(pairs)} pairs for method {cfg.method}; no models loaded")
        return 0
    engine = MaterialTransfer(cfg)
    for pair in pairs:
        engine.config = replace(cfg, seed=pair.seed if pair.seed is not None else cfg.seed)
        target = load_rgb(pair.target)
        mask = load_mask(pair.mask, target.size) if pair.mask else None
        depth = depth_image(pair.depth, target.size) if pair.depth else None
        artifacts, timings = engine.run(target, load_rgb(pair.exemplar), pair.material_prompt, mask, depth)
        directory = args.output / pair.pair_id if args.command == "batch" else args.output
        record = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in asdict(pair).items()}
        print(save_run(directory, artifacts, engine.config, timings, record))
    return 0
