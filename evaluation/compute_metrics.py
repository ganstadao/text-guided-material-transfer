#!/usr/bin/env python3
"""Per-pair, per-group evaluation metrics for the material-transfer project.

Reads completed run directories (run.json + saved artifacts) and writes one
CSV row per pair x group with:

  - reference-mask IoU / Dice vs manual GT (groups B and D)
  - background preservation outside the target mask: MAE, SSIM, LPIPS (optional)
  - CLIP image similarity between the edited object crop and the exemplar (optional)

Core dependencies are numpy / Pillow / scipy only. LPIPS needs the `lpips`
package; CLIP similarity uses `transformers` and downloads weights unless cached.

Example:
    python evaluation/compute_metrics.py \
        --manifest data/manifest.csv --outputs outputs \
        --groups A B C D --gt-masks data/masks_exemplar_gt \
        --out evaluation/results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_gray(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def load_rgb_array(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def binarize(mask, threshold=127.0):
    return (mask >= threshold).astype(np.uint8)


def resize_to(mask, shape):
    """Resize a 0/255 float mask to (H, W) and binarize at the 0..255 midpoint."""
    image = Image.fromarray(np.clip(mask, 0, 255).astype(np.uint8))
    resized = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return binarize(np.asarray(resized, dtype=np.float32))


def iou_dice(pred, gt):
    intersection = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    total = float(pred.sum()) + float(gt.sum())
    return (intersection / union if union else float("nan"),
            2 * intersection / total if total else float("nan"))


def background_mae(result, target, keep):
    values = np.abs(result - target).mean(axis=-1)[keep]
    return float(values.mean()) if values.size else float("nan")


def ssim_luminance(result, target, keep, sigma=1.5):
    from scipy.ndimage import gaussian_filter
    la, lb = result.mean(axis=-1), target.mean(axis=-1)
    mu_a = gaussian_filter(la, sigma)
    mu_b = gaussian_filter(lb, sigma)
    var_a = gaussian_filter(la * la, sigma) - mu_a ** 2
    var_b = gaussian_filter(lb * lb, sigma) - mu_b ** 2
    cov = gaussian_filter(la * lb, sigma) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    values = ssim_map[keep]
    return float(values.mean()) if values.size else float("nan")


def lpips_score(result, target):
    import lpips
    import torch
    model = getattr(lpips_score, "_model", None)
    if model is None:
        model = lpips.LPIPS(net="alex")
        lpips_score._model = model
    to_tensor = lambda array: torch.from_numpy((array * 2 - 1)).permute(2, 0, 1)[None].float()
    return float(model(to_tensor(result), to_tensor(target)).item())


def clip_similarity(image_a, image_b, model_name):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    to_pil = lambda array: Image.fromarray((array * 255).astype(np.uint8))
    inputs = processor(images=[to_pil(image_a), to_pil(image_b)], return_tensors="pt")
    embeddings = model.get_image_features(**inputs)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return float((embeddings[0] @ embeddings[1]).item())


def object_crop(image, mask):
    rows, cols = np.where(mask > 127)
    if rows.size == 0:
        return image
    return image[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def read_manifest(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--groups", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--gt-masks", type=Path, default=Path("data/masks_exemplar_gt"))
    parser.add_argument("--out", type=Path, default=Path("evaluation/results.csv"))
    parser.add_argument("--lpips", action="store_true", help="add background LPIPS (needs the lpips package)")
    parser.add_argument("--clip-model", default="", help="optional CLIP model id, e.g. openai/clip-vit-base-patch32")
    args = parser.parse_args(argv)

    fields = ["pair_id", "group", "bg_mae", "bg_ssim"]
    if args.lpips:
        fields.append("bg_lpips")
    fields += ["ref_iou", "ref_dice", "clip_sim"]
    rows, skipped = [], 0
    for row in read_manifest(args.manifest):
        pair_id = row["pair_id"]
        gt_path = args.gt_masks / f"{pair_id}.png"
        for group in args.groups:
            run_dir = args.outputs / group / pair_id
            if not (run_dir / "run.json").exists():
                print(f"skip missing run: {run_dir}", file=sys.stderr)
                skipped += 1
                continue
            record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            artifacts = record.get("artifacts", {})

            def art(name):
                return run_dir / artifacts[name]

            result = load_rgb_array(art("result"))
            target = load_rgb_array(art("target"))
            mask = binarize(load_gray(art("target_mask")))
            background = mask < 128
            values = [pair_id, group, background_mae(result, target, background),
                      ssim_luminance(result, target, background)]
            if args.lpips:
                values.append(lpips_score(result, target))
            ref_iou = ref_dice = float("nan")
            if "reference_mask" in artifacts and gt_path.is_file():
                pred = resize_to(load_gray(art("reference_mask")), load_gray(gt_path).shape)
                ref_iou, ref_dice = iou_dice(pred, binarize(load_gray(gt_path)))
            values += [ref_iou, ref_dice]
            clip_value = float("nan")
            if args.clip_model and "exemplar" in artifacts:
                try:
                    clip_value = clip_similarity(object_crop(result, mask),
                                                 load_rgb_array(art("exemplar")), args.clip_model)
                except Exception as exc:
                    print(f"clip failed for {pair_id}/{group}: {exc}", file=sys.stderr)
            values.append(clip_value)
            rows.append(dict(zip(fields, values)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    suffix = f" ({skipped} runs skipped)" if skipped else ""
    print(f"wrote {len(rows)} rows to {args.out}{suffix}")


if __name__ == "__main__":
    main()
