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
import importlib.metadata
from functools import lru_cache
import re
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
    mask = np.asarray(mask)
    if mask.dtype == bool:
        return mask
    return mask >= (0.5 if mask.size and mask.max() <= 1 else threshold)


def resize_to(mask, shape):
    """Resize a 0/255 float mask to (H, W) and binarize at the 0..255 midpoint."""
    image = Image.fromarray(binarize(mask).astype(np.uint8) * 255)
    resized = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return binarize(np.asarray(resized, dtype=np.float32))


def aligned_inputs(result_path, target_path, mask_path):
    result = load_rgb_array(result_path)
    size = (result.shape[1], result.shape[0])
    with Image.open(target_path) as image:
        target = np.asarray(image.convert('RGB').resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255
    mask = resize_to(load_gray(mask_path), result.shape[:2])
    return result, target, mask


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
    from scipy.ndimage import gaussian_filter, binary_erosion
    # Exclude windows touching the edited foreground, rather than measuring
    # foreground changes through neighboring background SSIM windows.
    radius = int(4 * sigma + 0.5)
    keep = binary_erosion(keep, structure=np.ones((2 * radius + 1, 2 * radius + 1), bool), border_value=1)
    luminance = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    la, lb = result @ luminance, target @ luminance
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


def lpips_score(result, target, keep):
    import lpips
    import torch
    model = getattr(lpips_score, "_model", None)
    if model is None:
        model = lpips.LPIPS(net="alex", spatial=True).eval()
        lpips_score._model = model
    to_tensor = lambda array: torch.from_numpy((array * 2 - 1)).permute(2, 0, 1)[None].float()
    if not keep.any():
        return float('nan')
    # Identical neutral foreground prevents intended material changes from
    # contributing to this explicitly masked-background protocol.
    a, b = result.copy(), target.copy()
    a[~keep] = b[~keep] = 0.5
    with torch.inference_mode():
        distance = model(to_tensor(a), to_tensor(b))[0, 0].cpu().numpy()
    return float(distance[keep].mean())


@lru_cache(maxsize=2)
def clip_components(model_name):
    from transformers import CLIPModel, CLIPProcessor
    return CLIPModel.from_pretrained(model_name).eval(), CLIPProcessor.from_pretrained(model_name)


def clip_similarity(image_a, image_b, model_name):
    import torch
    model, processor = clip_components(model_name)
    to_pil = lambda array: Image.fromarray((array * 255).astype(np.uint8))
    inputs = processor(images=[to_pil(image_a), to_pil(image_b)], return_tensors="pt")
    with torch.inference_mode():
        embeddings = model.get_image_features(**inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return float((embeddings[0] @ embeddings[1]).item())


def object_crop(image, mask):
    if image.shape[:2] != mask.shape:
        raise ValueError('Object mask and image sizes must agree')
    rows, cols = np.where(binarize(mask))
    if rows.size == 0:
        return image
    return image[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def read_manifest(path):
    with open(path, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen = set()
    for row in rows:
        pair_id = row.get('pair_id', '')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', pair_id) or pair_id in seen:
            raise ValueError(f'Unsafe or duplicate pair_id: {pair_id}')
        seen.add(pair_id)
    if not rows:
        raise ValueError('Manifest has no pairs')
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--groups", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--gt-masks", type=Path, default=Path("data/masks_exemplar_gt"))
    parser.add_argument("--out", type=Path, default=Path("evaluation/results.csv"))
    parser.add_argument("--lpips", action="store_true", help="add background LPIPS (needs the lpips package)")
    parser.add_argument("--clip-model", default="", help="optional CLIP model id, e.g. openai/clip-vit-base-patch32")
    parser.add_argument('--allow-missing', action='store_true', help='Allow partial summaries; still report missing rows')
    parser.add_argument('--require-reviewed-gt', action='store_true', help='Require independently reviewed GT metadata')
    args = parser.parse_args(argv)

    if any(group not in 'ABCD' or len(group) != 1 for group in args.groups):
        parser.error('groups must be A, B, C or D')

    fields = ["pair_id", "group", "bg_mae", "bg_ssim"]
    if args.lpips:
        fields.append("bg_lpips")
    fields += ["ref_iou", "ref_dice", "clip_sim", 'bg_pixels', 'status', 'gt_status', 'error']
    rows, skipped, errors = [], 0, 0
    gt_metadata_path = args.gt_masks / 'annotations.json'
    gt_metadata = json.loads(gt_metadata_path.read_text(encoding='utf-8')) if gt_metadata_path.exists() else {}
    for row in read_manifest(args.manifest):
        pair_id = row["pair_id"]
        gt_path = args.gt_masks / f"{pair_id}.png"
        for group in args.groups:
            run_dir = args.outputs / group / pair_id
            if not (run_dir / "run.json").exists():
                print(f"skip missing run: {run_dir}", file=sys.stderr)
                skipped += 1
                rows.append(dict(pair_id=pair_id, group=group, status='missing_run', error='run.json missing'))
                continue
            try:
                record = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
                if record['config']['method'] != group:
                    raise ValueError('Method differs from output group')
                artifacts = record['artifacts']
                def art(name):
                    path = (run_dir / artifacts[name]).resolve()
                    if not path.is_relative_to(run_dir.resolve()):
                        raise ValueError('Artifact escapes run directory')
                    return path
                result, target, mask = aligned_inputs(art('result'), art('target'), art('target_mask'))
                background = ~mask
                values = dict(pair_id=pair_id, group=group, status='complete', error='',
                    bg_mae=background_mae(result, target, background),
                    bg_ssim=ssim_luminance(result, target, background), bg_pixels=int(background.sum()),
                    ref_iou=float('nan'), ref_dice=float('nan'), clip_sim=float('nan'), gt_status='not_applicable')
                if args.lpips:
                    values['bg_lpips'] = lpips_score(result, target, background)
                if 'reference_mask' in artifacts:
                    values['gt_status'] = gt_metadata.get('status', 'unverified') if gt_path.is_file() else 'missing'
                    if args.require_reviewed_gt and values['gt_status'] != 'reviewed':
                        raise ValueError('Reference GT is missing or not independently reviewed')
                    if gt_path.is_file():
                        gt = binarize(load_gray(gt_path))
                        pred = resize_to(load_gray(art('reference_mask')), gt.shape)
                        values['ref_iou'], values['ref_dice'] = iou_dice(pred, gt)
                if args.clip_model:
                    values['clip_sim'] = clip_similarity(object_crop(result, mask), load_rgb_array(art('exemplar')), args.clip_model)
                rows.append(values)
            except Exception as exc:
                errors += 1
                rows.append(dict(pair_id=pair_id, group=group, status='error', error=str(exc)))
                print(f'error {pair_id}/{group}: {exc}', file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    suffix = f" ({skipped} runs skipped)" if skipped else ""
    print(f"wrote {len(rows)} rows to {args.out}{suffix}")
    summary = dict(expected_runs=len(rows), completed_runs=sum(r['status'] == 'complete' for r in rows),
        missing_runs=skipped, invalid_runs=errors,
        protocols=dict(background_alignment='RGB Lanczos; binary mask nearest to result size',
            ssim='Rec709 luminance; exclude foreground-touching Gaussian windows',
            lpips='spatial AlexNet; neutralize foreground equally; average background map',
            reference_gt_status=gt_metadata.get('status', 'unverified'), clip_model=args.clip_model or None))
    summary['versions'] = {}
    for name in ('numpy', 'Pillow', 'scipy', 'torch', 'lpips', 'transformers'):
        try:
            summary['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            summary['versions'][name] = None
    summary['groups'] = {}
    for group in args.groups:
        group_rows = [r for r in rows if r['group'] == group and r['status'] == 'complete']
        summary['groups'][group] = {}
        for metric in ('bg_mae', 'bg_ssim', 'ref_iou', 'ref_dice', 'clip_sim', 'bg_lpips'):
            numbers = [float(r[metric]) for r in group_rows if metric in r and np.isfinite(r[metric])]
            summary['groups'][group][metric] = dict(n=len(numbers), mean=float(np.mean(numbers)) if numbers else None)
    args.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return int(bool(errors or (skipped and not args.allow_missing)))


if __name__ == "__main__":
    raise SystemExit(main())
