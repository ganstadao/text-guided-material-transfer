"""Prepare versioned, fixed course inputs and independent draft reference masks.

Annotations remain assistant drafts until independently reviewed. No CLIPSeg
prediction is used to create reference GT. Existing frozen inputs are immutable.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from material_transfer.images import load_rgb


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def region_mask(image, annotation):
    if annotation.get('full_frame'):
        return Image.new('L', image.size, 255)
    if annotation.get('direct_polygons'):
        result = Image.new('L', image.size)
        for polygon in annotation['direct_polygons']:
            ImageDraw.Draw(result).polygon([tuple(p) for p in polygon], fill=255)
        for polygon in annotation.get('holes', []):
            ImageDraw.Draw(result).polygon([tuple(p) for p in polygon], fill=0)
        return result
    coarse = Image.new('L', image.size)
    draw = ImageDraw.Draw(coarse)
    draw.polygon([tuple(p) for p in annotation['polygon']], fill=255)
    holes = Image.new('L', image.size)
    for polygon in annotation.get('holes', []):
        ImageDraw.Draw(holes).polygon([tuple(p) for p in polygon], fill=255)
    inside = np.asarray(coarse) > 0
    excluded = np.asarray(holes) > 0
    cv2.setRNGSeed(0)
    cv2.setNumThreads(1)
    labels = np.full(inside.shape, cv2.GC_BGD, dtype='uint8')
    labels[cv2.dilate(inside.astype('uint8'), np.ones((7, 7), 'uint8')) > 0] = cv2.GC_PR_BGD
    labels[inside] = cv2.GC_PR_FGD
    for x, y in annotation.get('foreground', []):
        cv2.circle(labels, (x, y), 3, cv2.GC_FGD, -1)
    for x, y in annotation.get('background', []):
        cv2.circle(labels, (x, y), 6, cv2.GC_BGD, -1)
    labels[excluded] = cv2.GC_BGD
    cv2.grabCut(np.asarray(image).copy(), labels, None,
                np.zeros((1, 65), 'float64'), np.zeros((1, 65), 'float64'), 5, cv2.GC_INIT_WITH_MASK)
    mask = ((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD)) & ~excluded
    if not mask.any():
        raise ValueError('Annotation produced an empty mask')
    return Image.fromarray(mask.astype('uint8') * 255)


def gallery(items, destination):
    canvas = Image.new('RGB', (768, 286 * len(items)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (name, image, mask) in enumerate(items):
        overlay = Image.composite(Image.blend(image, Image.new('RGB', image.size, (20, 230, 60)), 0.5), image, mask)
        for col, (label, item) in enumerate(zip(('source', 'mask', 'overlay'), (image, mask.convert('RGB'), overlay))):
            draw.text((col * 256 + 8, row * 286 + 5), f'{name}: {label}', fill='black')
            canvas.paste(ImageOps.contain(item, (256, 256)), (col * 256, row * 286 + 26))
    canvas.save(destination)


def prepare(manifest, annotations, output, depth_model, masks_only=False):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if (output / 'frozen_inputs.json').exists():
        raise FileExistsError('Inputs already frozen; use a new versioned output directory')
    annotation = json.loads(Path(annotations).read_text(encoding='utf-8'))
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for name in ('masks', 'depths', 'gt_by_exemplar', 'gt_by_pair', 'previews'):
        (output / name).mkdir(parents=True, exist_ok=True)
    target_items, reference_items, inputs = [], [], {}
    for stem, info in annotation['targets'].items():
        path = manifest.parent / 'targets' / f'{stem}.png'
        image = load_rgb(path)
        mask = region_mask(image, info)
        mask.save(output / 'masks' / f'{stem}.png')
        inputs[str(path)] = sha(path)
        target_items.append((stem, image, mask))
    for stem, info in annotation['references'].items():
        path = manifest.parent / 'exemplars' / f'{stem}.png'
        image = load_rgb(path)
        mask = region_mask(image, info)
        mask.save(output / 'gt_by_exemplar' / f'{stem}.png')
        inputs[str(path)] = sha(path)
        reference_items.append((stem, image, mask))
    gallery(target_items, output / 'previews/target_masks.png')
    gallery(reference_items, output / 'previews/reference_masks.png')
    if masks_only:
        print(f'Draft mask previews: {output / "previews"}')
        return
    from material_transfer.config import TransferConfig
    from material_transfer.pipeline import MaterialTransfer
    engine = MaterialTransfer(TransferConfig(device='cpu', dtype='float32',
        depth_model=str(Path(depth_model).resolve()), local_files_only=True))
    for stem, image, mask in target_items:
        destination = output / 'depths' / f'{stem}.png'
        print(f'Computing fixed depth: {stem}', flush=True)
        engine.depth(image).save(destination)
    fields = ['pair_id', 'target', 'exemplar', 'mask', 'depth', 'material_prompt', 'seed']
    frozen_rows, groups, original_seeds = [], [], {}
    for row in rows:
        target, exemplar = Path(row['target']).stem, Path(row['exemplar']).stem
        original_seeds[row['pair_id']] = row.get('seed')
        shutil.copy2(output / 'gt_by_exemplar' / f'{exemplar}.png', output / 'gt_by_pair' / f"{row['pair_id']}.png")
        frozen_rows.append(dict(pair_id=row['pair_id'],
            target=os.path.relpath(manifest.parent / row['target'], output),
            exemplar=os.path.relpath(manifest.parent / row['exemplar'], output),
            mask=f'masks/{target}.png', depth=f'depths/{target}.png', material_prompt=row['material_prompt'], seed=42))
        groups.append(dict(pair_id=row['pair_id'], target_id=target, reference_id=exemplar,
            target_object=annotation['targets'][target]['object'],
            target_context='simple' if target in ('t01', 't02', 't03') else 'cluttered',
            reference_context=annotation['references'][exemplar]['group']))
    with (output / 'manifest.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frozen_rows)
    metadata = dict(status=annotation['status'], annotation_source=str(Path(annotations).resolve()),
        independent_of_clipseg=True, note='Draft annotations, not independently reviewed human GT')
    (output / 'gt_by_pair/annotations.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (output / 'strata.json').write_text(json.dumps(groups, indent=2), encoding='utf-8')
    files = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*.png')) if 'previews' not in p.parts}
    frozen = dict(source_manifest=str(manifest), source_inputs_sha256=inputs,
        annotations_sha256=sha(annotations), artifacts_sha256=files, seed=42, original_seeds=original_seeds,
        target_annotation_status=annotation['status'], reference_annotation_status=annotation['status'],
        depth_model=str(Path(depth_model).resolve()))
    (output / 'frozen_inputs.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    print(f'Frozen {len(frozen_rows)} pairs: {output / "manifest.csv"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/manifest.csv')
    parser.add_argument('--annotations', default='experiments/annotations/course_v1.json')
    parser.add_argument('--output', default='data/course_fixed_v1')
    parser.add_argument('--depth-model', default='models/local/dpt')
    parser.add_argument('--masks-only', action='store_true')
    args = parser.parse_args()
    prepare(args.manifest, args.annotations, args.output, args.depth_model, args.masks_only)
