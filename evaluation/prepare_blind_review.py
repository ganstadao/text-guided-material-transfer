"""Create a rater package and a separate private method mapping; no ratings invented."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import uuid

from PIL import Image, ImageDraw, ImageOps


def prepare(manifest, outputs, destination, seed=20261112):
    outputs, destination = Path(outputs).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Use a new blind-review destination; never overwrite ratings')
    with Path(manifest).open(encoding='utf-8-sig', newline='') as stream:
        pairs = list(csv.DictReader(stream))
    records = {}
    for pair in pairs:
        pair_id = pair['pair_id']
        for method in 'ABCD':
            directory = outputs / method / pair_id
            record = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
            if record['config']['method'] != method:
                raise ValueError('Mismatched run method')
            paths = {name: (directory / record['artifacts'][name]).resolve()
                     for name in ('result', 'target', 'target_mask', 'exemplar')}
            if any(not path.is_relative_to(directory) or not path.is_file() for path in paths.values()):
                raise ValueError('Missing or unsafe artifact')
            records[pair_id, method] = paths
    public = destination / 'rater_package'
    private = destination / 'private'
    public.mkdir(parents=True)
    private.mkdir()
    rng = random.Random(seed)
    rating_rows, mapping = [], []
    for pair in pairs:
        pair_id = pair['pair_id']
        directory = public / pair_id
        directory.mkdir()
        order = list('ABCD')
        rng.shuffle(order)
        source = records[pair_id, 'A']
        panels = []
        for name, label in (('target', 'Target'), ('exemplar', 'Reference'), ('target_mask', 'Edit region')):
            with Image.open(source[name]) as image:
                clean = image.convert('RGB')
                clean.save(directory / f'{name}.png')
                panels.append((label, clean.copy()))
        canvas = Image.new('RGB', (1024, 572), 'white')
        draw = ImageDraw.Draw(canvas)
        for index, (label, image) in enumerate(panels):
            draw.text((index * 256 + 8, 6), label, fill='black')
            canvas.paste(ImageOps.contain(image, (256, 256)), (index * 256, 26))
        for index, method in enumerate(order):
            token = uuid.uuid4().hex[:12]
            item_id = f'{pair_id}_{token}'
            with Image.open(records[pair_id, method]['result']) as image:
                clean = image.convert('RGB')
                clean.save(directory / f'{item_id}.png')
                canvas.paste(ImageOps.contain(clean, (256, 256)), (index * 256, 312))
            draw.text((index * 256 + 8, 292), item_id, fill='black')
            rating_rows.append(dict(rater='', pair_id=pair_id, item_id=item_id,
                material_match='', shape='', naturalness='', notes=''))
            mapping.append(dict(pair_id=pair_id, item_id=item_id, group=method,
                result_sha256=hashlib.sha256(records[pair_id, method]['result'].read_bytes()).hexdigest()))
        canvas.save(directory / 'comparison.png')
    with (public / 'ratings.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rating_rows[0]))
        writer.writeheader()
        writer.writerows(rating_rows)
    with (private / 'mapping.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    (public / 'README.md').write_text(
        '# 匿名评分说明\n\n只共享 rater_package 目录，不向评分者提供 private/mapping.csv。\n\n'
        '每个目录包含目标图、材质参考图、编辑区域和四张匿名结果。请结合白色编辑区域评分。\n\n'
        '在 ratings.csv 填自己的匿名编号，并对每张结果按 1–5 分填写三个维度：\n'
        '- material_match：编辑区域与参考材质的符合程度。\n'
        '- shape：目标形状、结构与编辑区域外内容的保留程度。\n'
        '- naturalness：光照、纹理、边缘衔接是否自然。\n\n'
        '1=明显不符合；3=部分符合且有明显问题；5=符合且无明显问题。允许填写 notes。\n'
        '独立评分，不讨论或猜测方法。无法判断的条目留空并说明原因。评分表不含预填分数。\n',
        encoding='utf-8')
    print(f'Prepared {len(rating_rows)} anonymous items: {public}')
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--outputs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.manifest, args.outputs, args.output)
