"""Prepare official ZeST assets and an approximate manual smoke-test mask.

The polygon is for functional verification only; it is not benchmark ground truth.
"""
from pathlib import Path
import urllib.request
from PIL import Image, ImageDraw


def main():
    root = Path('data/smoke')
    root.mkdir(parents=True, exist_ok=True)
    for source, name in [('input_imgs/pumpkin.png', 'target.png'),
                         ('material_exemplars/cup_glaze.png', 'exemplar.png'),
                         ('depths/pumpkin.png', 'depth.png')]:
        destination = root / name
        if not destination.exists():
            urllib.request.urlretrieve('https://raw.githubusercontent.com/ttchengab/zest_code/main/demo_assets/' + source, destination)
    with Image.open(root / 'target.png') as target:
        mask = Image.new('L', target.size)
        # Manually approximated pumpkin body; exclude stem for this diagnostic.
        points = [(150, 555), (180, 478), (250, 413), (360, 385), (450, 380),
                  (490, 445), (550, 465), (590, 385), (720, 380), (835, 443),
                  (890, 540), (903, 665), (875, 775), (800, 855), (675, 912),
                  (550, 940), (420, 915), (300, 880), (220, 815), (175, 695)]
        points = [(round(x * target.width / 1024), round(y * target.height / 1024)) for x, y in points]
        ImageDraw.Draw(mask).polygon(points, fill=255)
        mask.save(root / 'target_mask.png')
    (root / 'manifest.csv').write_text(
        'pair_id,target,exemplar,mask,depth,material_prompt,seed\n'
        'pumpkin_glaze,target.png,exemplar.png,target_mask.png,depth.png,glazed ceramic,42\n')
    (root / 'SOURCE.txt').write_text(
        'Images and depth from https://github.com/ttchengab/zest_code/tree/main/demo_assets\n'
        'Target mask: approximate manual polygon, functional smoke test only.\n')


if __name__ == '__main__':
    main()
