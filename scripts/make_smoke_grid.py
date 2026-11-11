"""Render the shared inputs and four results without evaluation claims."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path, help='outputs/smoke-JOBID')
args = parser.parse_args()
items = [('Target', args.root / 'A/pumpkin_glaze/target.png'),
         ('Exemplar', args.root / 'A/pumpkin_glaze/exemplar.png')]
items += [(m, args.root / f'{m}/pumpkin_glaze/result.png') for m in 'ABCD']
canvas = Image.new('RGB', (3 * 384, 2 * 412), 'white')
draw = ImageDraw.Draw(canvas)
for i, (label, path) in enumerate(items):
    with Image.open(path) as im:
        im = im.convert('RGB').resize((384, 384))
    x, y = (i % 3) * 384, (i // 3) * 412
    canvas.paste(im, (x, y + 28))
    draw.text((x + 12, y + 7), label, fill='black')
canvas.save(args.root / 'comparison.png')
print(args.root / 'comparison.png')
