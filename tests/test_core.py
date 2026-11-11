import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from diffusers.models.attention_processor import Attention
from material_transfer.attention import material_attention_mask, heatmap_image
from material_transfer.config import TransferConfig
from material_transfer.images import decolorize, depth_image, load_mask
from material_transfer.records import read_manifest, save_run


def test_methods_and_invalid_settings():
    assert [(TransferConfig(method=m).segment_reference, TransferConfig(method=m).fuse_attention)
            for m in 'ABCD'] == [(False, False), (True, False), (False, True), (True, True)]
    for settings in ({'method': ''}, {'width': 65}, {'device': 'cpu'}, {'steps': 1}, {'segmentation_threshold': 1}):
        with pytest.raises(ValueError):
            TransferConfig(**settings)


def test_foreground_and_depth(tmp_path):
    image = Image.new('RGB', (2, 1), (200, 10, 30))
    mask = Image.fromarray(np.array([[0, 255]], dtype='uint8'))
    result = decolorize(image, mask)
    assert result.getpixel((0, 0)) == image.getpixel((0, 0))
    assert len(set(result.getpixel((1, 0)))) == 1
    for dtype, values in [('uint8', [[0, 255]]), ('uint16', [[0, 65535]])]:
        path = tmp_path / f'{dtype}.png'
        Image.fromarray(np.array(values, dtype=dtype)).save(path)
        assert depth_image(path, (2, 1)).getpixel((1, 0)) == (255, 255, 255)
    mask.save(tmp_path / 'mask.png')
    with pytest.raises(ValueError, match='differs'):
        load_mask(tmp_path / 'mask.png', (3, 1))


def test_attention_heads_and_subwords():
    torch.manual_seed(2)
    attn = Attention(query_dim=8, cross_attention_dim=12, heads=2, dim_head=4)
    hidden, text = torch.randn(2, 5, 8), torch.randn(2, 7, 12)
    valid = torch.tensor([[0, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    result = material_attention_mask(attn, hidden, text, valid)
    assert result.shape == (2, 5, 1)
    assert torch.isfinite(result).all()
    # Compare explicitly per head. Averaging projected Q/K before softmax is wrong.
    q = attn.to_q(hidden).reshape(2, 5, 2, 4).transpose(1, 2)
    k = attn.to_k(text).reshape(2, 7, 2, 4).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2) / 2).softmax(-1).mean(1)
    expected = torch.sigmoid(((scores * valid[:, None, :]).sum(-1) / valid.sum(-1)[:, None]) * 5)
    torch.testing.assert_close(result[..., 0], expected)
    assert heatmap_image(result, (20, 30)).size == (20, 30)


def test_manifest_and_record(tmp_path):
    for name in ('target', 'exemplar', 'mask'):
        Image.new('RGB', (8, 8), 'white').save(tmp_path / f'{name}.png')
    manifest = tmp_path / 'pairs.csv'
    manifest.write_text('pair_id,target,exemplar,mask,depth,material_prompt,seed\np01,target.png,exemplar.png,mask.png,,woven cloth,17\n')
    pair = read_manifest(manifest)[0]
    assert pair.seed == 17 and pair.target.is_absolute()
    path = save_run(tmp_path / 'out', {'result': Image.new('RGB', (8, 8))}, TransferConfig(), {}, {})
    record = json.loads((path.parent / 'run.json').read_text())
    assert record['schema_version'] == 1 and record['package_source_sha256']
    with pytest.raises(FileExistsError):
        save_run(path.parent, {}, TransferConfig(), {}, {})
    manifest.write_text(manifest.read_text().replace('p01', '../bad'))
    with pytest.raises(ValueError, match='Unsafe'):
        read_manifest(manifest)
