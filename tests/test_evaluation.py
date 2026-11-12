import json
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest
import torch
from evaluation import compute_metrics as metrics


def test_binary_masks_exclude_foreground_and_crop_object():
    raw = np.zeros((32, 32), dtype=np.uint8)
    raw[8:24, 8:24] = 255
    mask = metrics.binarize(raw)
    target = np.zeros((32, 32, 3), dtype=np.float32)
    result = target.copy()
    result[mask] = 1
    assert metrics.background_mae(result, target, ~mask) == 0
    assert metrics.ssim_luminance(result, target, ~mask) == pytest.approx(1)
    assert metrics.object_crop(result, mask).shape == (16, 16, 3)
    assert metrics.object_crop(result, raw).shape == (16, 16, 3)
    assert np.array_equal(metrics.binarize(mask.astype('uint8')), mask)


def test_native_target_is_aligned_to_generated_resolution(tmp_path):
    Image.new('RGB', (8, 8), 'black').save(tmp_path / 'target.png')
    result = np.zeros((16, 16, 3), dtype='uint8')
    result[4:12, 4:12] = 255
    Image.fromarray(result).save(tmp_path / 'result.png')
    mask = np.zeros((8, 8), dtype='uint8')
    mask[2:6, 2:6] = 255
    Image.fromarray(mask).save(tmp_path / 'mask.png')
    result, target, mask = metrics.aligned_inputs(tmp_path / 'result.png', tmp_path / 'target.png', tmp_path / 'mask.png')
    assert result.shape == target.shape == (16, 16, 3)
    assert mask.shape == (16, 16) and mask.sum() == 64
    assert metrics.background_mae(result, target, ~mask) == 0


def test_empty_background_has_no_fabricated_score():
    image = np.zeros((8, 8, 3), dtype=np.float32)
    keep = np.zeros((8, 8), bool)
    assert np.isnan(metrics.background_mae(image, image, keep))
    assert np.isnan(metrics.ssim_luminance(image, image, keep))


def test_lpips_neutralizes_foreground_and_reduces_background_only(monkeypatch):
    class FakeLPIPS:
        def eval(self):
            return self
        def __call__(self, a, b):
            return (a - b).abs().mean(1, keepdim=True)
    monkeypatch.setitem(sys.modules, 'lpips', SimpleNamespace(LPIPS=lambda **kw: FakeLPIPS()))
    monkeypatch.delattr(metrics.lpips_score, '_model', raising=False)
    a = np.zeros((16, 16, 3), dtype=np.float32)
    b = a.copy()
    b[4:12, 4:12] = 1
    keep = np.ones((16, 16), bool)
    keep[4:12, 4:12] = False
    try:
        assert metrics.lpips_score(a, b, keep) == 0
        b[:2] = 0.2
        assert metrics.lpips_score(a, b, keep) > 0
    finally:
        del metrics.lpips_score._model


def test_metrics_cli_reports_complete_and_missing_runs(tmp_path):
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('pair_id\np01\n')
    directory = tmp_path / 'runs/A/p01'
    directory.mkdir(parents=True)
    for name in ('result', 'target'):
        Image.new('RGB', (32, 32), 'black').save(directory / f'{name}.png')
    mask = np.zeros((32, 32), dtype='uint8')
    mask[8:24, 8:24] = 255
    Image.fromarray(mask).save(directory / 'target_mask.png')
    (directory / 'run.json').write_text(json.dumps(dict(config={'method': 'A'},
        artifacts={name: f'{name}.png' for name in ('result', 'target', 'target_mask')})))
    out = tmp_path / 'metrics.csv'
    common = ['--manifest', str(manifest), '--outputs', str(tmp_path / 'runs'), '--out', str(out)]
    assert metrics.main(common + ['--groups', 'A']) == 0
    summary = json.loads(out.with_suffix('.summary.json').read_text())
    assert summary['groups']['A']['bg_mae']['mean'] == 0
    assert metrics.main(common + ['--groups', 'A', 'B']) == 1
    summary = json.loads(out.with_suffix('.summary.json').read_text())
    assert summary['completed_runs'] == 1 and summary['missing_runs'] == 1
