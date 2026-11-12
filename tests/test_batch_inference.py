from types import SimpleNamespace
import torch
from PIL import Image
import pytest
from material_transfer.config import TransferConfig
from material_transfer.pipeline import MaterialTransfer


@pytest.mark.parametrize('method', list('ABCD'))
@pytest.mark.parametrize('guidance', [1.0, 5.0])
def test_distinct_references_and_generators_survive_batching(method, guidance, monkeypatch):
    class Pipe:
        _execution_device = 'cpu'
        def prepare_ip_adapter_image_embeds(self, *args):
            raise AssertionError('Stock repeat helper must not run for pre-batched embeddings')
        def encode_image(self, image, *args):
            value = image.getpixel((0, 0))[0]
            return torch.full((1, 16), value), torch.zeros(1, 16)
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            self.prepared = self.prepare_ip_adapter_image_embeds(None, kwargs['ip_adapter_image_embeds'],
                'cpu', len(kwargs['prompt']), guidance > 1)
            return SimpleNamespace(images=kwargs['image'])
    pipe = Pipe()
    original = pipe.prepare_ip_adapter_image_embeds
    engine = MaterialTransfer(TransferConfig(method=method, device='cpu', dtype='float32',
        batch_size=4, width=64, height=64, guidance_scale=guidance))
    engine.pipe = pipe
    engine.reference = lambda image, phrase: (image, Image.new('L', image.size, 255) if method in 'BD' else None)
    def fuse(pipe, image, phrase, block):
        return torch.full((1, 16), image.getpixel((0, 0))[0]), Image.new('L', image.size, 128)
    monkeypatch.setattr('material_transfer.attention.fuse_reference', fuse)
    samples = [dict(target=Image.new('RGB', (32, 32), 'red'),
        exemplar=Image.new('RGB', (32, 32), (i+1, 0, 0)), material_prompt='material',
        target_mask=Image.new('L', (32, 32), 255), depth=Image.new('RGB', (32, 32)), seed=i+11)
        for i in range(4)]
    results = engine.run_many(samples)
    assert len(results) == 4
    assert [g.initial_seed() for g in pipe.kwargs['generator']] == [11, 12, 13, 14]
    embeddings = pipe.prepared[0]
    assert embeddings.shape == (8 if guidance > 1 else 4, 1, 16)
    assert embeddings[-4:, 0, 0].tolist() == [1, 2, 3, 4]
    if guidance > 1:
        assert not embeddings[:4].any()
    assert pipe.prepare_ip_adapter_image_embeds == original
    for i, (artifacts, meta) in enumerate(results):
        assert artifacts['result'].size == (64, 64)
        assert ('reference_mask' in artifacts) == (method in 'BD')
        assert ('attention_mask' in artifacts) == (method in 'CD')
        assert meta['actual_batch_size'] == 4 and meta['batch_index'] == i
        assert meta['generation_seconds'] * 4 == pytest.approx(meta['batch_generation_seconds'])
