"""Exercise real preparation/serialization with a recording diffusion backend.
No weights downloaded: these tests do not establish real-model output quality.
"""
from dataclasses import replace
from types import SimpleNamespace
import torch
from PIL import Image
import pytest
from material_transfer.config import TransferConfig
from material_transfer.pipeline import MaterialTransfer


class RecordingPipe:
    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(images=[kwargs['image']])


@pytest.mark.parametrize('method', list('ABCD'))
def test_four_paths_and_shared_inputs(method, monkeypatch):
    cfg = TransferConfig(method=method, device='cpu', dtype='float32', width=64, height=64)
    engine = MaterialTransfer(cfg)
    engine.pipe = RecordingPipe()
    target, exemplar = Image.new('RGB', (32, 32), 'red'), Image.new('RGB', (32, 32), 'blue')
    target_mask, ref_mask = Image.new('L', target.size, 255), Image.new('L', exemplar.size, 255)
    segment_calls, attention_calls = [], []
    def reference(image, phrase):
        segment_calls.append(cfg.segment_reference)
        return image, ref_mask if cfg.segment_reference else None
    engine.reference = reference
    def fuse(*args):
        attention_calls.append(True)
        return torch.ones(1, 16), Image.new('L', exemplar.size)
    monkeypatch.setattr('material_transfer.attention.fuse_reference', fuse)
    artifacts, timings = engine.run(target, exemplar, 'metal', target_mask, target)
    assert timings['segmentation_applied'] == cfg.segment_reference
    assert timings['attention_applied'] == cfg.fuse_attention
    assert ('reference_mask' in artifacts) == cfg.segment_reference
    assert bool(attention_calls) == cfg.fuse_attention
    args = engine.pipe.kwargs
    assert args['generator'].initial_seed() == 42
    assert args['image'].getpixel((0, 0))[0] == args['image'].getpixel((0, 0))[1]
    if cfg.fuse_attention:
        assert args['ip_adapter_image_embeds'][0].shape == (2, 1, 16)
        assert not args['ip_adapter_image_embeds'][0][0].any()
    else:
        assert args['ip_adapter_image'].tobytes() == exemplar.tobytes()


def test_clipseg_non_native_resolution():
    # Real tiny CLIPSeg: processor-sized pixels differ from pretrained grid.
    # This catches the 352-vs-224 failure without downloading a checkpoint.
    from transformers import CLIPSegConfig, CLIPSegForImageSegmentation
    config = CLIPSegConfig(
        text_config=dict(vocab_size=20, hidden_size=8, intermediate_size=16,
                         num_hidden_layers=2, num_attention_heads=2, eos_token_id=2),
        vision_config=dict(hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                           num_attention_heads=2, image_size=16, patch_size=8),
        projection_dim=8, extract_layers=[0, 1], reduce_dim=4,
        decoder_num_attention_heads=2, decoder_intermediate_size=8)
    model = CLIPSegForImageSegmentation(config).eval()
    class Processor:
        @staticmethod
        def tokenizer(*args, **kwargs):
            return {'input_ids': torch.tensor([[0, 3, 4, 2]]),
                    'attention_mask': torch.ones(1, 4, dtype=torch.long)}
        @staticmethod
        def image_processor(*args, **kwargs):
            return {'pixel_values': torch.zeros(1, 3, 24, 24)}
    cfg = TransferConfig(method='B', device='cpu', dtype='float32', segmentation_threshold=1e-6)
    engine = MaterialTransfer(cfg)
    engine.segmenter = (Processor(), model)
    image = Image.new('RGB', (40, 24), 'blue')
    reference, mask = engine.reference(image, 'ceramic')
    assert reference.size == mask.size == image.size
    assert mask.getbbox() is not None
