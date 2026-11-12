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


@pytest.mark.parametrize('mode', ['none', 'model', 'sequential'])
@pytest.mark.parametrize('tile_size', [None, 512])
def test_offload_loading_never_moves_entire_pipe_to_cuda(mode, tile_size, monkeypatch):
    import diffusers
    import transformers
    calls = []
    model = SimpleNamespace(eval=lambda: None)
    pipe = SimpleNamespace(**{name: model for name in
        ('unet', 'image_encoder', 'text_encoder', 'text_encoder_2', 'controlnet')})
    pipe.vae = SimpleNamespace(eval=lambda: None, enable_slicing=lambda: None)
    pipe.vae_scale_factor = 8
    pipe.load_ip_adapter = lambda *a, **kw: None
    pipe.set_ip_adapter_scale = lambda *a: None
    pipe.to = lambda device: calls.append(('to', device))
    pipe.enable_vae_tiling = lambda: calls.append(('tiling',))
    pipe.enable_model_cpu_offload = lambda **kw: calls.append(('model', kw['gpu_id']))
    pipe.enable_sequential_cpu_offload = lambda **kw: calls.append(('sequential', kw['gpu_id']))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(diffusers.ControlNetModel, 'from_pretrained', lambda *a, **kw: model)
    monkeypatch.setattr(transformers.CLIPVisionModelWithProjection, 'from_pretrained', lambda *a, **kw: model)
    monkeypatch.setattr(diffusers.StableDiffusionXLControlNetInpaintPipeline, 'from_pretrained', lambda *a, **kw: pipe)
    engine = MaterialTransfer(TransferConfig(cpu_offload=mode, vae_tiling=True,
                                           vae_tile_size=tile_size, device='cuda:2'))
    engine.load()
    engine.load()  # Loading twice must not install hooks twice.
    assert calls == [('tiling',), ('to', 'cuda:2') if mode == 'none' else (mode, 2)]
    if mode == 'sequential':
        assert 'vae' in pipe._exclude_from_cpu_offload
    if tile_size is not None:
        assert pipe.vae.tile_sample_min_size == 512
        assert pipe.vae.tile_latent_min_size == 64


def test_tiled_vae_preserves_output_resolution(monkeypatch):
    from diffusers import AutoencoderKL
    model = AutoencoderKL(block_out_channels=(8, 16), norm_num_groups=4,
        down_block_types=('DownEncoderBlock2D',) * 2,
        up_block_types=('UpDecoderBlock2D',) * 2, sample_size=128).eval()
    model.enable_tiling()
    model.tile_sample_min_size, model.tile_latent_min_size = 64, 32
    encoder_sizes, decoder_sizes = [], []
    encode, decode = model.encoder.forward, model.decoder.forward
    def encode_tile(x):
        encoder_sizes.append(x.shape[-2:])
        return encode(x)
    def decode_tile(x, *args, **kwargs):
        decoder_sizes.append(x.shape[-2:])
        return decode(x, *args, **kwargs)
    monkeypatch.setattr(model.encoder, 'forward', encode_tile)
    monkeypatch.setattr(model.decoder, 'forward', decode_tile)
    with torch.inference_mode():
        latent = model.encode(torch.zeros(1, 3, 128, 128)).latent_dist.mode()
        result = model.decode(latent).sample
    assert result.shape == (1, 3, 128, 128) and torch.isfinite(result).all()
    assert len(encoder_sizes) > 1 and max(max(size) for size in encoder_sizes) <= 64
    assert len(decoder_sizes) > 1 and max(max(size) for size in decoder_sizes) <= 32


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
