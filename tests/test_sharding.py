import json
import torch
from safetensors.torch import load_file, save_file
import pytest
from scripts.shard_local_unet import shard_unet


def test_sharding_preserves_all_tensor_values_and_original(tmp_path):
    tensors = {'a': torch.arange(64, dtype=torch.float16).reshape(8, 8),
               'b': torch.tensor([-1, 2], dtype=torch.int64),
               'c': torch.linspace(-1, 1, 20)}
    source = tmp_path / 'diffusion_pytorch_model.fp16.safetensors'
    save_file(tensors, source, metadata={'format': 'pt'})
    original = source.read_bytes()
    index = json.loads(shard_unet(tmp_path, max_shard_bytes=160).read_text())
    restored = {}
    for filename in set(index['weight_map'].values()):
        restored.update(load_file(tmp_path / filename))
    assert set(restored) == set(tensors)
    assert len(set(index['weight_map'].values())) > 1
    for name, expected in tensors.items():
        assert restored[name].dtype == expected.dtype
        assert torch.equal(restored[name], expected)
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        shard_unet(tmp_path)
