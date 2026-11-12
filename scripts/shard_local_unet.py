"""Split a local Diffusers UNet safetensors file without changing tensor bytes.

The original file is retained. The index is published only after all shards have
been verified, so Diffusers can then load smaller files on Windows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def hash_stream(stream, length):
    digest = hashlib.sha256()
    while length:
        block = stream.read(min(length, 8 * 1024 * 1024))
        if not block:
            raise ValueError('Unexpected end of tensor data')
        digest.update(block)
        length -= len(block)
    return digest.hexdigest()


def shard_unet(directory, max_shard_bytes=512 * 1024**2, variant='fp16'):
    directory = Path(directory).resolve()
    if max_shard_bytes < 1:
        raise ValueError('Shard size must be positive')
    suffix = f'.{variant}' if variant else ''
    source = directory / f'diffusion_pytorch_model{suffix}.safetensors'
    index_path = directory / f'diffusion_pytorch_model.safetensors.index{suffix}.json'
    if index_path.exists():
        raise FileExistsError(f'Shard index already exists: {index_path}')
    with source.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    data_start = 8 + header_size
    tensors = sorted(((k, v) for k, v in header.items() if k != '__metadata__'),
                     key=lambda item: item[1]['data_offsets'][0])
    groups, group, group_bytes, cursor = [], [], 0, 0
    for name, spec in tensors:
        start, end = spec['data_offsets']
        if start != cursor or end < start:
            raise ValueError('Tensor offsets must be contiguous')
        length = end - start
        if group and group_bytes + length > max_shard_bytes:
            groups.append(group)
            group, group_bytes = [], 0
        group.append((name, spec))
        group_bytes += length
        cursor = end
    if group:
        groups.append(group)
    if not tensors or source.stat().st_size != data_start + cursor:
        raise ValueError('Invalid source payload size')
    names = [f'diffusion_pytorch_model{suffix}-{i:05d}-of-{len(groups):05d}.safetensors'
             for i in range(1, len(groups) + 1)]
    if any((directory / name).exists() for name in names):
        raise FileExistsError('Shard files already exist; inspect them before retrying')
    weight_map, verified = {}, []
    with source.open('rb') as original:
        source_sha256 = hash_stream(original, source.stat().st_size)
        for filename, group in zip(names, groups):
            shard_header = {'__metadata__': header.get('__metadata__', {'format': 'pt'})}
            offset = 0
            for name, spec in group:
                length = spec['data_offsets'][1] - spec['data_offsets'][0]
                shard_header[name] = {**spec, 'data_offsets': [offset, offset + length]}
                weight_map[name] = filename
                offset += length
            encoded = json.dumps(shard_header, separators=(',', ':')).encode('utf-8')
            encoded += b' ' * (-len(encoded) % 8)
            path = directory / filename
            temporary = path.with_suffix('.tmp')
            digest = hashlib.sha256()
            with temporary.open('wb') as output:
                output.write(struct.pack('<Q', len(encoded)))
                output.write(encoded)
                for _, spec in group:
                    start, end = spec['data_offsets']
                    original.seek(data_start + start)
                    remaining = end - start
                    while remaining:
                        block = original.read(min(remaining, 8 * 1024 * 1024))
                        if not block:
                            raise ValueError('Unexpected end of source')
                        output.write(block)
                        digest.update(block)
                        remaining -= len(block)
            with temporary.open('rb') as check:
                check.seek(8 + len(encoded))
                if hash_stream(check, offset) != digest.hexdigest():
                    raise ValueError(f'Tensor payload verification failed: {filename}')
            temporary.replace(path)
            verified.append(dict(file=filename, payload_sha256=digest.hexdigest(), bytes=offset))
            print(f'Verified {filename}: {offset / 1024**2:.1f} MiB', flush=True)
    provenance = dict(source=source.name, source_sha256=source_sha256, shards=verified,
                      tensor_count=len(tensors), tensor_bytes=cursor)
    (directory / f'shard_provenance{suffix}.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    index_path.write_text(json.dumps(dict(metadata={'total_size': cursor}, weight_map=weight_map), indent=2), encoding='utf-8')
    return index_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--max-shard-mib', type=int, default=512)
    parser.add_argument('--variant', default='fp16')
    args = parser.parse_args()
    print(shard_unet(args.directory, args.max_shard_mib * 1024**2, args.variant))
