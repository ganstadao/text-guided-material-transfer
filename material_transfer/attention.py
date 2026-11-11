"""Experimental reuse of a pretrained SDXL block on CLIP vision tokens.

This preserves the prototype's basic operation, not a trained/aligned multimodal
module. The SDXL and CLIP feature spaces need not be compatible.
"""
import math
import torch
from PIL import Image


def material_attention_mask(attention, hidden, text, valid_tokens):
    q = attention.head_to_batch_dim(attention.to_q(hidden))
    k = attention.head_to_batch_dim(attention.to_k(text))
    scores = attention.get_attention_scores(q, k, None)
    batch, patches = hidden.shape[:2]
    scores = scores.reshape(batch, attention.heads, patches, text.shape[1]).mean(1)
    valid = valid_tokens.to(scores.dtype)
    if not valid.any():
        raise ValueError("Material prompt contains no usable tokens")
    weights = (scores * valid[:, None, :]).sum(-1) / valid.sum(-1, keepdim=True)
    return torch.sigmoid(weights[..., None] * 5)


def heatmap_image(mask, size):
    patches = mask[0, 1:, 0].detach().float().cpu()
    side = math.isqrt(patches.numel())
    if side * side != patches.numel():
        raise ValueError("Vision tokens do not form a square patch grid")
    return Image.fromarray((patches.reshape(side, side).numpy() * 255).astype("uint8")).resize(size)


@torch.inference_mode()
def fuse_reference(pipe, image, phrase, block):
    encoder = pipe.image_encoder
    pixels = pipe.feature_extractor(images=image, return_tensors="pt").pixel_values
    vision = encoder.vision_model(pixels.to(device=pipe.device, dtype=encoder.dtype))
    text = []
    for tokenizer, model in [(pipe.tokenizer, pipe.text_encoder), (pipe.tokenizer_2, pipe.text_encoder_2)]:
        tokens = tokenizer(phrase, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        tokens = {key: value.to(pipe.device) for key, value in tokens.items()}
        text.append(model(**tokens).last_hidden_state)
    features = torch.cat(text, dim=-1).to(encoder.dtype)
    # Match the original prototype: last_hidden_state from both text encoders.
    # Exclude BOS, EOS and padding; include every subword in the material phrase.
    first = pipe.tokenizer(phrase, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    valid = first.attention_mask.bool().to(pipe.device)
    valid[:, 0] = False
    valid &= first.input_ids.to(pipe.device) != pipe.tokenizer.eos_token_id
    hidden = vision.last_hidden_state
    mask = material_attention_mask(block.attn2, hidden, features, valid)
    fused = block(hidden_states=hidden, encoder_hidden_states=features) * mask
    pooled = encoder.vision_model.post_layernorm(fused)[:, 0, :]
    return encoder.visual_projection(pooled), heatmap_image(mask, image.size)
