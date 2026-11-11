"""Image preparation shared by single runs, batches and the UI."""
import numpy as np
from PIL import Image, ImageOps


def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_mask(path, size):
    with Image.open(path) as image:
        mask = ImageOps.exif_transpose(image).convert("L")
    if mask.size != size:
        raise ValueError(f"Target mask size {mask.size} differs from target {size}")
    if mask.getextrema()[1] == 0:
        raise ValueError("Target mask is empty (white means edit)")
    return mask


def decolorize(target, mask):
    return Image.composite(target.convert("L").convert("RGB"), target, mask)


def depth_image(path, size):
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if not np.isfinite(array).all():
        raise ValueError("Depth contains non-finite pixels")
    # DPT/ZeST depth PNGs may be uint16 (Pillow can expose them as int32).
    if array.dtype == np.uint16 or array.max() > 255:
        array = np.clip(array / 256, 0, 255)
    return Image.fromarray(array.astype(np.uint8)).convert("RGB").resize(size, Image.Resampling.BICUBIC)


def automatic_mask(target):
    try:
        from rembg import remove
    except ImportError as exc:
        raise RuntimeError("Supply --mask or install the auto-mask extra") from exc
    # Foreground is opacity, not RGB brightness: black objects must remain foreground.
    return remove(target.convert("RGBA")).getchannel("A")
