import cv2
import numpy as np
from PIL import Image
import torch

from .utils import shape



_INTERPOLATION_METHODS = {
    "pil": {
        'nearest': Image.Resampling.NEAREST,
        'bilinear': Image.Resampling.BILINEAR,
        'bicubic': Image.Resampling.BICUBIC,
        'lanczos': Image.Resampling.LANCZOS,
        'linear': Image.Resampling.BILINEAR,
        'box': Image.Resampling.BOX,
        'hamming': Image.Resampling.HAMMING,
        'cubic': Image.Resampling.BICUBIC,
    },
    "cv2": {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4,
        'linear': cv2.INTER_LINEAR,
        'box': cv2.INTER_AREA,
        'hamming': cv2.INTER_AREA,
        'cubic': cv2.INTER_CUBIC,
    },
    "torch": {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic',
        'lanczos': 'lanczos',
        'linear': 'bilinear',
        'box': 'area',
        'hamming': 'area',
        'cubic': 'bicubic',
    },
}


def resize(img, size=0, short=0, long=0, scale=0, h=0, w=0, mode="linear", pillow=False, engine="auto"):
    if engine == "auto":
        if isinstance(img, torch.Tensor):
            engine = "torch"
        elif isinstance(img, np.ndarray):
            engine = "cv2"
        elif isinstance(img, Image.Image):
            engine = "pil"
        else:
            raise TypeError("img must be torch.Tensor, np.ndarray or PIL.Image.Image")
    else:
        original_type = type(img)
        img = cast(img, engine)

    c, oh, ow = shape(img)

    mode = _INTERPOLATION_METHODS[engine][mode]
    mi, ma = min(oh, ow), max(oh, ow)
    if short:
        scale = short / mi
    elif long:
        scale = long / ma
    elif h and w:
        size = (h, w)
    elif h:
        scale = h / oh
    elif w:
        scale = w / oh

    if scale:
        if isinstance(scale, tuple):
            h, w = scale[0] * oh, scale[1] * ow
        else:
            h = scale * oh
            w = scale * ow
    elif size:
        if isinstance(size, tuple):
            h, w = size
        else:
            h = w = size
    else:
        raise ValueError("short, long, scale, size must be specified at least one")

    h = round(h)
    w = round(w)

    if isinstance(img, np.ndarray):
        return cv2.resize(img, (w, h), interpolation=mode)
    else:
        return img.resize((w, h), resample=mode)
