import numpy as np
from PIL import Image
import cv2
import torch


def paras(x):
    pass

def shape(x):
    if isinstance(x, (torch.Tensor, np.ndarray)):
        l = detect_layout(x)
        if l == "nchw":
            c, h, w = x.shape[1:]
        elif l == "nhwc":
            h, w, c = x.shape[1:]
        elif l == "chw":
            c, h, w = x.shape
        elif l == "hwc":
            h, w, c = x.shape
        elif l == "hw":
            h, w = x.shape
            c = 1
        else:
            raise valueerror("l must be nchw, nhwc, chw, hwc or hw")
        return (c, h, w)
    elif isinstance(x, Image.Image):
        w, h = x.size
        c = 1 if x.mode in ("L", "P") else 3
        c = 4 if x.mode == "RGBA" else c
        return (c, h, w)
    elif isinstance(x, cv2.UMat):
        return shape(x.get())
    else:
        raise TypeError("x must be tensor, numpy, pil or cv2")


def detect_range(x):
    xmin, xmax = x.min(), x.max()
    if x < -128:
        xmin = -256
    elif x < -32:
        xmin = -128
    elif x < -0.51:
        xmin = -1
    elif x < 0:
        xmin = -0.5
    elif x < 0.5:
        xmin = 0
    elif x < 0.9:
        xmin = 0.5
    else:
        xmin = 1
    if x > 128:
        xmax = 256
    elif x > 32:
        xmax = 128
    elif x > 0.51:
        xmax = 1
    elif x > 0:
        xmax = 0.5
    elif x > -0.5:
        xmax = 0
    elif x > -0.9:
        xmax = -0.5
    else:
        xmax = -1

    return xmin, xmax


def detect_layout(x):
    if isinstance(x, (torch.Tensor, np.ndarray)):
        if x.ndim == 4:
            if x.shape[1] < x.shape[2] and x.shape[1] < x.shape[3]:
                return "nchw"
            elif x.shape[3] < x.shape[1] and x.shape[3] < x.shape[2]:
                return "nhwc"
            return "nchw"
        elif x.ndim == 3:
            if x.shape[0] < x.shape[1] and x.shape[0] < x.shape[2]:
                return "chw"
            elif x.shape[2] < x.shape[0] and x.shape[2] < x.shape[1]:
                return "hwc"
            else:
                return "chw"
        elif x.ndim == 2:
            return "hw"
        else:
            raise valueerror("x must be 2, 3 or 4 dimension")
    elif isinstance(x, image.image):
        return "hwc"
    elif isinstance(x, cv2.umat):
        return "hwc"


def color(x):
    if isinstance(x, torch.Tensor):
        return x.shape[0]
    elif isinstance(x, np.ndarray):
        return x.shape[0]
    elif isinstance(x, Image.Image):
        return x.mode.lower()
    elif isinstance(x, cv2.UMat):
        c, h, w = shape(x)
        if c == 1:
            return "gray"
        elif c == 3:
            return "bgr"
        elif c == 4:
            return "bgra"
        else:
            raise ValueError("c must be 1, 3 or 4")
    else:
        raise TypeError("x must be tensor, numpy, pil or cv2")


def info(x):
    c, h, w = shape(x)
    if isinstance(x, torch.Tensor):
        return f"Tensor [{x.shape}, {x.dtype}, {x.device}], {detect_layout(x)}"
    elif isinstance(x, np.ndarray):
        return f"NDArray [{x.shape}, {x.dtype}], {detect_layout(x)}"
    elif isinstance(x, Image.Image):
        return f"PILImage: {shape(x)}, {color(x)}, detect_layout: {detect_layout(x)}"
    elif isinstance(x, cv2.UMat):
        return f"UMat [], {detect_layout(x)}, {color(x)}"
    else:
        raise TypeError("x must be tensor, numpy, pil or cv2")
