import cv2
import numpy as np
from PIL import Image
import torch
from .utils import detect_layout, detect_range

# type: tensor, numpy / np, pil, cv2
# dtype: auto, fp32, fp16
# from_range: (0, 255), (-1.0, 1.0), (0.0, 1.0)
# to_range: (0, 255), (-1.0, 1.0), (0.0, 1.0)
# device: cuda, cpu
# color: bgr, rgb, gray, rgb2bgr, bgr2rgb, rgb2gray, bgr2gray, gray2rgb, gray2bgr
# layout: nchw, nhwc, hwc, chw, hw, hwc2nchw, hwc2nhwc, hwc2chw, hwc2hw, chw2nchw, chw2nhwc, chw2hwc, chw2hw, hw2nchw, hw2nhwc, hw2chw, hw2hwc

__DEFAULT_SETTING = {
    torch.Tensor: {
        "dtype": torch.float32,
        "range": [0.0, 1.0],
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "color": "rgb",
        "layout": "nchw",
    },
    np.ndarray: {
        "dtype": np.float32,
        "range": [0.0, 1.0],
        "device": "cpu",
        "color": "rgb",
        "layout": "chw",
    },
    Image.Image: {
        "dtype": np.float32,
        "range": [0.0, 1.0],
        "device": "cpu",
        "color": "rgb",
        "layout": "nchw",
    },
    cv2.UMat: {
        "dtype": np.uint8,
        "range": [0, 255],
        "device": "cpu",
        "color": "bgr",
        "layout": "hwc",
    },
    "cv2": {
        "dtype": np.uint8,
        "range": [0, 255],
        "device": "cpu",
        "color": "bgr",
        "layout": "hwc",
    },
}
# auto default setting:
#   tensor:
#       dtype: torch.float32
#       range: (0.0, 1.0)
#       



def _cast_tensor(x, dtype="auto", from_range="auto", to_range="auto", device="auto", color="auto", layout="auto"):
    if isinstance(x, torch.Tensor):
        raise NotImplementedError
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if x.ndim == 3:
            x.unsqueeze_(0)
        x = x.permute(0, 3, 1, 2).float()
        return x
    elif isinstance(x, Image.Image):
        x = np.array(x)
        return _cast_tensor(x)
    elif isinstance(x, cv2.UMat):
        x = torch.from_numpy(np.array(x))
        return _cast_tensor(x, dtype, to_range, device, color, layout)
    else:
        raise TypeError("x must be tensor, numpy, pil or cv2")


def _cast_numpy(x, dtype="fp32", to_range="auto", device="cuda", color="rgb", layout="nchw"):
    if isinstance(x, np.ndarray):
        if dtype == "auto":
            dtype = x.dtype
        if to_range == "auto":
            to_range = [0.0, 1.0]
        return x
    elif isinstance(x, list):
        return np.array([_cast_numpy(i, dtype, to_range, device, color, layout) for i in x])
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        return _cast_numpy(x, dtype, to_range, device, color, layout)
    elif isinstance(x, Image.Image):
        x = np.array(x)
        return _cast_numpy(x, dtype, to_range, device, color, layout)
    elif isinstance(x, cv2.UMat):
        x = np.array(x)
        return _cast_numpy(x, dtype, to_range, device, color, layout)
    
def _cast_pil(x, from_range="auto", to_range="auto", color="auto"):
    if isinstance(x, Image.Image):
        return x
    elif isinstance(x, list):
        return [_cast_pil(i, from_range=from_range, to_range=to_range, color=color) for i in x ]
    elif isinstance(x, np.ndarray):
        if from_range != to_range:
            if from_range == "auto":
                from_range = detect_range(x)
            if to_range == "auto":
                to_range = [0, 255]
            scale = (to_range[1] - to_range[0]) / (from_range[1] - from_range[0])
            x = x.astype(np.float32) * scale + (to_range[0] - from_range[0] * scale)
            x = x.clip(*to_range).astype(np.uint8)
        if x.dtype == np.float32 or x.dtype == np.float64:
            x = x.clip(0, 255).astype(np.uint8)
        layout = detect_layout(x)
        if layout == "chw":
            x = x.transpose(1, 2, 0)
        elif layout == "nchw":
            x = [i.transpose(1, 2, 0) for i in x]
        elif layout == "nhwc":
            x = [i for i in x]

        if isinstance(x, list):
            return [_cast_pil(i, color=color) for i in x]

        if color == "rgb2bgr" or color == "bgr2rgb":
            x = x[..., ::-1]

        x = Image.fromarray(x)
        if color == "rgb":
            x = x.convert("RGB")
        elif color == "rgba":
            x = x.convert("RGBA")
        elif color == "L" or color == "grey":
            x = x.convert("L")
        return x
    elif isinstance(x, torch.Tensor):
        return _cast_pil(x.detach().cpu().numpy(), from_range=from_range, to_range=to_range, color=color)
    elif isinstance(x, cv2.UMat):
        raise NotImplementedError
    else:
        raise TypeError("x must be tensor, numpy, pil or cv2")

def _cast_cv2(x, dtype="fp32", to_range="auto", device="cuda", color="rgb", layout="nchw"):
    if isinstance(x, cv2.UMat):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        #  x = cv2.UMat(x)
        return np.array(x)
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        return _cast_cv2(x, dtype, to_range, device, color, layout)
    elif isinstance(x, Image.Image):
        if x.mode == "RGB":
            color = "bgr" if color == "auto" else color.lower()
            if color.endswith("bgr"):
                x = np.array(x)[..., 2::-1]
            elif color.endswith("rgb"):
                x = np.array(x)
        if x.mode == "RGBA":
            color = "bgra" if color == "auto" else color.lower()
            if color.endswith("bgra"):
                x = np.array(x)
                x = np.concatenate([x[..., 2::-1], x[..., 3:4]], -1)
            elif color.endswith("rgba"):
                x = np.array(x)
            elif color.endswith("bgr"):
                x = np.array(x).astype(np.float32)
                x = x[:, :, 2::-1] * (x[..., 3:4] / 255.)
                x = x.round().clip(0, 255).astype(np.uint8)
            elif color.endswith("rgb"):
                x = np.array(x).astype(np.float32)
                x = x[:, :, :2] * (x[..., 3:4] / 255.)
                x = x.round().clip(0, 255).astype(np.uint8)
        elif color.endswith("rgb"):
            x = np.array(x)[:, :, ::-1]
        return cv2.UMat(x)
    else:
        raise TypeError("x must be tensor, numpy, pil or cv2")


def cast(x, type, dtype="auto", from_range="auto", to_range="auto", device="cuda", color="auto", layout="auto"):
    color = color.lower()
    if type == "tensor" or type == "torch":
        return _cast_tensor(x, dtype, to_range, device, color, layout)
    elif type == "np" or type == "numpy":
        return _cast_numpy(x, dtype, to_range, device, color, layout)
    elif type == "pil":
        return _cast_pil(x, from_range=from_range, to_range=to_range, color=color)
    elif type == "cv2":
        return _cast_cv2(x, dtype, to_range, device, color, layout)
    else:
        raise ValueError("type must be tensor, numpy, pil or cv2", type)
