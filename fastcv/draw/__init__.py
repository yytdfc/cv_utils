import numpy as np
import cv2
from PIL import Image 
from ..cast import cast

def blend(img, mask, engine="auto"):
    if engine == "auto":
        if isinstance(img, Image.Image):
            mask = cast(mask, "pil", color="L")
            empty = empty_like(img)
            return Image.composite(img, empty, mask)

            print(img, mask)
            return Image.blend(img, mask, alpha=0.5)
            empty = cast(np.empty((mask.shape[0], mask.shape[1], 3), dtype=np.uint8), "pil")
            return Image.blend(img, empty, alpha=mask)
        else:
            engine = "pil"
        print("I'm blend in the package")


def seamless_clone(img0, img1, mask):
    x0, y0, w0, h0 = cv2.boundingRect(mask)
    return cv2.seamlessClone(
        cast(img0, "np"), cast(img1, "np"), mask, 
        (x0+w0//2, y0+h0//2), cv2.NORMAL_CLONE,
    )


def empty_like(x):
    if isinstance(x, Image.Image):
        return Image.new(x.mode, x.size, (0, 0, 0))
    elif isinstance(x, np.ndarray):
        return np.zeros_like(x)
    elif isinstance(x, torch.Tensor):
        return torch.zeros_like(x)
    elif isinstance(x, cv2.UMat):
        return cv2.UMat(np.zeros_like(x))
    else:
        raise TypeError("x must be tensor, numpy, pil or cv2")
