import io
import os
import sh
import cv2
import numpy as np
from PIL import Image


from ..cast import cast
from ..concat import grid_concat

from .ffmpeg_utils import FFmpegVideoLoader, FFmpegVideoDumper
from .video_utils import VideoLoader, VideoDumper


def save(x, path, engine="auto", quality=75):
    if isinstance(x, list):
        x = grid_concat(x)
    try:
        cast(x, type="pil").save(path, quality=quality)
    except Exception as e:
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
            save(x, path, engine=engine)
        else:
            raise e


def open(path, engine="auto", **kwargs):
    if path.endswith(".mp4"):
        if engine == "opencv":
            return VideoLoader(path, **kwargs)
        else:
            return FFmpegVideoLoader(path, **kwargs)
    else:
        return Image.open(path)

def view(x, *args):
    if args:
        x = grid_concat([x, *args])
    raw = io.BytesIO()
    x = cast(x, type="pil")
    if isinstance(x, list):
        x = grid_concat(x)
    x.save(raw, format="webp")
    print(sh.imgcat(_in=raw.getvalue()), end="", flush=True)
