__version__ = "0.1.2"
__description__ = "A basic cv library"
__doc__ = """
Document:
    pass

"""

from .cast import cast
from .draw_text import draw_text
from .resize import resize
from .concat import concat
from .concat import grid_concat
from .utils import shape, info
from .timer import timer
from .io.video_utils import VideoLoader, VideoDumper
from .io.ffmpeg_utils import FFmpegVideoLoader, FFmpegVideoDumper
from .io import save, open, view
from .draw import empty_like, blend


__all__ = [
    "draw_text",
    "resize",
    "concat",
    "grid_concat",
    "cast",
    "view",
    "shape",
    "info",
    "VideoLoader",
    "VideoDumper",
    "FFmpegVideoLoader",
    "FFmpegVideoDumper",
    "timer",
]

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
