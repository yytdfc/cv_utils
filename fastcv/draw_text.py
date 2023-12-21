import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .cast import cast
from .utils import shape
from .environment import ENV


def draw_text(
    im, 
    text, 
    font_scale=0.1, 
    font_size=0,
    font_type="PingFang",
    align="top_center", 
    margin_scale=0.01, 
    color=(255, 0, 0, 255), 
    tx=-1, 
    ty=-1, 
    text_width=80,
    stroke_width=0,
    spacing=0,
):
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    _, ori_h, ori_w  = shape(im)
    impil = cast(im, "pil")
    if len(text) > 0:    
        if text_width:
            text_split = text.split()
            text_lines = []
            text_line = text_split[0]
            for t in text_split[1:]:
                if len(text_line) + len(t) > text_width:
                    text_lines.append(text_line)
                    text_line = t
                else:
                    text_line += " " + t
            text_lines.append(text_line)
            text = "\n".join(text_lines)
        draw = ImageDraw.Draw(impil)
        if font_size == 0:
            font_size = round(min(ori_h, ori_w) * font_scale)

        font_index = 0
        if font_type.split("-")[-1].isdigit():
            font_index = font_type.split("-")[-1]
            font_type = font_type[:-len(font_index) - 1]
        if ENV.is_mac():
            fontStyle = ImageFont.truetype(font_type, font_size, encoding="utf-8", index=int(font_index))
        else:
            fontStyle = ImageFont.truetype(
                #  f"{os.path.dirname(__file__)}/PingFang.ttc", font_size, encoding="utf-8"
                f"{os.path.dirname(__file__)}/Roboto-Regular.ttf", font_size, encoding="utf-8"
            )
        # fontStyle = ImageFont.load_default()
        # fontStyle.getsize('test')
        h_word = font_size
        w_word = fontStyle.getlength(text)
        if tx>= 0:
            pass
        elif align.endswith("left"):
            tx = round(ori_w * margin_scale)
        elif align.endswith("right"):
            tx = round(ori_w * (1 - margin_scale) - w_word)
        else:
            # defalut center
            tx = round((ori_w - w_word) / 2)
        if ty>= 0:
            pass
        elif align.startswith("bottom"):
            ty = round(ori_h * (1 - margin_scale) - h_word)

        elif align.startswith("top"):
            ty = round(ori_h * margin_scale)
        else:
            # defalut center
            ty = round((ori_h - h_word) / 2)

        draw.text((tx, ty), text, font=fontStyle, fill=(color), stroke_width=stroke_width,spacing=spacing)
        return impil
    else:
        return im
