import localization
import numpy as np
import os
import sys
from typing import Optional
from PIL import Image, ImageDraw, ImageFont

pil_fonts = {}
_last_font_source = None

def get_last_font_source():
    return _last_font_source

def _get_pil_font (font, size):
    global pil_fonts
    global _last_font_source

    def _iter_font_candidates(font_value: Optional[str]):
        env_font = os.environ.get('DFL_TTF_FONT', None)
        if env_font:
            yield env_font

        if font_value is not None:
            yield font_value
            yield font_value + ".ttf"
            yield font_value + ".ttc"
            yield font_value + ".otf"

        platform = sys.platform
        if platform == 'darwin':
            yield "/System/Library/Fonts/PingFang.ttc"
            yield "/System/Library/Fonts/STHeiti Medium.ttc"
            yield "/System/Library/Fonts/STHeiti Light.ttc"
            yield "/System/Library/Fonts/Supplemental/Songti.ttc"
            yield "/System/Library/Fonts/Supplemental/STSong.ttf"
            yield "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
        elif platform[0:3] == 'win':
            win_dir = os.environ.get('WINDIR', r"C:\\Windows")
            yield os.path.join(win_dir, "Fonts", "msyh.ttc")
            yield os.path.join(win_dir, "Fonts", "msyh.ttf")
            yield os.path.join(win_dir, "Fonts", "msyhbd.ttc")
            yield os.path.join(win_dir, "Fonts", "msjh.ttc")
            yield os.path.join(win_dir, "Fonts", "simsun.ttc")
            yield os.path.join(win_dir, "Fonts", "simsun.ttf")
            yield os.path.join(win_dir, "Fonts", "simsun.ttc")
            yield os.path.join(win_dir, "Fonts", "simhei.ttf")
            yield os.path.join(win_dir, "Fonts", "simkai.ttf")
            yield os.path.join(win_dir, "Fonts", "fangsong.ttf")
        else:
            yield "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
            yield "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
            yield "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.otf"
            yield "/usr/share/fonts/truetype/arphic/uming.ttc"

    try:
        font_str_id = '%s_%d' % (font, size)
        if font_str_id in pil_fonts:
            return pil_fonts[font_str_id]

        for candidate in _iter_font_candidates(font):
            if not candidate:
                continue
            try:
                if os.path.isabs(candidate) and not os.path.exists(candidate):
                    continue
                pil_font = ImageFont.truetype(candidate, size=size, encoding="unic")
                pil_fonts[font_str_id] = pil_font
                _last_font_source = candidate
                return pil_font
            except:
                continue

        pil_font = ImageFont.load_default()
        pil_fonts[font_str_id] = pil_font
        _last_font_source = 'PIL_default'
        return pil_font
    except:
        return ImageFont.load_default()

def get_text_image( shape, text, color=(1,1,1), border=0.2, font=None):
    h,w,c = shape
    try:
        pil_font = _get_pil_font( localization.get_default_ttf_font_name() , h-2)

        canvas = Image.new('RGB', (w,h) , (0,0,0) )
        draw = ImageDraw.Draw(canvas)
        offset = ( 0, 0)
        draw.text(offset, text, font=pil_font, fill=tuple((np.array(color)*255).astype(np.int32)) )

        result = np.asarray(canvas) / 255

        if c > 3:
            result = np.concatenate ( (result, np.ones ((h,w,c-3)) ), axis=-1 )
        elif c < 3:
            result = result[...,0:c]
        return result
    except:
        return np.zeros ( (h,w,c) )

def draw_text( image, rect, text, color=(1,1,1), border=0.2, font=None):
    h,w,c = image.shape

    l,t,r,b = rect
    l = np.clip (l, 0, w-1)
    r = np.clip (r, 0, w-1)
    t = np.clip (t, 0, h-1)
    b = np.clip (b, 0, h-1)

    image[t:b, l:r] += get_text_image (  (b-t,r-l,c) , text, color, border, font )


def draw_text_lines (image, rect, text_lines, color=(1,1,1), border=0.2, font=None):
    text_lines_len = len(text_lines)
    if text_lines_len == 0:
        return

    l,t,r,b = rect
    h = b-t
    h_per_line = h // text_lines_len

    for i in range(0, text_lines_len):
        draw_text (image, (l, i*h_per_line, r, (i+1)*h_per_line), text_lines[i], color, border, font)

def get_draw_text_lines ( image, rect, text_lines, color=(1,1,1), border=0.2, font=None):
    image = np.zeros ( image.shape, dtype=np.float32 )
    draw_text_lines ( image, rect, text_lines, color, border, font)
    return image
