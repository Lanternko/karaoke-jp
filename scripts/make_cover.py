"""Generate a 1920x1080 karaoke cover / title card: title (Japanese Mincho)
centered, artist below.

Title uses Noto Serif CJK JP (明朝体). Backdrop is a video frame or image,
blurred + dimmed for legibility, with a soft dark band behind the text.

Usage:
  python scripts/make_cover.py --title 千鳥 --artist ヨルシカ \
      --video songs/chidori/background.mp4 --at 110 \
      --out outputs/chidori/cover.png
  # or --backdrop <image.png> instead of --video/--at
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Noto Serif CJK Bold .ttc -- index 0 is the JP (明朝) face.
MINCHO = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
JP_INDEX = 0
W, H = 1920, 1080


def load_backdrop(a) -> Image.Image:
    if a.backdrop:
        return Image.open(a.backdrop).convert("RGB").resize((W, H))
    if a.video:
        tmp = tempfile.mktemp(suffix=".png")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(a.at), "-i", a.video, "-frames:v", "1",
             "-vf", f"scale={W}:{H}", tmp], check=True, capture_output=True)
        im = Image.open(tmp).convert("RGB")
        Path(tmp).unlink(missing_ok=True)
        return im
    return Image.new("RGB", (W, H), (18, 18, 24))  # plain dark fallback


def fit_font(text: str, max_w: int, start: int) -> ImageFont.FreeTypeFont:
    size = start
    while size > 40:
        f = ImageFont.truetype(MINCHO, size, index=JP_INDEX)
        if f.getbbox(text)[2] <= max_w:
            return f
        size -= 6
    return ImageFont.truetype(MINCHO, size, index=JP_INDEX)


def centered(d, text, font, cy, fill):
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (W - tw) // 2 - bbox[0]
    y = cy - th // 2 - bbox[1]
    d.text((x, y), text, font=font, fill=fill,
           stroke_width=max(2, font.size // 60), stroke_fill=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--artist", required=True)
    ap.add_argument("--backdrop")
    ap.add_argument("--video")
    ap.add_argument("--at", type=float, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=float, default=0.5, help="0..1 darken amount")
    a = ap.parse_args()

    bg = load_backdrop(a).filter(ImageFilter.GaussianBlur(7))
    bg = Image.blend(bg, Image.new("RGB", (W, H), (0, 0, 0)), a.dim)
    # soft dark band behind the text for contrast
    band = Image.new("L", (W, H), 0)
    ImageDraw.Draw(band).rectangle([0, 350, W, 770], fill=110)
    bg = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)),
                         bg, band.filter(ImageFilter.GaussianBlur(90)))

    d = ImageDraw.Draw(bg)
    centered(d, a.title, fit_font(a.title, int(W * 0.82), 230), 500, (246, 246, 249))
    d.rectangle([W // 2 - 95, 646, W // 2 + 95, 649], fill=(208, 208, 214))
    centered(d, a.artist, fit_font(a.artist, int(W * 0.6), 78), 716, (224, 224, 230))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    bg.save(a.out)
    print(f"[cover] wrote {a.out} ({a.title} / {a.artist})")


if __name__ == "__main__":
    main()
