#!/usr/bin/env python3
"""Portrait (9:16) karaoke renderer — Pillow overlay frames piped to ffmpeg.

Reads the portrait grid JSON from make_portrait_grid.py and renders a
1080×1920 MP4. Four display rows, top to bottom:

    bars A   — pitch bar row (snake-packed)
    bars B
    lyric A  — lyric subtitle row (one aligned line each)
    lyric B

Bar lines and lyric lines are INDEPENDENT systems: each wipes by its own
real_start/real_end against the playback clock, so the cursor is always on
whatever is actually being sung.

Background: an MV video is composited under the overlay by ffmpeg
(blur-filled 9:16 + the 16:9 video centered in the top area). A still
image gets the same treatment in Pillow; with no background a flat dark
fill is used.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import click
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

W, H = 1080, 1920
FPS = 60
MARGIN_X = 60

# vertical layout (top -> bottom: bars A, bars B, MV, lyric A, lyric B)
BARS_A_TOP = 70
BAR_AREA_H = 190
BARS_B_TOP = BARS_A_TOP + BAR_AREA_H + 20
BARS_PANEL = (40, BARS_B_TOP + BAR_AREA_H + 20)
LYRIC_A_TOP = 1530
LYRIC_B_TOP = LYRIC_A_TOP + 130
LYRIC_PANEL = (LYRIC_A_TOP - 60, LYRIC_B_TOP + 110)

BARS_Y = {"A": BARS_A_TOP, "B": BARS_B_TOP}
LYRIC_Y = {"A": LYRIC_A_TOP, "B": LYRIC_B_TOP}

# MV: sides cropped so the picture runs taller, centered between the two
# panels over the blurred fill
MV_H = 960
MV_Y = (BARS_PANEL[1] + (LYRIC_PANEL[0] - BARS_PANEL[1] - MV_H) // 2)

BAR_RADIUS = 5
BAR_H_PX = 14

COL_BAR_UPCOMING = (150, 150, 150, 255)
COL_BAR_WIPED = (255, 140, 0, 255)
COL_BAR_PREVIEW = (90, 90, 90, 200)
COL_TEXT_UPCOMING = (235, 235, 235, 255)
COL_TEXT_WIPED = (255, 140, 0, 255)
COL_TEXT_PREVIEW = (150, 150, 150, 230)
COL_PANEL = (0, 0, 0, 150)
COL_CURSOR = (255, 255, 255, 220)

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_SIZE = 52
RUBY_SIZE = 24
CHAR_SPACING = 3

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _font_cache:
        try:
            _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
        except Exception:
            _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


def _bar_y(pitch: int, pitch_min: int, pitch_max: int) -> int:
    """Map MIDI pitch to y offset within the bar area (higher = higher)."""
    span = max(pitch_max - pitch_min, 1)
    frac = (pitch - pitch_min) / span
    y_bottom = BAR_AREA_H - BAR_H_PX - 8
    y_top = 8
    return int(y_bottom - frac * (y_bottom - y_top))


def _find_visible(lines: list[dict], row: str, t: float) -> dict | None:
    """Latest line on `row` whose [preview_start, replace_time) covers t."""
    found = None
    for ln in lines:
        if ln["row"] != row:
            continue
        if ln["preview_start"] <= t < ln["replace_time"]:
            found = ln
    return found


def _wipe_q(bars: list[dict], t: float) -> float:
    """Wipe cursor position (in quarter units) at time t."""
    if not bars:
        return 0.0
    for bar in bars:
        if t < bar["real_start"]:
            return bar["x_q"]
        if bar["real_start"] <= t <= bar["real_end"]:
            frac = (t - bar["real_start"]) / max(bar["real_end"] - bar["real_start"], 1e-6)
            return bar["x_q"] + frac * bar["w_q"]
    return bars[-1]["x_q"] + bars[-1]["w_q"]


def _draw_bar_row(draw: ImageDraw.ImageDraw, line: dict, t: float,
                  q_px: float, pitch_min: int, pitch_max: int) -> None:
    bars = line["bars"]
    top = BARS_Y[line["row"]]
    preview = t < line["time_start"]
    wipe = -1.0 if preview else _wipe_q(bars, t)

    for bar in bars:
        bx = MARGIN_X + bar["x_q"] * q_px
        bw = bar["w_q"] * q_px
        by = top + _bar_y(bar["pitch"], pitch_min, pitch_max)

        if preview:
            col = COL_BAR_PREVIEW
        elif bar["x_q"] + bar["w_q"] <= wipe:
            col = COL_BAR_WIPED
        elif bar["x_q"] < wipe:
            frac = (wipe - bar["x_q"]) / bar["w_q"]
            split_x = bx + frac * bw
            draw.rounded_rectangle([bx, by, split_x, by + BAR_H_PX],
                                   radius=BAR_RADIUS, fill=COL_BAR_WIPED)
            draw.rounded_rectangle([split_x, by, bx + bw, by + BAR_H_PX],
                                   radius=BAR_RADIUS, fill=COL_BAR_UPCOMING)
            continue
        else:
            col = COL_BAR_UPCOMING
        draw.rounded_rectangle([bx, by, bx + bw, by + BAR_H_PX],
                               radius=BAR_RADIUS, fill=col)

    # vertical cursor while this line is actually being sung
    if line["time_start"] <= t <= line["time_end"] and bars:
        cx = MARGIN_X + wipe * q_px
        draw.line([cx, top + 2, cx, top + BAR_AREA_H - 2],
                  fill=COL_CURSOR, width=3)


def _draw_partial_char(frame: Image.Image, ch: str, x: float, y: float,
                       cw: float, font: ImageFont.FreeTypeFont,
                       frac: float) -> None:
    """Paste the left `frac` of a glyph in the wiped colour over the base."""
    box_h = int(font.size * 1.6)
    tmp = Image.new("RGBA", (int(cw) + 8, box_h), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((0, 0), ch, fill=COL_TEXT_WIPED, font=font)
    cut = max(1, int(cw * frac))
    part = tmp.crop((0, 0, cut, box_h))
    frame.paste(part, (int(x), int(y)), part)


def _draw_lyric_row(frame: Image.Image, draw: ImageDraw.ImageDraw,
                    line: dict, t: float) -> None:
    chars = line["chars"]
    text_top = LYRIC_Y[line["row"]]
    preview = t < line["time_start"]
    usable = W - 2 * MARGIN_X

    # auto-fit: shrink the font until the line fits the width
    size = FONT_SIZE
    while size > 24:
        font = _load_font(size)
        widths = [font.getbbox(c["char"])[2] - font.getbbox(c["char"])[0]
                  for c in chars]
        total = sum(widths) + max(0, len(chars) - 1) * CHAR_SPACING
        if total <= usable:
            break
        size -= 4
    ruby_font = _load_font(RUBY_SIZE)

    cx = (W - total) / 2
    for ci, c in enumerate(chars):
        partial = None
        if preview:
            col = COL_TEXT_PREVIEW
        elif t >= c["real_end"]:
            col = COL_TEXT_WIPED
        elif t >= c["real_start"]:
            col = COL_TEXT_UPCOMING
            dur = max(c["real_end"] - c["real_start"], 1e-6)
            partial = (t - c["real_start"]) / dur
        else:
            col = COL_TEXT_UPCOMING

        draw.text((cx, text_top), c["char"], fill=col, font=font)
        if partial is not None:
            _draw_partial_char(frame, c["char"], cx, text_top,
                               widths[ci], font, partial)

        if "ruby" in c:
            ruby_col = COL_TEXT_WIPED if (not preview and t >= c["real_start"]) else col
            rbbox = ruby_font.getbbox(c["ruby"])
            rw = rbbox[2] - rbbox[0]
            rx = cx + (widths[ci] - rw) / 2
            draw.text((rx, text_top - RUBY_SIZE - 4), c["ruby"],
                      fill=ruby_col, font=ruby_font)
        cx += widths[ci] + CHAR_SPACING


def render_frame(frame: Image.Image, grid: dict, t: float) -> None:
    draw = ImageDraw.Draw(frame)
    q_px = (W - 2 * MARGIN_X) / grid["quarters_per_row"]
    pmin, pmax = grid["pitch_min"], grid["pitch_max"]

    draw.rectangle([0, BARS_PANEL[0], W, BARS_PANEL[1]], fill=COL_PANEL)
    draw.rectangle([0, LYRIC_PANEL[0], W, LYRIC_PANEL[1]], fill=COL_PANEL)

    for row in ("A", "B"):
        ln = _find_visible(grid["bar_lines"], row, t)
        if ln:
            _draw_bar_row(draw, ln, t, q_px, pmin, pmax)
        ll = _find_visible(grid["lyric_lines"], row, t)
        if ll:
            _draw_lyric_row(frame, draw, ll, t)


def _build_ffmpeg(audio_path: str, out_path: str, bg_video: str | None,
                  duration: float) -> list[str]:
    base = ["ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{W}x{H}", "-pix_fmt", "rgba", "-r", str(FPS),
            "-i", "-"]
    if bg_video:
        filt = (
            "[1:v]split[bga][bgb];"
            f"[bga]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},boxblur=20:2,eq=brightness=-0.25[blur];"
            f"[bgb]scale=-2:{MV_H},crop=min(iw\\,{W}):{MV_H}[mv];"
            f"[blur][mv]overlay=(W-w)/2:{MV_Y},fps={FPS}[bg];"
            "[bg][0:v]overlay=0:0,format=yuv420p[v]"
        )
        return base + [
            "-stream_loop", "-1", "-i", bg_video,
            "-i", audio_path,
            "-filter_complex", filt,
            "-map", "[v]", "-map", "2:a",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart", out_path]
    return base + [
        "-i", audio_path,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", out_path]


def _still_background(bg_path: str) -> Image.Image:
    """Blur-fill + centered image — same look as the video path."""
    src = Image.open(bg_path).convert("RGB")
    scale = max(W / src.width, H / src.height)
    fill = src.resize((int(src.width * scale), int(src.height * scale)),
                      Image.LANCZOS)
    fill = fill.crop(((fill.width - W) // 2, (fill.height - H) // 2,
                      (fill.width - W) // 2 + W, (fill.height - H) // 2 + H))
    fill = fill.filter(ImageFilter.GaussianBlur(20))
    fill = ImageEnhance.Brightness(fill).enhance(0.75)
    fg = src.resize((int(src.width * MV_H / src.height), MV_H), Image.LANCZOS)
    if fg.width > W:
        fg = fg.crop(((fg.width - W) // 2, 0, (fg.width - W) // 2 + W, MV_H))
    fill.paste(fg, ((W - fg.width) // 2, MV_Y))
    return fill.convert("RGBA")


@click.command()
@click.option("--grid", "grid_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Portrait grid JSON from make_portrait_grid.py")
@click.option("--audio", "audio_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--duration", type=float, default=None,
              help="Override duration in seconds (default: from audio).")
@click.option("--bg", "bg_path", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Background MV video (composited by ffmpeg) or still image.")
def main(grid_path, audio_path, out_path, duration, bg_path):
    grid = json.loads(Path(grid_path).read_text(encoding="utf-8"))

    if duration is None:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", audio_path],
            capture_output=True, text=True)
        duration = float(probe.stdout.strip())

    bg_video = None
    bg_still = None
    if bg_path:
        if Path(bg_path).suffix.lower() in (".mp4", ".mov", ".webm", ".mkv"):
            bg_video = bg_path
        else:
            bg_still = _still_background(bg_path)

    total_frames = int(duration * FPS)
    click.echo(f"[portrait] rendering {total_frames} frames ({duration:.1f}s) "
               f"at {FPS}fps bg={'video' if bg_video else 'still' if bg_still else 'flat'}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        _build_ffmpeg(audio_path, out_path, bg_video, duration),
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    for fi in range(total_frames):
        t = fi / FPS
        if bg_video:
            frame = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        elif bg_still:
            frame = bg_still.copy()
        else:
            frame = Image.new("RGBA", (W, H), (15, 15, 25, 255))
        render_frame(frame, grid, t)
        ffmpeg.stdin.write(frame.tobytes())
        if fi % (FPS * 30) == 0:
            click.echo(f"  frame {fi}/{total_frames} ({t:.0f}s)")

    ffmpeg.stdin.close()
    ret = ffmpeg.wait()
    if ret != 0:
        raise SystemExit(f"[portrait] ffmpeg exited with {ret}")
    click.echo(f"[portrait] wrote {out_path}")


if __name__ == "__main__":
    main()
