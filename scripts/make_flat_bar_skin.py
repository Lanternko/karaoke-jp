#!/usr/bin/env python3
"""Generate the "8_flat" MID2BAR bar skin: clean flat bars, no outline,
no glow (Kojek 2026-06-11: 減弱光暈/描邊, closer to plain JOYSOUND bars).

Sprites are written into the (gitignored) MID2BAR images tree; this script is
the tracked, reproducible source: re-run it on a fresh checkout.

CONTRACT with tools.draw_stretchable_rounded_rect: the renderer blits the
left cap at x - PADDING and expects the outer PADDING px of each cap (and
the top/bottom PADDING rows) to be transparent margin.  The bar body must
therefore span exactly [PADDING, 3*SEG_W - PADDING] in the wide strip —
body pixels outside that range render *outside* the note's time span and
eat the inter-mora gaps (root cause of the v8 "no gap between notes" bug).
"""
from __future__ import annotations

from pathlib import Path

import click
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "third_party" / "MID2BAR-Player" / "images" / "bar" / "8_flat"

SEG_W, SEG_H = 200, 300
PADDING = 100                      # transparent margin; must match app.py bar_padding
BODY_TOP, BODY_BOTTOM = 100, 200   # vertical body = the non-padding band
RADIUS = 16                        # soft ends, not a full pill

# state -> fill RGBA (no outline: clean flat rectangles)
STYLES = {
    "back":   (255, 255, 255, 255),   # upcoming: solid white
    "fill":   (255, 224, 40, 255),    # being sung: solid yellow
    "passed": (255, 224, 40, 255),    # sung: same yellow
}


def _draw_state(out_dir: Path, state: str, fill) -> None:
    # draw one wide rounded bar across the non-padding region, then slice
    # into left/mid/right segments so MID2BAR's 3-slice assembly reproduces
    # it at any width
    wide = Image.new("RGBA", (SEG_W * 3, SEG_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(wide)
    d.rounded_rectangle(
        [PADDING, BODY_TOP, SEG_W * 3 - 1 - PADDING, BODY_BOTTOM],
        radius=RADIUS, fill=fill,
    )
    for i, name in enumerate(("left", "mid", "right")):
        seg = wide.crop((SEG_W * i, 0, SEG_W * (i + 1), SEG_H))
        seg.save(out_dir / f"{state}_{name}.png")


@click.command()
@click.option("--out-dir", type=click.Path(file_okay=False), default=str(DEFAULT_OUT),
              show_default=True)
def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for state, fill in STYLES.items():
        _draw_state(out, state, fill)
    # fully transparent glow: the renderer still blits it, we just remove the halo
    Image.new("RGBA", (500, 127), (0, 0, 0, 0)).save(out / "glow.png")
    click.echo(f"[flat-bar-skin] wrote {out}")


if __name__ == "__main__":
    main()
