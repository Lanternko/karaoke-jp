"""M5: CLI wrapper for karaoke_jp.mix.mix_vocals.

Usage:
    python scripts/mix_audio.py \\
        --instrumental outputs/<song>/instrumental.wav \\
        --vocals       outputs/<song>/vocals.wav \\
        --out          outputs/<song>/mixed.wav \\
        --vocal-ratio  0.30
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import click

from karaoke_jp.mix import mix_vocals


@click.command()
@click.option("--instrumental", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--vocals",       type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out",          type=click.Path(dir_okay=False), required=True)
@click.option("--vocal-ratio",  type=float, default=0.30, show_default=True,
              help="Vocal volume as a fraction of instrumental (0 = mute, 1 = equal).")
def main(instrumental: str, vocals: str, out: str, vocal_ratio: float) -> None:
    out_path = mix_vocals(instrumental, vocals, out, vocal_ratio=vocal_ratio)
    print(f"[mix] wrote {out_path}")


if __name__ == "__main__":
    main()
