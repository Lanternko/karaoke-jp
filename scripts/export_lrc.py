"""CLI: aligned.json -> MID2BAR-flavored .lrc"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import click

from karaoke_jp.lrc_export import export_lrc


@click.command()
@click.argument("aligned_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "-o", "out_path", type=click.Path(dir_okay=False), required=True)
def main(aligned_path: str, out_path: str) -> None:
    export_lrc(Path(aligned_path), Path(out_path))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
