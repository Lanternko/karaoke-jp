"""CLI: melody.mid + aligned.json -> melody_markers.mid"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import click

from karaoke_jp.midi_markers import inject_line_markers


@click.command()
@click.option("--midi", "midi_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--block-size", default=2, type=int, show_default=True)
def main(midi_path: str, aligned_path: str, out_path: str, block_size: int) -> None:
    n = inject_line_markers(midi_path, aligned_path, out_path, block_size=block_size)
    print(f"injected {n} page markers -> {out_path}")


if __name__ == "__main__":
    main()
