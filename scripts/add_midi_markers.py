"""CLI: melody.mid + (aligned.json | bpm) -> melody_markers.mid"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import click

from karaoke_jp.midi_markers import inject_beat_markers, inject_line_markers


@click.command()
@click.option("--midi", "midi_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Required for --mode line; optional in --mode beat to filter notes to lyric windows.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option(
    "--mode",
    type=click.Choice(["line", "beat"]),
    default="beat",
    show_default=True,
    help="line = page-per-block-of-lyrics (legacy); beat = fixed quarter-note pages "
    "for uniform pixels-per-quarter on the bar display.",
)
@click.option("--block-size", default=2, type=int, show_default=True,
              help="(--mode line) lyric blocks per page.")
@click.option("--bpm", type=float, default=None, help="(--mode beat) song BPM.")
@click.option("--bpm-file", type=click.Path(exists=True, dir_okay=False),
              default=None, help="(--mode beat) sidecar with BPM as a single float.")
@click.option("--quarters-per-page", default=8, type=int, show_default=True,
              help="(--mode beat) quarter notes per page = visual horizontal scale.")
@click.option("--note-window-margin", default=0.25, type=float, show_default=True,
              help="(--mode beat with --aligned) seconds of padding around lyric windows when filtering notes.")
def main(
    midi_path: str,
    aligned_path: str | None,
    out_path: str,
    mode: str,
    block_size: int,
    bpm: float | None,
    bpm_file: str | None,
    quarters_per_page: int,
    note_window_margin: float,
) -> None:
    if mode == "line":
        if aligned_path is None:
            raise click.UsageError("--aligned is required for --mode line.")
        n = inject_line_markers(midi_path, aligned_path, out_path, block_size=block_size)
    else:  # beat
        if bpm is None and bpm_file is None:
            raise click.UsageError("--bpm or --bpm-file required for --mode beat.")
        if bpm is None:
            bpm = float(Path(bpm_file).read_text().strip())
        n = inject_beat_markers(
            midi_path,
            out_path,
            bpm=bpm,
            quarters_per_page=quarters_per_page,
            aligned_path=aligned_path,
            note_window_margin=note_window_margin,
        )
    print(f"injected {n} page markers ({mode}) -> {out_path}")


if __name__ == "__main__":
    main()
