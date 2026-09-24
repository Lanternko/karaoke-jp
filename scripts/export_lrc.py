"""CLI: aligned.json -> MID2BAR-flavored .lrc"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import click

from karaoke_jp.lrc_export import export_lrc


def _page_starts_real(grid_midi: str, warp_path: str) -> list[float]:
    """Page-marker display times (P01, P02, ...) mapped back to real time."""
    import json

    import mido
    import numpy as np

    mf = mido.MidiFile(grid_midi)
    starts, tick, sec, tempo = [], 0, 0.0, 500000
    for msg in mido.merge_tracks(mf.tracks):
        sec += mido.tick2second(msg.time, mf.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "marker" and msg.text.startswith("P"):
            starts.append(sec)
    warp = json.loads(Path(warp_path).read_text())
    return np.interp(starts, warp["display"], warp["real"]).tolist()


@click.command()
@click.argument("aligned_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "-o", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--block-size", default=2, type=int, show_default=True)
@click.option("--page-grid", "grid_midi", type=click.Path(exists=True, dir_okay=False),
              default=None, help="display MIDI from make_display_grid (P01.. page "
              "markers); with --time-warp, lines are cut where the bars flip page.")
@click.option("--time-warp", "warp_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="warp JSON paired with --page-grid.")
@click.option("--breaks-from", "breaks_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="per-song overrides/<song>_render.json; its "
              "\"lrc_breaks\" list (lines with | at the cut) is applied first.")
def main(aligned_path: str, out_path: str, block_size: int,
         grid_midi: str | None, warp_path: str | None, breaks_path: str | None) -> None:
    page_starts = None
    if grid_midi or warp_path:
        if not (grid_midi and warp_path):
            raise click.UsageError("--page-grid and --time-warp go together")
        page_starts = _page_starts_real(grid_midi, warp_path)
    breaks = None
    if breaks_path:
        import json

        breaks = json.loads(Path(breaks_path).read_text(encoding="utf-8")).get("lrc_breaks")
    export_lrc(Path(aligned_path), Path(out_path), block_size=block_size,
               page_starts=page_starts, manual_breaks=breaks)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
