#!/usr/bin/env python3
"""Union two melody MIDIs: primary wins, fallback fills the silences.

Built for the GAME backbone: GAME's notes are score-accurate but it clips
soft low-register notes and long sustain tails — regions the mora-fitted
RMVPE chain does cover. Fallback notes are clipped to the primary's silent
gaps (with a small guard so joints never overlap) and only pieces of at
least --min-piece survive.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from karaoke_jp.melody import _write_midi  # noqa: E402
from karaoke_jp.score_melody import read_first_tempo_bpm, read_midi_notes  # noqa: E402

Note = tuple[float, float, int]


def union(
    primary: list[Note],
    fallback: list[Note],
    *,
    min_piece: float = 0.08,
    guard: float = 0.02,
) -> list[Note]:
    prim = sorted(primary)
    out = list(prim)
    for s, e, p in sorted(fallback):
        cursor = s
        for ps, pe, _ in prim:
            if pe <= cursor or ps >= e:
                continue
            if ps - cursor >= min_piece + 2 * guard:
                out.append((cursor + guard, ps - guard, p))
            cursor = max(cursor, pe)
        if e - cursor >= min_piece + 2 * guard:
            out.append((cursor + guard, e - guard, p))
    return sorted(out)


@click.command()
@click.option("--primary", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--fallback", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--min-piece", type=float, default=0.08, show_default=True)
@click.option("--guard", type=float, default=0.02, show_default=True)
def main(primary: str, fallback: str, out_path: str, min_piece: float, guard: float) -> None:
    prim = [(n.start, n.end, n.pitch) for n in read_midi_notes(Path(primary))]
    fb = [(n.start, n.end, n.pitch) for n in read_midi_notes(Path(fallback))]
    merged = union(prim, fb, min_piece=min_piece, guard=guard)
    _write_midi(merged, Path(out_path), tempo=read_first_tempo_bpm(Path(primary)))
    print(f"[melody-union] wrote {out_path} primary={len(prim)} "
          f"fallback_pieces={len(merged) - len(prim)} total={len(merged)}")


if __name__ == "__main__":
    main()
