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


QUANT_MULTIPLES = (0.25, 0.5, 1.0, 2.0, 4.0)  # 2^-2 .. 2^2 quarter notes


def quantize_durations(notes: list[Note], *, quarter: float) -> list[Note]:
    """Snap each note's DURATION to the nearest power-of-two multiple of a
    quarter note (16th .. whole). Onsets stay put; the visual gap pass after
    this caps tails at the next onset, so quantizing cannot re-introduce
    overlaps that survive to the screen."""
    import math

    out: list[Note] = []
    for s, e, p in notes:
        dur = e - s
        target = min(
            (m * quarter for m in QUANT_MULTIPLES),
            key=lambda d: abs(math.log(dur / d)) if dur > 0 else 0.0,
        )
        out.append((s, s + target, p))
    return out


def count_overlaps(notes: list[Note]) -> int:
    ordered = sorted(notes)
    return sum(1 for a, b in zip(ordered, ordered[1:]) if b[0] < a[1] - 1e-6)


def apply_visual_gap(notes: list[Note], *, gap: float, min_len: float = 0.05) -> list[Note]:
    """Shorten note tails so consecutive bars never touch on screen."""
    out = sorted(notes)
    trimmed: list[Note] = []
    for i, (s, e, p) in enumerate(out):
        if i + 1 < len(out):
            nxt = out[i + 1][0]
            if nxt - e < gap:
                e = max(s + min_len, nxt - gap)
        trimmed.append((s, e, p))
    return trimmed


@click.command()
@click.option("--primary", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--fallback", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--min-piece", type=float, default=0.08, show_default=True)
@click.option("--guard", type=float, default=0.02, show_default=True)
@click.option("--visual-gap", type=float, default=0.0, show_default=True,
              help="Display-only: trim each note's tail so adjacent bars show "
              "a visible gap between morae (sprite-level insets vanish after "
              "scaling; the gap has to live in the time domain).")
@click.option("--quantize-bpm-file", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Display-only: snap note durations to 2^-2..2^2 quarter "
              "notes using the BPM in this sidecar file.")
def main(primary: str, fallback: str, out_path: str, min_piece: float, guard: float,
         visual_gap: float, quantize_bpm_file: str | None) -> None:
    prim = [(n.start, n.end, n.pitch) for n in read_midi_notes(Path(primary))]
    fb = [(n.start, n.end, n.pitch) for n in read_midi_notes(Path(fallback))]
    merged = union(prim, fb, min_piece=min_piece, guard=guard)
    if quantize_bpm_file:
        quarter = 60.0 / float(Path(quantize_bpm_file).read_text().strip())
        merged = quantize_durations(merged, quarter=quarter)
    if visual_gap > 0:
        merged = apply_visual_gap(merged, gap=visual_gap)
    overlaps = count_overlaps(merged)
    _write_midi(merged, Path(out_path), tempo=read_first_tempo_bpm(Path(primary)))
    print(f"[melody-union] wrote {out_path} primary={len(prim)} "
          f"fallback_pieces={len(merged) - len(prim)} total={len(merged)} "
          f"overlaps={overlaps}")


if __name__ == "__main__":
    main()
