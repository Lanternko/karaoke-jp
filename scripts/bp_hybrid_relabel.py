#!/usr/bin/env python3
"""BasicPitch second-opinion relabel for a melody MIDI.

For each note, if BasicPitch's strongest overlapping event sits exactly one
semitone away and clearly out-scores BasicPitch's support for the current
pitch, adopt the BasicPitch pitch.  This is the d1/s/r rule family from the
Chidori hybrid sweep, packaged for cross-song use.  Audio-side evidence only.
"""
from __future__ import annotations

import csv
from pathlib import Path

import click
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from karaoke_jp.melody import _write_midi  # noqa: E402
from karaoke_jp.score_melody import read_first_tempo_bpm, read_midi_notes  # noqa: E402

PITCH_LO, PITCH_HI = 40, 90


def load_events(path: Path) -> list[tuple[float, float, int, float]]:
    events = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            s, e, p, vel = float(row[0]), float(row[1]), int(row[2]), float(row[3])
            if PITCH_LO <= p <= PITCH_HI:
                events.append((s, e, p, vel / 127.0))
    return sorted(events)


def relabel(
    notes: list[tuple[float, float, int]],
    events: list[tuple[float, float, int, float]],
    *,
    max_dist: int = 1,
    min_score: float = 0.02,
    min_ratio: float = 1.2,
) -> tuple[list[tuple[float, float, int]], int]:
    out = []
    replaced = 0
    for s, e, p in notes:
        dur = e - s
        best_score, best_pitch = 0.0, None
        cur_support = 0.0
        for bs, be, bp, conf in events:
            ov = min(be, e) - max(bs, s)
            if ov < 0.3 * min(dur, be - bs):
                continue
            score = conf * min(1.0, ov / dur)
            if bp == p:
                cur_support = max(cur_support, score)
            elif 1 <= abs(bp - p) <= max_dist and score > best_score:
                best_score, best_pitch = score, bp
        if (
            best_pitch is not None
            and best_score >= min_score
            and best_score >= min_ratio * max(cur_support, 1e-6)
        ):
            out.append((s, e, best_pitch))
            replaced += 1
        else:
            out.append((s, e, p))
    return out, replaced


@click.command()
@click.option("--midi", "midi_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--basicpitch-csv", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--max-dist", type=int, default=1, show_default=True)
@click.option("--min-score", type=float, default=0.02, show_default=True)
@click.option("--min-ratio", type=float, default=1.2, show_default=True)
def main(midi_path, basicpitch_csv, out_path, max_dist, min_score, min_ratio):
    notes = [(n.start, n.end, n.pitch) for n in read_midi_notes(Path(midi_path))]
    events = load_events(Path(basicpitch_csv))
    fixed, replaced = relabel(
        notes, events, max_dist=max_dist, min_score=min_score, min_ratio=min_ratio
    )
    _write_midi(fixed, Path(out_path), tempo=read_first_tempo_bpm(Path(midi_path)))
    print(f"[bp-hybrid-relabel] wrote {out_path} notes={len(fixed)} replaced={replaced}")


if __name__ == "__main__":
    main()
