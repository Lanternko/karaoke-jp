#!/usr/bin/env python3
"""MIREX-style note-level transcription metrics (COn / COnP / COnPOff).

Wraps mir_eval.transcription so the pitch benchmark can speak the same
language as the literature (MIREX 2020 Singing Transcription: onset 100 ms,
pitch 50 cents, offset max(50 ms, 0.2 x ref duration)).

A small global-offset sweep per candidate compensates for the known
source-timing offsets between our references and candidates (the frame
evaluator does the same); the chosen shift is reported.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import click
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mir_eval.transcription as mt  # noqa: E402

from karaoke_jp.score_melody import read_midi_notes  # noqa: E402


def _load(path: str, merge_gap: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    notes = [(n.start, n.end, n.pitch) for n in read_midi_notes(Path(path))]
    if merge_gap is not None:
        merged: list[list[float | int]] = []
        for s, e, pch in sorted(notes):
            if merged and pch == merged[-1][2] and s - merged[-1][1] <= merge_gap:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e, pch])
        notes = [tuple(n) for n in merged]
    intervals = np.array([[s, e] for s, e, _ in notes], dtype=float)
    hz = np.array([440.0 * 2 ** ((pch - 69) / 12) for _s, _e, pch in notes], dtype=float)
    return intervals, hz


def _scores(ref, est, *, onset_tol: float, shift: float) -> dict[str, float]:
    ri, rh = ref
    ei, eh = est
    ei = ei + shift
    onp = mt.precision_recall_f1_overlap(
        ri, rh, ei, eh,
        onset_tolerance=onset_tol, pitch_tolerance=50.0, offset_ratio=None,
    )
    onpoff = mt.precision_recall_f1_overlap(
        ri, rh, ei, eh,
        onset_tolerance=onset_tol, pitch_tolerance=50.0,
        offset_ratio=0.2, offset_min_tolerance=0.05,
    )
    on = mt.onset_precision_recall_f1(ri, ei, onset_tolerance=onset_tol)
    return {
        "COn_F": on[2], "COnP_P": onp[0], "COnP_R": onp[1], "COnP_F": onp[2],
        "COnPOff_F": onpoff[2],
    }


@click.command()
@click.option("--reference-midi", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--candidate", "candidates", multiple=True, required=True,
              help="LABEL:PATH, repeatable.")
@click.option("--onset-tolerance", type=float, default=0.1, show_default=True,
              help="MIREX ST uses 0.1; much of the literature uses 0.05.")
@click.option("--max-shift", type=float, default=0.06, show_default=True)
@click.option("--shift-step", type=float, default=0.02, show_default=True)
@click.option("--merge-same-pitch", type=float, default=None,
              help="Normalize BOTH sides by merging same-pitch notes with gaps "
              "up to this many seconds (karaoke bars merge repeated notes; "
              "mixed merge conventions otherwise dominate the note metrics).")
@click.option("--out-tsv", type=click.Path(dir_okay=False), default=None)
def main(reference_midi, candidates, onset_tolerance, max_shift, shift_step,
         merge_same_pitch, out_tsv):
    ref = _load(reference_midi, merge_gap=merge_same_pitch)
    shifts = np.arange(-max_shift, max_shift + 1e-9, shift_step)
    rows = []
    for spec in candidates:
        label, path = spec.split(":", 1)
        est = _load(path, merge_gap=merge_same_pitch)
        best = None
        for shift in shifts:
            s = _scores(ref, est, onset_tol=onset_tolerance, shift=float(shift))
            if best is None or s["COnP_F"] > best[1]["COnP_F"]:
                best = (float(shift), s)
        shift, s = best
        rows.append({"candidate": label, "shift": f"{shift:+.2f}",
                     **{k: f"{v:.4f}" for k, v in s.items()}})
    cols = ["candidate", "shift", "COn_F", "COnP_P", "COnP_R", "COnP_F", "COnPOff_F"]
    widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in cols}
    click.echo("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        click.echo("  ".join(r[c].ljust(widths[c]) for c in cols))
    if out_tsv:
        with open(out_tsv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
