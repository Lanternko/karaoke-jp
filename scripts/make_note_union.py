#!/usr/bin/env python3
"""Build note-level unions of two MIR-ST500-format predictions, to test whether
combining complementary strengths beats either alone (survey: ensemble to break
the COnPOff offset hole).

Modes:
  coverage          A primary; add B notes that fill temporal gaps (B note kept
                    iff <``--cov-overlap`` fraction of its span is already
                    covered by A). Raises recall without double-counting.
  offset_transplant A onsets+pitches kept verbatim; each A note's OFFSET is
                    replaced by the offset of the matching B note (onset within
                    ``--onset-tol`` and same rounded pitch) when one exists.
                    Directly tests "GAME coverage + CE+CTC learned offset".
  both              offset_transplant, then coverage-fill from B.

Format: {"song_id": [[onset, offset, midi], ...], ...}.
"""
from __future__ import annotations

import json
from pathlib import Path

import click


def _covered_fraction(span, intervals):
    s, e = span
    dur = e - s
    if dur <= 0:
        return 1.0
    covered = 0.0
    for a, b in intervals:
        covered += max(0.0, min(e, b) - max(s, a))
    return min(1.0, covered / dur)


def coverage_union(prim, sec, cov_overlap):
    prim_spans = [(n[0], n[1]) for n in prim]
    out = [list(n) for n in prim]
    for n in sec:
        if _covered_fraction((n[0], n[1]), prim_spans) < cov_overlap:
            out.append(list(n))
    return sorted(out)


def offset_transplant(prim, sec, onset_tol):
    sec_sorted = sorted(sec)
    out = []
    for p in prim:
        on, off, pit = p[0], p[1], round(p[2])
        best = None
        for s in sec_sorted:
            if abs(s[0] - on) <= onset_tol and round(s[2]) == pit:
                d = abs(s[0] - on)
                if best is None or d < best[0]:
                    best = (d, s[1])
        new_off = best[1] if best is not None else off
        out.append([on, max(new_off, on + 1e-3), p[2]])
    return sorted(out)


@click.command()
@click.option("--primary", required=True)
@click.option("--secondary", required=True)
@click.option("--mode", type=click.Choice(["coverage", "offset_transplant", "both"]),
              required=True)
@click.option("--onset-tol", default=0.05, show_default=True)
@click.option("--cov-overlap", default=0.5, show_default=True,
              help="keep a secondary note iff <this fraction already covered by primary")
@click.option("--out", required=True)
def main(primary, secondary, mode, onset_tol, cov_overlap, out):
    P = json.loads(Path(primary).read_text())
    S = json.loads(Path(secondary).read_text())
    result = {}
    for sid, prim in P.items():
        sec = S.get(sid, [])
        if mode == "coverage":
            result[sid] = coverage_union(prim, sec, cov_overlap)
        elif mode == "offset_transplant":
            result[sid] = offset_transplant(prim, sec, onset_tol)
        else:  # both
            t = offset_transplant(prim, sec, onset_tol)
            result[sid] = coverage_union(t, sec, cov_overlap)
    Path(out).write_text(json.dumps(result))
    tot = sum(len(v) for v in result.values())
    print(f"[union:{mode}] {len(result)} songs, {tot} notes -> {out}")


if __name__ == "__main__":
    main()
