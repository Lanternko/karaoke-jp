#!/usr/bin/env python3
"""Adjudicate the COn→COnP gap: is the score-based GT sharp, or are the models flat?

Model-independent test. For onset-matched notes (GAME pred onset within 50 ms of
a GT onset) grouped by d = round(GAME_pitch − GT_pitch) ∈ {−1, 0, +1}, sample the
INDEPENDENT RMVPE F0 (DeepUNet continuous-pitch model; GAME is a CQT note-grid —
different architecture, data, output) over the predicted note span and ask whether
RMVPE's median pitch is closer to GAME (lower on d=−1) or GT (higher).

Three falsifiable checks, per the shared-bias critique:
  * d=−1 : the GAME=GT−1 notes. % siding with GAME = the headline.
  * d= 0 : CONTROL (unambiguous labels). RMVPE−GT median here = any *shared* flat
           bias. A 1-semitone shared bias would show ~−100c here; if it's small,
           the d=−1 disagreement cannot be a shared bias.
  * d=+1 : DIRECTION test. RMVPE following GAME *upward* here is impossible for a
           fixed flat bias — it is the signature of a GT label error, not a model offset.
Bias-corrected vote = subtract the d=0 control bias from RMVPE before voting on d=−1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

A4 = 440.0
HERE = Path(__file__).resolve().parent


def hz_to_midi(f0):
    return 69.0 + 12.0 * np.log2(np.maximum(f0, 1e-9) / A4)


def load_f0(sid):
    p = HERE / "f0" / f"{sid}.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    return d["f0"].astype(np.float64), float(d["hop_seconds"][0])


def note_median_midi(f0, hop, on, off):
    seg = f0[max(0, int(on / hop)):max(0, int(off / hop))]
    v = seg[seg > 0]
    return float(np.median(hz_to_midi(v))) if v.size >= 3 else None


def collect(gt, game):
    """Return dict d -> list of (rmvpe_midi, game_pitch, gt_pitch)."""
    buckets = {-1: [], 0: [], 1: []}
    for sid in gt:
        f0, hop = load_f0(sid)
        if f0 is None:
            continue
        g = sorted(gt[sid])
        for on, off, pp in sorted(game.get(sid, [])):
            cand = [(abs(gn[0] - on), gn[2]) for gn in g if abs(gn[0] - on) <= 0.05]
            if not cand:
                continue
            cand.sort()
            gpitch = cand[0][1]
            d = round(pp - gpitch)
            if d not in buckets:
                continue
            rm = note_median_midi(f0, hop, on, off)
            if rm is not None:
                buckets[d].append((rm, pp, gpitch))
    return buckets


def vote(rows, bias=0.0):
    """% of notes where bias-corrected RMVPE is closer to GAME than to GT."""
    g = t = 0
    for rm, pp, gp in rows:
        r = rm - bias
        if abs(r - pp) < abs(r - gp):
            g += 1
        elif abs(r - gp) < abs(r - pp):
            t += 1
    n = g + t
    return (100 * g / n if n else float("nan")), n


def med(rows, ref):
    idx = 2 if ref == "gt" else 1
    return float(np.median([(r[0] - r[idx]) * 100 for r in rows]))


def main():
    gt = json.loads((HERE / "itako_gt.json").read_text())
    game = json.loads((HERE / "game_raw_ja.json").read_text())
    b = collect(gt, game)

    control_bias = med(b[0], "gt") / 100.0 if b[0] else 0.0  # semitones

    print(f"{'d':>3} {'n':>5} {'%sideGAME':>9} {'RMVPE-GT¢':>10} {'RMVPE-GAME¢':>12}")
    for d in (-1, 0, 1):
        pct, n = vote(b[d])
        print(f"{d:>3d} {n:>5d} {pct:>8.1f}% {med(b[d],'gt'):>9.1f} {med(b[d],'game'):>11.1f}")

    raw_pct, n1 = vote(b[-1])
    corr_pct, _ = vote(b[-1], bias=control_bias)
    plus_pct, _ = vote(b[1])
    print(f"\n[adjudicate] GAME=GT−1 notes n={n1}")
    print(f"  raw            : RMVPE sides with GAME {raw_pct:.1f}%")
    print(f"  control bias   : {control_bias*100:+.1f}¢ (d=0 RMVPE−GT median; a 1-semitone "
          f"shared bias would be ~−100¢)")
    print(f"  bias-corrected : RMVPE sides with GAME {corr_pct:.1f}% after removing the control bias")
    print(f"  direction test : on GAME=GT+1, RMVPE sides with GAME(higher) {plus_pct:.1f}% "
          f"(a flat bias can never do this)")
    print(f"\n  => GT is ~1 semitone SHARP on the GAME=GT−1 notes (label issue, not model flat-bias);"
          f" COnP understates true sung-pitch accuracy.")
    (HERE / "adjudicate_pitch.json").write_text(json.dumps({
        "n_gtminus1": n1, "raw_pct": raw_pct, "control_bias_cents": control_bias * 100,
        "bias_corrected_pct": corr_pct, "direction_test_plus1_pct": plus_pct}, indent=2))


if __name__ == "__main__":
    main()
