#!/usr/bin/env python3
"""Alignment audit — is UltraSinger's low COn a harness/parse misalignment, or real?

Runs four checks on BOTH the vocadito (English + others) and Kiritan predictions:
  1. harness identity: est=GT must give COn/COnP/COnPOff = 1.0 (and +40ms stays
     1.0, +60ms drops) — proves the matcher isn't deflating scores;
  2. global-offset sweep: where does COn peak? Δ≈0 = aligned; large Δ = a bug;
  3. per-clip best-offset consistency + drift slope: flat = constant latency,
     sloped = BPM/scale error;
  4. precision/recall decomposition at Δ=0 and Δ=best.

Verdict from the run committed alongside: harness identity = 1.0 on both DBs;
beat->sec formula equals UltraSinger's own converter; drift ~0.04 ms/s; COn
peaks at Δ=-0.04s (a real ~40ms latency), correcting it lifts English .492->.532
and Kiritan .304->.339 — small, conclusion unchanged. The score is real
(precision .60 / recall .49): UltraSinger misses ~half the onsets, not misaligned.

  ~/venvs/karaoke-jp/bin/python alignment_audit.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kiritan"))
from conpoff_l import load_notes, match_notes, f1  # noqa: E402


def shift(notes, d):
    return [[s + d, e + d, p] for s, e, p in notes]


def con(ref, est):
    return f1(len(match_notes(ref, est, pitch=False, offset=False)), len(ref), len(est))


def audit(name, gt, pred, ids):
    print(f"\n########## {name}  (n={len(ids)}) ##########")

    idn = [tuple(f1(len(match_notes(gt[t], gt[t], pitch=p, offset=o)), len(gt[t]), len(gt[t]))
                 for p, o in [(False, False), (True, False), (True, True)]) for t in ids]
    print("1) identity est=GT  COn/COnP/COnPOff =",
          tuple(round(statistics.mean(x[i] for x in idn), 3) for i in range(3)), "(expect 1/1/1)")
    print("   GT+40ms COn =", round(statistics.mean(con(gt[t], shift(gt[t], .04)) for t in ids), 3),
          "| GT+60ms COn =", round(statistics.mean(con(gt[t], shift(gt[t], .06)) for t in ids), 3))

    def macro_con(d):
        return statistics.mean(con(gt[t], shift(pred[t], d)) for t in ids)
    best = max(((macro_con(k / 100), k / 100) for k in range(-40, 41, 2)))
    print(f"2) offset sweep: COn(Δ=0)={macro_con(0):.3f}  best={best[0]:.3f} @Δ={best[1]:+.2f}s")

    offs = []
    for t in ids:
        b = max(((con(gt[t], shift(pred[t], k / 100)), k / 100) for k in range(-50, 51, 2)))
        offs.append(b[1])
    pairs = []
    for t in ids:
        used = set()
        for rs, _, _ in gt[t]:
            cand = [(abs(es - rs), j) for j, (es, _, _) in enumerate(pred[t])
                    if j not in used and abs(es - rs) < 0.12]
            if cand:
                _, j = min(cand)
                used.add(j)
                pairs.append((rs, (pred[t][j][0] - rs) * 1000))
    mx = statistics.mean(x for x, _ in pairs)
    slope = (sum((x - mx) * (y - statistics.mean(v for _, v in pairs)) for x, y in pairs)
             / sum((x - mx) ** 2 for x, _ in pairs))
    print(f"3) per-clip bestΔ median={statistics.median(offs):+.3f}s stdev={statistics.pstdev(offs):.3f}s"
          f" | drift slope={slope:+.2f} ms/s (≈0 ⇒ no scale bug)")

    for d in (0.0, best[1]):
        P = statistics.mean(len(match_notes(gt[t], shift(pred[t], d), pitch=False, offset=False)) / len(pred[t])
                            for t in ids if pred[t])
        R = statistics.mean(len(match_notes(gt[t], shift(pred[t], d), pitch=False, offset=False)) / len(gt[t])
                            for t in ids if gt[t])
        print(f"4) Δ={d:+.2f}s  precision={P:.3f} recall={R:.3f}")


def main() -> None:
    lang = json.loads((HERE / "clip_lang.json").read_text())
    vg = load_notes(HERE / "gt_notesA1.json")
    vp = load_notes(HERE / "vocadito_pred.json")
    eng = [t for t in vp if lang[t]["language"] == "English" and vg.get(t)]
    audit("vocadito English (A1)", vg, vp, eng)

    kd = HERE.parent / "kiritan"
    kg = load_notes(kd / "gt_timefix.json")
    kp = load_notes(kd / "ultrasinger/ultrasinger_pred.json")
    audit("Kiritan JA", kg, kp, [s for s in kg if kp.get(s)])


if __name__ == "__main__":
    main()
