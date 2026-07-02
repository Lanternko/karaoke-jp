#!/usr/bin/env python3
"""Language-confound control: UltraSinger COn/COnP/COnPOff on vocadito
(English + 6 other languages, solo vocals, note-level GT).

The question this settles: UltraSinger has NO pitch-onset note detector — every
note boundary is a whisperx syllable boundary (midi_creator.py:153). So the
Kiritan COn = .304 collapse could be Japanese-ASR-driven rather than intrinsic.
If English COn here jumps well above .304, the story is "non-English ASR
segmentation bottleneck"; if it stays ~.30, "no onset detector => weak regardless
of language."

Reuses the exact matching harness (conpoff_l.match_notes / f1), same tolerances
(onset 50 ms, pitch 50 cents, offset max(50 ms, 0.2*dur)) as every Kiritan row.
No L axis (no aligner for English; not the question). Two annotators (A1 primary,
A2 reported — vocadito's inter-annotator agreement is known to be low).

  ~/venvs/karaoke-jp/bin/python vocadito_eval.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kiritan"))
from conpoff_l import load_notes, match_notes, f1  # noqa: E402

KIRITAN_CON = 0.304  # UltraSinger x Kiritan JA a cappella, reference point


def eval_clip(ref, est):
    m_on = match_notes(ref, est, pitch=False, offset=False)
    m_onp = match_notes(ref, est, offset=False)
    m_onpoff = match_notes(ref, est)
    return {
        "COn": f1(len(m_on), len(ref), len(est)),
        "COnP": f1(len(m_onp), len(ref), len(est)),
        "COnPOff": f1(len(m_onpoff), len(ref), len(est)),
        "n_ref": len(ref), "n_est": len(est),
    }


def macro(ps, k):
    return statistics.mean(p[k] for p in ps) if ps else 0.0


def boot_ci(ps, k, iters=2000, seed=7):
    rng = random.Random(seed)
    n = len(ps)
    if n == 0:
        return (0.0, 0.0, 0.0)
    vals = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(statistics.mean(ps[i][k] for i in idx))
    vals.sort()
    return (statistics.mean(vals), vals[int(0.025 * iters)], vals[int(0.975 * iters)])


def run(gt_path: Path, tag: str, pred, lang):
    gt = load_notes(gt_path)
    per = {}
    for tid in pred:
        if tid not in gt or not gt[tid] or not pred[tid]:
            continue
        per[tid] = eval_clip(gt[tid], pred[tid])

    # by language group
    groups = {}
    for tid, r in per.items():
        groups.setdefault(lang[tid]["language"], []).append(r)

    print(f"\n=== vocadito UltraSinger — {tag} (COn/COnP/COnPOff, macro F1) ===")
    print(f"{'group':22s} {'n':>3s} {'COn':>6s} {'COnP':>6s} {'COnPOff':>8s} "
          f"{'est/ref':>8s}")
    allps = list(per.values())
    def line(name, ps):
        er = statistics.mean(p["n_est"] / p["n_ref"] for p in ps) if ps else 0
        print(f"{name:22s} {len(ps):>3d} {macro(ps,'COn'):>6.3f} "
              f"{macro(ps,'COnP'):>6.3f} {macro(ps,'COnPOff'):>8.3f} {er:>8.2f}")
    line("ALL", allps)
    for g in sorted(groups, key=lambda x: -len(groups[x])):
        line(g, groups[g])

    eng = groups.get("English", [])
    m, lo, hi = boot_ci(eng, "COn")
    print(f"\nEnglish COn = {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n={len(eng)})")
    print(f"Kiritan JA COn (reference) = {KIRITAN_CON:.3f}")
    verdict = ("ASR-SEGMENTATION BOTTLENECK (English >> Japanese)"
               if lo > KIRITAN_CON else
               "NO CLEAN LANGUAGE EFFECT (English CI overlaps/below Kiritan .304)")
    print(f"VERDICT [{tag}]: {verdict}")
    return {"all": {k: round(macro(allps, k), 4) for k in ("COn", "COnP", "COnPOff")},
            "english_COn": round(m, 4), "english_COn_CI": [round(lo, 4), round(hi, 4)],
            "english_n": len(eng), "n_clips": len(allps),
            "by_language": {g: {"n": len(ps), **{k: round(macro(ps, k), 4)
                                for k in ("COn", "COnP", "COnPOff")}}
                            for g, ps in groups.items()}}


def main() -> None:
    pred = load_notes(HERE / "vocadito_pred.json")
    lang = json.loads((HERE / "clip_lang.json").read_text())
    out = {}
    out["A1"] = run(HERE / "gt_notesA1.json", "annotator A1 (primary)", pred, lang)
    out["A2"] = run(HERE / "gt_notesA2.json", "annotator A2", pred, lang)
    out["kiritan_ja_COn_reference"] = KIRITAN_CON
    (HERE / "vocadito_results.json").write_text(json.dumps(out, indent=2))
    print("\nwritten: vocadito_results.json")


if __name__ == "__main__":
    main()
