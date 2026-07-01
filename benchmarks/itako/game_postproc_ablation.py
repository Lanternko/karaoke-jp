#!/usr/bin/env python3
"""GAME note post-processing ablation on Itako (no GPU; operates on the pred JSON).

Mirrors the Kiritan note-cleanup ablation:
  * min-dur absorb : notes shorter than MIN_DUR fold into the previous note's tail
  * merge same-pitch: consecutive equal-pitch notes within MAX_GAP are merged
Writes the post-processed prediction variants and evaluates each against a GT
with the official MIR-ST500 evaluator (imported, not shelled out).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path("/home/kojiek/side_projects/music-ai/karaoke-jp")
EVAL_DIR = BASE / "benchmarks/singing_transcription_ICASSP2021/evaluate"
sys.path.insert(0, str(EVAL_DIR))

MIN_DUR = 0.10
MERGE_GAP = 0.025


def min_dur_absorb(notes, min_dur=MIN_DUR):
    out = []
    for on, off, p in notes:
        if off - on < min_dur and out:
            out[-1][1] = max(out[-1][1], off)  # extend previous tail
        else:
            out.append([on, off, p])
    return out


def merge_same_pitch(notes, max_gap=MERGE_GAP):
    out = []
    for on, off, p in sorted(notes):
        if out and p == out[-1][2] and on - out[-1][1] <= max_gap:
            out[-1][1] = max(out[-1][1], off)
        else:
            out.append([on, off, p])
    return out


def apply(pred, fn):
    return {sid: fn(notes) for sid, notes in pred.items()}


def evaluate(gt, pred):
    import evaluate as ev  # official evaluator module
    ids = sorted(set(gt) & set(pred))
    at = [gt[i] for i in ids]
    ap = [pred[i] for i in ids]
    # eval_all returns averaged [COnPOff_P,R,F, COnP_P,R,F, COn_P,R,F, gtnum, trnum]
    avg = ev.eval_all(at, ap, onset_tolerance=0.05, print_result=False, id_list=ids)
    return {"COn": avg[8], "COnP": avg[5], "COnPOff": avg[2],
            "gt": avg[9], "tr": avg[10]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="game_raw_ja.json")
    ap.add_argument("--gt", default="itako_gt.json")
    ap.add_argument("--tag", default="game")
    args = ap.parse_args()

    pred = json.loads(Path(args.pred).read_text())
    gt = json.loads(Path(args.gt).read_text())

    variants = {
        "raw": pred,
        "mindur": apply(pred, min_dur_absorb),
        "merge+mindur": apply(apply(pred, merge_same_pitch), min_dur_absorb),
    }
    rows = {}
    for name, p in variants.items():
        n = sum(len(v) for v in p.values())
        m = evaluate(gt, p)
        rows[name] = {**m, "notes": n}
        if name != "raw":
            Path(f"{args.tag}_{name.replace('+','_')}.json").write_text(json.dumps(p))

    print(f"{'variant':14} {'COn':>7} {'COnP':>7} {'COnPOff':>8} {'notes':>7}")
    for name, r in rows.items():
        print(f"{name:14} {r['COn']:.4f}  {r['COnP']:.4f}  {r['COnPOff']:.4f}  {r['notes']:>7d}")
    Path(f"{args.tag}_ablation.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
