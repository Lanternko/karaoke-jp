#!/usr/bin/env python3
"""No-GPU GT-defect preview: which Itako songs are transposed / time-shifted?

Two model-consensus signals (GAME + CE+CTC are architecturally independent, so
agreement implicates the GT, not a single model — the same logic the RMVPE audit
formalizes with a non-model reference):

  * TRANSPOSITION: among onset-matched notes (pred onset within 50 ms of a GT
    onset), the modal integer (pred_pitch - gt_pitch). If BOTH models agree on a
    non-zero offset for a song, its GT key is likely shifted by that much.
  * TIME-SHIFT: per-song COn. If BOTH models collapse (COn < THRESH) on the same
    song while the corpus average is ~0.80, the note times are likely shifted.

This previews — and later cross-checks — the RMVPE audits (audit_gt_rmvpe.py /
audit_timeshift.py), which are the authoritative, model-independent corrections.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

BASE = Path("/home/kojiek/side_projects/music-ai/karaoke-jp")
sys.path.insert(0, str(BASE / "benchmarks/singing_transcription_ICASSP2021/evaluate"))
import evaluate as ev  # noqa: E402

COLLAPSE = 0.55  # per-song COn below this = collapse candidate


def per_song_con(gt_notes, pred_notes):
    ret = ev.eval_one_data(gt_notes, pred_notes, onset_tolerance=0.05)
    return ret[8]  # COn F1


def modal_pitch_offset(gt_notes, pred_notes, tol=0.05):
    gt = sorted(gt_notes)
    diffs = []
    j = 0
    for on, off, pp in sorted(pred_notes):
        # nearest GT onset
        best = None
        for k in range(len(gt)):
            d = abs(gt[k][0] - on)
            if best is None or d < best[0]:
                best = (d, gt[k][2])
        if best and best[0] <= tol:
            diffs.append(round(pp - best[1]))
    if len(diffs) < 5:
        return None, 0
    c = Counter(diffs)
    return c.most_common(1)[0][0], len(diffs)


def main() -> None:
    gt = json.loads(Path("itako_gt.json").read_text())
    game = json.loads(Path("game_raw_ja.json").read_text())
    cectc = json.loads(Path("ctcce_pred.json").read_text())

    rows = []
    for sid in sorted(gt):
        g = per_song_con(gt[sid], game.get(sid, []))
        c = per_song_con(gt[sid], cectc.get(sid, []))
        go, gn = modal_pitch_offset(gt[sid], game.get(sid, []))
        co, cn = modal_pitch_offset(gt[sid], cectc.get(sid, []))
        rows.append((sid, g, c, go, co))

    transp = []  # both models agree non-zero offset
    timesh = []  # both models collapse COn
    print(f"{'song':9} {'GAME_COn':>9} {'CECTC_COn':>9} {'GAMEoff':>8} {'CECTCoff':>9}  flags")
    for sid, g, c, go, co in rows:
        flags = []
        if go is not None and co is not None and go == co and go != 0:
            flags.append(f"TRANSPOSE{go:+d}")
            transp.append((sid, go))
        if g < COLLAPSE and c < COLLAPSE:
            flags.append("TIMESHIFT?")
            timesh.append(sid)
        go_s = f"{go:+d}" if go is not None else "?"
        co_s = f"{co:+d}" if co is not None else "?"
        print(f"{sid:9} {g:>9.3f} {c:>9.3f} {go_s:>8} {co_s:>9}  {' '.join(flags)}")

    print(f"\n[diag] corpus mean per-song COn: GAME "
          f"{statistics.mean(r[1] for r in rows):.3f}  CE+CTC "
          f"{statistics.mean(r[2] for r in rows):.3f}")
    print(f"[diag] TRANSPOSE candidates (both models agree non-zero, pre-RMVPE): "
          f"{transp}")
    print(f"[diag] TIMESHIFT candidates (both models collapse COn<{COLLAPSE}): {timesh}")
    Path("diagnose_persong.json").write_text(json.dumps(
        {"rows": rows, "transpose": transp, "timeshift": timesh}, default=str, indent=2))


if __name__ == "__main__":
    main()
