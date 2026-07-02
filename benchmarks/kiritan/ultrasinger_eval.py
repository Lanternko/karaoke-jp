#!/usr/bin/env python3
"""Phase 5: COnPOff+L for UltraSinger (first-ever quantitative eval).

Reuses the harness (conpoff_l) verbatim — no logic copied. UltraSinger is a
lyrics-UNKNOWN full-stack system (whisperx hears the lyrics itself); its L tax
therefore folds mis-heard characters AND time-attribution error together, a
strictly harder setting than our lyrics-KNOWN MMS forced-alignment rows. So we
report the fully-fair note axes (COn/COnP/COnPOff) alongside COnPOff+L.

UltraSinger est morae = the per-note (onset, label) list from ultrasinger_morae
.json (each note owns its own syllable's mora — the karaoke "right bar, right
word" reading). GT morae come from mono_label, same as the harness.

Sanity first: re-run GAMExMMS_FA and assert it reproduces conpoff_l_results.json
(0.408) before printing any UltraSinger number.

  ~/venvs/karaoke-jp/bin/python benchmarks/kiritan/ultrasinger_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from conpoff_l import (  # noqa: E402  (reuse, do not reimplement)
    KIRITAN, load_notes, group_morae, read_lab, eval_song, macro,
    bootstrap_diff,
)

USDIR = HERE / "ultrasinger"
SANITY_TARGET = 0.408  # GAMExMMS_FA COnPOff+L from conpoff_l_results.json


def eval_ultrasinger(gt, gt_morae, songs):
    """UltraSinger: notes + per-note (onset,label) morae. eval_song attributes
    each est note back to its own mora via the ownership-interval search."""
    pred = load_notes(USDIR / "ultrasinger_pred.json")
    morae_raw = json.loads((USDIR / "ultrasinger_morae.json").read_text())
    per_song = []
    used = []
    for s in songs:
        if s not in pred or not pred[s]:
            continue  # failed song — excluded, reported separately
        est_morae = [(float(on), lab) for on, lab in morae_raw[s]]
        per_song.append(eval_song(gt[s], pred[s], gt_morae[s], est_morae))
        used.append(s)
    return per_song, used


def eval_baseline(gt, gt_morae, songs, note_path, aligner_dir):
    nm = load_notes(note_path)
    per_song = []
    for s in songs:
        est_morae = group_morae(read_lab(aligner_dir / f"{s}.lab"))
        per_song.append(eval_song(gt[s], nm[s], gt_morae[s], est_morae))
    return per_song


def row(name, ps):
    matched = sum(x["n_matched"] for x in ps)
    lok = sum(x["n_lyric_ok"] for x in ps)
    return (f"{name:18s} {macro(ps,'COn'):>6.3f} {macro(ps,'COnP'):>6.3f} "
            f"{macro(ps,'COnPOff'):>8.3f} {macro(ps,'COnPOff+L'):>10.3f} "
            f"{lok/matched if matched else 0:>10.1%}  (N={len(ps)})")


def main() -> None:
    gt = load_notes(HERE / "gt_timefix.json")
    songs = sorted(gt)
    gt_morae = {s: group_morae(read_lab(KIRITAN / f"mono_label/{s}.lab")) for s in songs}

    # ---- Sanity: GAMExMMS_FA must reproduce 0.408 --------------------------
    game_mms = eval_baseline(gt, gt_morae, songs, HERE / "game_raw_ja.json",
                             HERE / "phone_boundary/mms_fa_htk")
    game_mms_L = round(macro(game_mms, "COnPOff+L"), 3)
    print(f"[sanity] GAMExMMS_FA COnPOff+L = {game_mms_L}  (target {SANITY_TARGET})")
    if abs(game_mms_L - SANITY_TARGET) > 0.002:
        print("!! SANITY FAILED — harness drifted; refusing to report UltraSinger.")
        sys.exit(1)
    print("[sanity] PASS\n")

    # ---- UltraSinger --------------------------------------------------------
    us, used = eval_ultrasinger(gt, gt_morae, songs)
    failed = [s for s in songs if s not in used]

    print("=== COnPOff+L ladder (Kiritan, gt_timefix, macro per-song F1) ===")
    print(f"{'system':18s} {'COn':>6s} {'COnP':>6s} {'COnPOff':>8s} "
          f"{'COnPOff+L':>10s} {'P(L|match)':>10s}")
    print(row("UltraSinger", us))
    print(row("GAMExMMS_FA", game_mms))

    # ---- paired bootstrap vs GAMExMMS_FA on the SAME songs ------------------
    # restrict GAMExMMS_FA to the songs UltraSinger produced, for a paired test.
    game_mms_used = [game_mms[songs.index(s)] for s in used]
    print("\n=== UltraSinger vs GAMExMMS_FA (paired bootstrap, same N songs, "
          "2000 iters, seed 7) ===")
    for key in ("COn", "COnP", "COnPOff", "COnPOff+L"):
        m, lo, hi = bootstrap_diff(us, game_mms_used, key)
        sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "n.s."
        print(f"  {key:10s}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  {sig}")

    if failed:
        print(f"\nFAILED songs ({len(failed)}): {failed}")
    else:
        print("\nno failed songs")

    out = {
        "UltraSinger": {
            "COn": round(macro(us, "COn"), 4),
            "COnP": round(macro(us, "COnP"), 4),
            "COnPOff": round(macro(us, "COnPOff"), 4),
            "COnPOff+L": round(macro(us, "COnPOff+L"), 4),
            "P_L_given_match": round(sum(x["n_lyric_ok"] for x in us)
                                     / max(1, sum(x["n_matched"] for x in us)), 4),
            "N": len(us),
            "failed_songs": failed,
        },
        "sanity_GAMExMMS_FA_COnPOffL": game_mms_L,
    }
    (USDIR / "ultrasinger_results.json").write_text(json.dumps(out, indent=2))
    print("\nwritten: ultrasinger/ultrasinger_results.json")


if __name__ == "__main__":
    main()
