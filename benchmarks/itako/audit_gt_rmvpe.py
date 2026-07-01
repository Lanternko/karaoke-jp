#!/usr/bin/env python3
"""RMVPE-based GT audit for Itako (transposition + global-tuning check).

Same model-independent protocol as the Kiritan audit (see ../kiritan/RESULTS.md):
the upstream README warns some songs have ambiguous / shifted keys. We verify the
note GT against an INDEPENDENT F0 tracker (RMVPE), not against any transcription
model, so a disagreement implicates the label, not the model.

Per song:
  * sample RMVPE F0 (Hz) inside each GT note's [onset, offset], voiced frames only;
  * note deviation = GT_pitch - median(F0 -> MIDI) over that note;
  * song offset = round(median note deviation)  -> the per-song semitone transpose;
  * residual cents (after removing the integer offset) = global-tuning sanity.

Songs with offset != 0 have a GT transposed relative to the audio; we shift their
GT pitches by -offset to match what was actually sung. Output: gt_transcorr.json.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

A4 = 440.0


def hz_to_midi(f0: np.ndarray) -> np.ndarray:
    return 69.0 + 12.0 * np.log2(np.maximum(f0, 1e-9) / A4)


def note_f0_midi(f0: np.ndarray, hop: float, on: float, off: float) -> float | None:
    i0, i1 = int(round(on / hop)), int(round(off / hop))
    seg = f0[max(0, i0):max(0, i1)]
    voiced = seg[seg > 0]
    if voiced.size < 3:  # need a few voiced frames to trust it
        return None
    return float(np.median(hz_to_midi(voiced)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="itako_gt.json")
    ap.add_argument("--f0-dir", default="f0")
    ap.add_argument("--out", default="gt_transcorr.json")
    ap.add_argument("--min-abs-offset", type=int, default=1,
                    help="only correct songs whose |offset| >= this")
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text())
    f0_dir = Path(args.f0_dir)

    corrected = {}
    rows = []
    for sid in sorted(gt):
        notes = gt[sid]
        npz = f0_dir / f"{sid}.npz"
        if not npz.exists():
            corrected[sid] = notes
            rows.append((sid, None, None, None, len(notes), "no-f0"))
            continue
        d = np.load(npz)
        f0, hop = d["f0"].astype(np.float64), float(d["hop_seconds"][0])
        devs = []
        for on, off, pitch in notes:
            m = note_f0_midi(f0, hop, on, off)
            if m is not None:
                devs.append(pitch - m)
        if len(devs) < 5:
            corrected[sid] = notes
            rows.append((sid, None, None, None, len(notes), "too-few-voiced"))
            continue
        med = statistics.median(devs)
        offset = int(round(med))
        residual_cents = 100.0 * (med - offset)
        flagged = abs(offset) >= args.min_abs_offset
        if flagged:
            corrected[sid] = [[on, off, pitch - offset] for on, off, pitch in notes]
        else:
            corrected[sid] = notes
        rows.append((sid, offset, round(med, 3), round(residual_cents, 1),
                     len(devs), "SHIFT" if flagged else ""))

    Path(args.out).write_text(json.dumps(corrected))

    # report
    print(f"{'song':9} {'offset':>6} {'median':>7} {'resid¢':>7} {'nvoiced':>7}  flag")
    n_shift = 0
    all_resid = []
    for sid, offset, med, rc, nv, flag in rows:
        if offset is None:
            print(f"{sid:9} {'--':>6} {'--':>7} {'--':>7} {nv:>7}  {flag}")
            continue
        if flag == "SHIFT":
            n_shift += 1
        all_resid.append(rc)
        print(f"{sid:9} {offset:>6d} {med:>7.3f} {rc:>7.1f} {nv:>7d}  {flag}")
    offsets = [r[1] for r in rows if r[1] is not None]
    from collections import Counter
    print(f"\n[audit] {n_shift}/{len(gt)} songs transposed (|offset|>={args.min_abs_offset}); "
          f"offset distribution: {dict(sorted(Counter(offsets).items()))}")
    if all_resid:
        print(f"[audit] global tuning: mean residual {statistics.mean(all_resid):+.1f}¢ "
              f"(near 0 ⇒ in tune to A440 once integer transpose removed)")
    print(f"[audit] wrote {args.out}")


if __name__ == "__main__":
    main()
