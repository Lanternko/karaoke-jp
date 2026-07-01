#!/usr/bin/env python3
"""Per-song TIME-shift audit for Itako (defect class #2, after transposition).

Same model-independent spirit as the Kiritan time-shift audit: some songs have
the whole note GT shifted in time vs the recording. We localize the shift by
cross-correlating the GT note-activity envelope against the RMVPE voicing
envelope (f0 > 0) — neither depends on any transcription model.

Per song:
  * activity[t] = 1 where any GT note is sounding (at the F0 hop grid);
  * voicing[t]  = 1 where RMVPE F0 > 0;
  * lag* = argmax_lag correlation(voicing[t], activity[t - lag]) over +-MAX_LAG;
  * if |lag*| >= MIN_SHIFT and the peak is decisive, shift the song's GT note
    times by +lag* so the labels line up with what was sung.

Input GT should already be transposition-corrected (gt_transcorr.json). Output:
gt_timefix.json (transposition + time corrected).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MAX_LAG_S = 0.6      # search +-600 ms
MIN_SHIFT_S = 0.06   # only correct shifts of at least 60 ms
PEAK_RATIO = 1.08    # peak must beat the zero-lag score by >=8% to act


def envelope(times, hop, n, intervals):
    env = np.zeros(n, dtype=np.float32)
    for on, off in intervals:
        i0, i1 = int(round(on / hop)), int(round(off / hop))
        env[max(0, i0):min(n, max(0, i1))] = 1.0
    return env


def best_lag(voicing, activity, hop):
    max_lag = int(round(MAX_LAG_S / hop))
    lags = range(-max_lag, max_lag + 1)
    # de-mean for a correlation that rewards alignment of the *patterns*
    v = voicing - voicing.mean()
    a = activity - activity.mean()
    scores = {}
    for L in lags:
        shifted = np.roll(a, L)
        if L > 0:
            shifted[:L] = 0
        elif L < 0:
            shifted[L:] = 0
        scores[L] = float(np.dot(v, shifted))
    best = max(scores, key=scores.get)
    zero = scores.get(0, 0.0)
    peak = scores[best]
    decisive = peak > 0 and (zero <= 0 or peak >= PEAK_RATIO * zero)
    return best, best * hop, decisive, peak, zero


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="gt_transcorr.json")
    ap.add_argument("--f0-dir", default="f0")
    ap.add_argument("--out", default="gt_timefix.json")
    args = ap.parse_args()

    gt = json.loads(Path(args.gt).read_text())
    f0_dir = Path(args.f0_dir)
    out = {}
    rows = []
    for sid in sorted(gt):
        notes = gt[sid]
        npz = f0_dir / f"{sid}.npz"
        if not npz.exists() or not notes:
            out[sid] = notes
            rows.append((sid, None, "", "no-f0"))
            continue
        d = np.load(npz)
        f0, hop = d["f0"].astype(np.float64), float(d["hop_seconds"][0])
        n = len(f0)
        voicing = (f0 > 0).astype(np.float32)
        activity = envelope(None, hop, n, [(o, f) for o, f, _ in notes])
        L, lag_s, decisive, peak, zero = best_lag(voicing, activity, hop)
        flag = ""
        if decisive and abs(lag_s) >= MIN_SHIFT_S:
            out[sid] = [[o + lag_s, f + lag_s, p] for o, f, p in notes]
            flag = "SHIFT"
        else:
            out[sid] = notes
        rows.append((sid, round(lag_s * 1000), flag,
                     f"peak/zero={peak:.0f}/{zero:.0f}"))

    Path(args.out).write_text(json.dumps(out))

    print(f"{'song':9} {'lag_ms':>7}  flag   detail")
    n_shift = 0
    for sid, lag_ms, flag, detail in rows:
        if lag_ms is None:
            print(f"{sid:9} {'--':>7}  {flag}")
            continue
        if flag == "SHIFT":
            n_shift += 1
        print(f"{sid:9} {lag_ms:>7d}  {flag:5}  {detail}")
    print(f"\n[timefix] {n_shift}/{len(gt)} songs time-shifted "
          f"(|lag|>={int(MIN_SHIFT_S*1000)}ms, decisive peak) -> {args.out}")


if __name__ == "__main__":
    main()
