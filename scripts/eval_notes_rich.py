#!/usr/bin/env python3
"""Rich, honest note-transcription diagnostics on top of mir_eval.

Why this exists (survey 2026-06-13, weakness D2 + B1):
  * The legacy ``eval_note_metrics.py`` sweeps a per-candidate global shift and
    reports the BEST COnP_F -- i.e. it tunes a parameter on the test metric, so
    scores are systematically optimistic and candidate ranking is distorted.
    Here the default is ``shift=0`` (honest); ``--best-shift`` additionally
    reports the optimism gap so we can SEE how much free shift was buying.
  * COnPOff is a binary pass/fail per note. The offset hole (GAME COnPOff .411
    vs SOTA .625) needs to be *seen*: per-matched-note IOU + signed onset/offset
    bias distributions tell us WHERE the offset is bleeding (early cut vs late).
  * Over/under-segmentation (n_est/n_ref, misses, spurious) exposes coverage vs
    merge/split behaviour that COnP alone hides.

Input formats (auto-detected):
  * MIR-ST500-style JSON: {"song_id": [[onset, offset, midi], ...], ...}
  * single-song JSON list: [[onset, offset, midi], ...]
  * .mid / .midi: read via karaoke_jp.score_melody.read_midi_notes

Usage:
  eval_notes_rich.py --ref GT.json --pred GAME:game_seg03.json --pred CECTC:ctcce.json
  eval_notes_rich.py --ref chidori_gold.mid --pred union:union.mid --onset-tol 0.05
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mir_eval.transcription as mt  # noqa: E402
import mir_eval.util as mu  # noqa: E402


# ----------------------------------------------------------------------------- loaders
def _notes_to_arrays(notes: list) -> tuple[np.ndarray, np.ndarray]:
    """notes = [[on, off, midi], ...] -> (intervals Nx2, hz N), dropping invalids."""
    rows = [
        (float(n[0]), float(n[1]), float(n[2]))
        for n in notes
        if n is not None and float(n[1]) - float(n[0]) > 0.0
    ]
    if not rows:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)
    intervals = np.array([[r[0], r[1]] for r in rows], dtype=float)
    hz = np.array([440.0 * 2 ** ((r[2] - 69.0) / 12.0) for r in rows], dtype=float)
    return intervals, hz


def load_dict(path: str) -> dict[str, list]:
    """Return {song_id: [[on,off,midi],...]}. MIDI files become {'__single__': ...}."""
    p = Path(path)
    if p.suffix.lower() in {".mid", ".midi"}:
        from karaoke_jp.score_melody import read_midi_notes

        notes = [[n.start, n.end, n.pitch] for n in read_midi_notes(p)]
        return {"__single__": notes}
    data = json.loads(p.read_text())
    if isinstance(data, list):
        return {"__single__": data}
    return {str(k): v for k, v in data.items()}


# ----------------------------------------------------------------------------- metrics
def _prf(matched: int, n_ref: int, n_est: int) -> tuple[float, float, float]:
    p = matched / n_est if n_est else 0.0
    r = matched / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def song_scores(
    ref: tuple[np.ndarray, np.ndarray],
    est: tuple[np.ndarray, np.ndarray],
    *,
    onset_tol: float,
    pitch_tol: float,
    offset_ratio: float,
    offset_min: float,
    shift: float,
) -> dict:
    ri, rh = ref
    ei, eh = est
    ei = ei + shift if ei.shape[0] else ei
    n_ref, n_est = ri.shape[0], ei.shape[0]

    out = {"n_ref": n_ref, "n_est": n_est}
    if n_ref == 0 or n_est == 0:
        for k in ("COn_F", "COnP_F", "COnPOff_F"):
            out[k] = 0.0
        out.update(matched_onp=0, iou=[], d_on=[], d_off=[])
        return out

    # COn (onset only)
    m_on = mu.match_events(ri[:, 0], ei[:, 0], onset_tol)
    out["COn_F"] = _prf(len(m_on), n_ref, n_est)[2]

    # COnP (onset + pitch, offset ignored)
    m_onp = mt.match_notes(ri, rh, ei, eh, onset_tolerance=onset_tol,
                           pitch_tolerance=pitch_tol, offset_ratio=None)
    out["COnP_F"] = _prf(len(m_onp), n_ref, n_est)[2]
    out["matched_onp"] = len(m_onp)

    # COnPOff (onset + pitch + offset)
    m_off = mt.match_notes(ri, rh, ei, eh, onset_tolerance=onset_tol,
                           pitch_tolerance=pitch_tol, offset_ratio=offset_ratio,
                           offset_min_tolerance=offset_min)
    out["COnPOff_F"] = _prf(len(m_off), n_ref, n_est)[2]

    # diagnostics on the COnP matches: IOU + signed onset/offset error (est - ref)
    iou, d_on, d_off = [], [], []
    for ri_i, ei_i in m_onp:
        rs, re = ri[ri_i]
        es, ee = ei[ei_i]
        inter = max(0.0, min(re, ee) - max(rs, es))
        union = max(re, ee) - min(rs, es)
        iou.append(inter / union if union > 0 else 0.0)
        d_on.append(es - rs)
        d_off.append(ee - re)
    out.update(iou=iou, d_on=d_on, d_off=d_off)
    return out


def aggregate(per_song: list[dict]) -> dict:
    """Micro (pooled) F-scores via summed match/ref/est + pooled diagnostics."""
    # macro F (per-song mean) and micro F (pooled). Report both.
    def macro(key):
        vals = [s[key] for s in per_song if s["n_ref"] > 0]
        return float(np.mean(vals)) if vals else 0.0

    iou = [x for s in per_song for x in s["iou"]]
    d_on = [x for s in per_song for x in s["d_on"]]
    d_off = [x for s in per_song for x in s["d_off"]]
    tot_ref = sum(s["n_ref"] for s in per_song)
    tot_est = sum(s["n_est"] for s in per_song)
    return {
        "songs": len([s for s in per_song if s["n_ref"] > 0]),
        "COn_F": macro("COn_F"),
        "COnP_F": macro("COnP_F"),
        "COnPOff_F": macro("COnPOff_F"),
        "n_ref": tot_ref,
        "n_est": tot_est,
        "seg_ratio": tot_est / tot_ref if tot_ref else 0.0,
        "iou_median": float(np.median(iou)) if iou else 0.0,
        "iou_mean": float(np.mean(iou)) if iou else 0.0,
        "onset_bias_ms": float(np.mean(d_on)) * 1000 if d_on else 0.0,
        "onset_mae_ms": float(np.mean(np.abs(d_on))) * 1000 if d_on else 0.0,
        "offset_bias_ms": float(np.mean(d_off)) * 1000 if d_off else 0.0,
        "offset_mae_ms": float(np.mean(np.abs(d_off))) * 1000 if d_off else 0.0,
        "offset_early_frac": float(np.mean(np.array(d_off) < 0)) if d_off else 0.0,
    }


def eval_pred(ref_d, est_d, *, onset_tol, pitch_tol, offset_ratio, offset_min, shift):
    keys = [k for k in ref_d if k in est_d]
    per_song = []
    for k in keys:
        per_song.append(song_scores(
            _notes_to_arrays(ref_d[k]), _notes_to_arrays(est_d[k]),
            onset_tol=onset_tol, pitch_tol=pitch_tol,
            offset_ratio=offset_ratio, offset_min=offset_min, shift=shift))
    agg = aggregate(per_song)
    agg["matched_keys"] = len(keys)
    return agg


@click.command()
@click.option("--ref", "ref_path", required=True)
@click.option("--pred", "preds", multiple=True, required=True, help="LABEL:PATH")
@click.option("--onset-tol", default=0.05, show_default=True)
@click.option("--pitch-tol", default=50.0, show_default=True)
@click.option("--offset-ratio", default=0.2, show_default=True)
@click.option("--offset-min", default=0.05, show_default=True)
@click.option("--shift", default=0.0, show_default=True, help="fixed global shift (honest=0)")
@click.option("--best-shift", is_flag=True, help="also report best-shift COnP_F + optimism gap")
@click.option("--max-shift", default=0.06, show_default=True)
@click.option("--shift-step", default=0.01, show_default=True)
@click.option("--json-out", default=None)
def main(ref_path, preds, onset_tol, pitch_tol, offset_ratio, offset_min,
         shift, best_shift, max_shift, shift_step, json_out):
    ref_d = load_dict(ref_path)
    rows = []
    for spec in preds:
        label, path = spec.split(":", 1)
        est_d = load_dict(path)
        agg = eval_pred(ref_d, est_d, onset_tol=onset_tol, pitch_tol=pitch_tol,
                        offset_ratio=offset_ratio, offset_min=offset_min, shift=shift)
        agg["label"] = label
        if best_shift:
            best = None
            for sh in np.arange(-max_shift, max_shift + 1e-9, shift_step):
                a = eval_pred(ref_d, est_d, onset_tol=onset_tol, pitch_tol=pitch_tol,
                              offset_ratio=offset_ratio, offset_min=offset_min, shift=float(sh))
                if best is None or a["COnP_F"] > best[1]:
                    best = (float(sh), a["COnP_F"], a["COnPOff_F"])
            agg["best_shift"] = best[0]
            agg["COnP_F_bestshift"] = best[1]
            agg["COnPOff_F_bestshift"] = best[2]
            agg["optimism_gap_COnP"] = best[1] - agg["COnP_F"]
        rows.append(agg)

    cols = ["label", "songs", "COn_F", "COnP_F", "COnPOff_F", "iou_median",
            "onset_bias_ms", "offset_bias_ms", "offset_early_frac", "seg_ratio", "n_est"]
    if best_shift:
        cols += ["best_shift", "COnP_F_bestshift", "optimism_gap_COnP"]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
        return str(v)
    widths = {c: max(len(c), *(len(fmt(r.get(c, ""))) for r in rows)) for c in cols}
    click.echo("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        click.echo("  ".join(fmt(r.get(c, "")).ljust(widths[c]) for c in cols))

    if json_out:
        Path(json_out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
