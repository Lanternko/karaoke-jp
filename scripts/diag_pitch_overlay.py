#!/usr/bin/env python3
"""Visual + textual QA for the consensus-veto octave fix.

Re-derives the consensus shift with the *production* helpers (unguarded by
default — the canonical path; pass --span-guard to reinstate the long-char
guard), then:
- prints a per-shift report (old->new, RMVPE/pYIN diff, stable vs transition, char)
- prints a per-residual report (notes still octave-off vs RMVPE in stable interior,
  with the reason they were left alone: not-octave-over-full-span / vetoed / guarded)
- writes overview.png (stacked 30 s stripes, both F0 contours underlaid)
- writes shifts_gallery.png and residual_gallery.png (per-event zoom panels)

Usage:
    python scripts/diag_pitch_overlay.py --song chidori \
        --current outputs/chidori/<current>.mid \
        --candidate tmp/canonical-octavefix/chidori.octavefix.mid \
        --rmvpe-f0 tmp/chidori_rmvpe_f0.npz --pyin-f0 tmp/chidori_pyin_f0.npz \
        --aligned outputs/chidori/<aligned>.json \
        --out-dir tmp/pitch-ablation/chidori/diag
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from karaoke_jp.pitch_eval import (  # noqa: E402
    F0Track,
    _window_pitch_span,
    lyric_char_windows,
    median_f0_diff_for_note,
    octave_like_mask,
    shift_octave_notes_by_f0_consensus,
    stable_char_windows,
)
from karaoke_jp.score_melody import MidiNote, read_midi_notes  # noqa: E402

OCT_TOL = 120.0
VETO_CLOSE = 250.0
SPAN_GUARD_MIN_DUR = 0.45


def _is_octave(diff_cents: float | None) -> bool:
    if diff_cents is None:
        return False
    return bool(octave_like_mask(np.array([diff_cents]), tolerance_cents=OCT_TOL)[0])


def _char_label(note: MidiNote, char_windows) -> str:
    best, best_ov = "", 0.0
    for w in char_windows:
        ov = min(note.end, w.end) - max(note.start, w.start)
        if ov > best_ov:
            best_ov, best = ov, w.label
    return best


def _stable_median_diff(note: MidiNote, f0: F0Track, stable_windows) -> float | None:
    """Median (f0 - note) cents over stable-window frames inside the note."""
    midi = f0.midi
    idx = (f0.times >= note.start) & (f0.times < note.end) & np.isfinite(midi)
    smask = np.zeros(f0.times.shape, dtype=bool)
    for w in stable_windows:
        smask |= (f0.times >= w.start) & (f0.times < w.end)
    idx &= smask
    if not np.any(idx):
        return None
    return float(np.median((midi[idx] - float(note.pitch)) * 100.0))


def _in_windows(t: float, windows) -> bool:
    return any(w.start <= t < w.end for w in windows)


def _draw_f0(ax, rmvpe: F0Track, pyin: F0Track | None, t0: float, t1: float, lo: int, hi: int):
    for track, color, label in ((rmvpe, "#888888", "RMVPE"), (pyin, "#e8820c", "pYIN")):
        if track is None:
            continue
        m = (track.times >= t0) & (track.times < t1) & np.isfinite(track.midi)
        ax.scatter(track.times[m], track.midi[m], s=4, c=color, alpha=0.55, linewidths=0, label=label)


def _draw_note(ax, note: MidiNote, color: str, *, fill: bool, hatch=None):
    ax.add_patch(
        plt.Rectangle(
            (note.start, note.pitch - 0.4),
            max(note.end - note.start, 0.03),
            0.8,
            facecolor=color if fill else "none",
            edgecolor=color,
            linewidth=1.4,
            alpha=0.85 if fill else 1.0,
            hatch=hatch,
        )
    )


def main_impl(song, current, candidate, rmvpe_f0, pyin_f0, aligned, out_dir, span_guard=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    current_notes = read_midi_notes(current)
    candidate_notes = read_midi_notes(candidate)
    rmvpe = F0Track.from_npz(rmvpe_f0)
    pyin = F0Track.from_npz(pyin_f0) if pyin_f0 else None

    aligned_data = json.loads(Path(aligned).read_text(encoding="utf-8"))
    char_windows = lyric_char_windows(aligned_data)
    stable_windows = stable_char_windows(char_windows)

    # Re-derive the shift (index-aligned with current_notes).  Default is the
    # canonical *unguarded* path; --span-guard reinstates the long-char guard.
    guard_windows = char_windows if span_guard else None
    guarded_notes, n_changes = shift_octave_notes_by_f0_consensus(
        current_notes, primary=rmvpe, veto=pyin, span_guard_windows=guard_windows
    )
    assert len(guarded_notes) == len(current_notes)

    # ---- per-shift report -------------------------------------------------
    shifts = []
    for i, (cur, new) in enumerate(zip(current_notes, guarded_notes, strict=True)):
        if cur.pitch == new.pitch:
            continue
        mid_t = 0.5 * (cur.start + cur.end)
        shifts.append(
            {
                "i": i,
                "t": round(cur.start, 3),
                "dur": round(cur.end - cur.start, 3),
                "old": cur.pitch,
                "new": new.pitch,
                "rmvpe_diff_c": round(median_f0_diff_for_note(cur, rmvpe) or 0.0, 1),
                "pyin_diff_c": (
                    None if pyin is None or median_f0_diff_for_note(cur, pyin) is None
                    else round(median_f0_diff_for_note(cur, pyin), 1)
                ),
                "region": "stable" if _in_windows(mid_t, stable_windows) else "transition/edge",
                "char": _char_label(cur, char_windows),
            }
        )

    # ---- residual octave report (candidate still octave-off vs RMVPE) -----
    residuals = []
    for note in candidate_notes:
        sdiff = _stable_median_diff(note, rmvpe, stable_windows)
        if not _is_octave(sdiff):
            continue
        full_diff = median_f0_diff_for_note(note, rmvpe)
        pyin_full = median_f0_diff_for_note(note, pyin) if pyin is not None else None
        if not _is_octave(full_diff):
            reason = "not-octave-over-full-span (onset/transition glide pulls median)"
        elif pyin_full is not None and abs(pyin_full) <= VETO_CLOSE:
            reason = f"vetoed-by-pyin (pyin diff {pyin_full:.0f}c keeps current)"
        elif span_guard:
            # span guard check
            direction = 12 if full_diff > 0 else -12
            cand_pitch = note.pitch + direction
            guarded = False
            for w in char_windows:
                if w.duration < SPAN_GUARD_MIN_DUR:
                    continue
                if min(note.end, w.end) - max(note.start, w.start) <= 0.0:
                    continue
                before = _window_pitch_span(candidate_notes, w)
                # approximate: candidate index lookup
                try:
                    ridx = candidate_notes.index(note)
                except ValueError:
                    ridx = None
                after = _window_pitch_span(candidate_notes, w, replace_index=ridx, replacement_pitch=cand_pitch)
                if after > before:
                    guarded = True
                    break
            reason = "blocked-by-span-guard" if guarded else "octave-in-stable-only (sub-window)"
        else:
            reason = "octave-in-stable-only (sub-window; full-span median octave-like but unshifted)"
        residuals.append(
            {
                "t": round(note.start, 3),
                "dur": round(note.end - note.start, 3),
                "pitch": note.pitch,
                "rmvpe_stable_c": round(sdiff, 1),
                "rmvpe_full_c": None if full_diff is None else round(full_diff, 1),
                "pyin_full_c": None if pyin_full is None else round(pyin_full, 1),
                "char": _char_label(note, char_windows),
                "reason": reason,
            }
        )

    report = {
        "song": song,
        "n_current": len(current_notes),
        "n_candidate": len(candidate_notes),
        "n_shifts": len(shifts),
        "n_residual_stable_octave": len(residuals),
        "shifts": shifts,
        "residuals": residuals,
    }
    (out / "diag.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n========== {song} ==========")
    print(f"current notes: {len(current_notes)}  candidate notes: {len(candidate_notes)}  shifts: {len(shifts)}")
    print(f"\n--- {len(shifts)} octave shifts (old->new) ---")
    print(f"{'t(s)':>7} {'dur':>5} {'char':<3} {'old->new':>9} {'rmvpeΔc':>8} {'pyinΔc':>8}  region")
    for s in shifts:
        py = "  none" if s["pyin_diff_c"] is None else f"{s['pyin_diff_c']:>8}"
        print(f"{s['t']:>7} {s['dur']:>5} {s['char']:<3} {s['old']:>4}->{s['new']:<4} {s['rmvpe_diff_c']:>8} {py}  {s['region']}")

    print(f"\n--- {len(residuals)} residual stable-octave notes (kept) ---")
    if residuals:
        print(f"{'t(s)':>7} {'dur':>5} {'char':<3} {'pitch':>5} {'stbΔc':>7} {'fullΔc':>7} {'pyinΔc':>7}  reason")
        for r in residuals:
            fc = "   none" if r["rmvpe_full_c"] is None else f"{r['rmvpe_full_c']:>7}"
            pc = "   none" if r["pyin_full_c"] is None else f"{r['pyin_full_c']:>7}"
            print(f"{r['t']:>7} {r['dur']:>5} {r['char']:<3} {r['pitch']:>5} {r['rmvpe_stable_c']:>7} {fc} {pc}  {r['reason']}")

    # ---- overview: stacked 30 s stripes -----------------------------------
    all_t = [n.end for n in current_notes + candidate_notes]
    dur = max(all_t) if all_t else 1.0
    all_p = [n.pitch for n in current_notes + candidate_notes]
    lo, hi = min(all_p) - 3, max(all_p) + 3
    stripe = 30.0
    nrows = max(1, math.ceil(dur / stripe))
    fig, axes = plt.subplots(nrows, 1, figsize=(22, 2.4 * nrows), squeeze=False)
    resid_t = {r["t"] for r in residuals}
    for r in range(nrows):
        ax = axes[r][0]
        t0, t1 = r * stripe, (r + 1) * stripe
        _draw_f0(ax, rmvpe, pyin, t0, t1, lo, hi)
        for cur, new in zip(current_notes, guarded_notes, strict=True):
            if cur.end < t0 or cur.start > t1:
                continue
            if cur.pitch != new.pitch:
                _draw_note(ax, cur, "#bbbbbb", fill=False)  # ghost of old
                _draw_note(ax, new, "#d62728", fill=True)    # shifted (red)
            else:
                _draw_note(ax, cur, "#2a8a2a", fill=True)    # unchanged (green)
        for note in candidate_notes:
            if round(note.start, 3) in resid_t:
                ax.plot(0.5 * (note.start + note.end), note.pitch + 1.2, marker="v", color="magenta", ms=7)
        ax.set_xlim(t0, t1)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("MIDI")
        ax.grid(True, alpha=0.15)
        if r == 0:
            ax.set_title(f"{song}: green=unchanged  red=shifted (gray ghost=old)  ▼magenta=residual stable-octave  · RMVPE gray / pYIN orange")
    axes[-1][0].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=110)
    plt.close(fig)
    print(f"\nwrote {out / 'overview.png'}")

    # ---- galleries --------------------------------------------------------
    def gallery(events, kind):
        if not events:
            return
        ncol = 5
        nrow = math.ceil(len(events) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
        for k, ev in enumerate(events):
            ax = axes[k // ncol][k % ncol]
            t = ev["t"]
            dur = ev.get("dur", 0.3)
            t0, t1 = t - 1.0, t + dur + 1.0
            seg_p = [n.pitch for n in current_notes + candidate_notes if n.end > t0 and n.start < t1]
            plo, phi = (min(seg_p) - 3, max(seg_p) + 3) if seg_p else (50, 70)
            _draw_f0(ax, rmvpe, pyin, t0, t1, plo, phi)
            for cur, new in zip(current_notes, guarded_notes, strict=True):
                if cur.end < t0 or cur.start > t1:
                    continue
                if cur.pitch != new.pitch:
                    _draw_note(ax, cur, "#bbbbbb", fill=False)
                    _draw_note(ax, new, "#d62728", fill=True)
                else:
                    _draw_note(ax, cur, "#2a8a2a", fill=False)
            ax.set_xlim(t0, t1)
            ax.set_ylim(plo, phi)
            ax.axvspan(t, t + dur, color="yellow", alpha=0.12)
            if kind == "shift":
                ax.set_title(f"{ev['char']} {ev['old']}->{ev['new']} r{ev['rmvpe_diff_c']:.0f} p{ev['pyin_diff_c']}", fontsize=9)
            else:
                ax.set_title(f"{ev['char']} p{ev['pitch']} {ev['reason'][:14]}", fontsize=8)
            ax.tick_params(labelsize=7)
        for k in range(len(events), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle(f"{song} — {kind} gallery ({len(events)})  green outline=note  red=shifted  RMVPE gray / pYIN orange", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        path = out / f"{kind}_gallery.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        print(f"wrote {path}")

    gallery(shifts, "shift")
    gallery(residuals, "residual")
    return report


@click.command()
@click.option("--song", required=True)
@click.option("--current", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--candidate", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--rmvpe-f0", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--pyin-f0", type=click.Path(exists=True, dir_okay=False))
@click.option("--aligned", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out-dir", required=True)
@click.option("--span-guard", is_flag=True, default=False,
              help="Reinstate the long-char span guard (default off = canonical).")
def main(song, current, candidate, rmvpe_f0, pyin_f0, aligned, out_dir, span_guard):
    main_impl(song, current, candidate, rmvpe_f0, pyin_f0, aligned, out_dir, span_guard=span_guard)


if __name__ == "__main__":
    main()
