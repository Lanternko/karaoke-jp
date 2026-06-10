#!/usr/bin/env python3
"""Score-style post-fixes for a mora-fitted melody MIDI.

Audio-only repairs aimed at "what the sheet would write", applied AFTER the
refit/validator/octavefix chain:

1. --fill-morae        resurrect validator-killed morae when F0 supports them
2. --refine-boundaries move note joints to the actual F0 crossing
3. --absorb-shakuri    fold short rising onset notes into the target note
4. --chroma-prior      semitone snap toward the accompaniment's pitch-class set
5. --extend-sustains   extend note tails while F0 keeps holding the pitch

Every lever uses only audio-side evidence (aligned morae, RMVPE F0,
accompaniment chroma) — never a reference/gold file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import refit_melody_to_mora as rmm  # noqa: E402

from karaoke_jp.melody import _write_midi  # noqa: E402
from karaoke_jp.pitch_eval import F0Track  # noqa: E402
from karaoke_jp.score_melody import read_first_tempo_bpm, read_midi_notes  # noqa: E402

Note = tuple[float, float, int]  # start, end, pitch


def _voiced_midi(track: F0Track, start: float, end: float, edge_trim: float = 0.0):
    trim = min(edge_trim, max(0.0, (end - start) * 0.25))
    mask = (track.times >= start + trim) & (track.times < end - trim)
    midi = track.midi[mask]
    times = track.times[mask]
    keep = np.isfinite(midi)
    return times[keep], midi[keep]


def fill_missing_morae(
    notes: list[Note],
    windows: list[tuple[float, float, str, int]],
    track: F0Track,
    *,
    min_voiced_ratio: float = 0.4,
    min_duration: float = 0.10,
) -> tuple[list[Note], int]:
    """Add a note for aligned morae that have no candidate note at all."""
    hop = track.hop_seconds or 0.01
    added = 0
    out = list(notes)
    for ws, we, _kana, _line in windows:
        if we - ws < min_duration:
            continue
        covered = sum(
            max(0.0, min(we, e) - max(ws, s)) for s, e, _ in out if e > ws and s < we
        )
        if covered >= 0.3 * (we - ws):
            continue
        times, midi = _voiced_midi(track, ws, we, edge_trim=0.04)
        need = min_voiced_ratio * (we - ws) / hop
        if midi.size < max(2.0, need):
            continue
        pitch = int(np.round(np.median(midi)))
        # insert into the largest free sub-interval of [ws, we): an existing
        # note that sits ENTIRELY INSIDE the window splits it in two, so a
        # simple two-sided clip would overlap it.
        overlapping = sorted(
            (max(ws, ns), min(we, ne)) for ns, ne, _ in out if ne > ws and ns < we
        )
        gaps = []
        cursor = ws
        for os_, oe_ in overlapping:
            if os_ > cursor:
                gaps.append((cursor, os_))
            cursor = max(cursor, oe_)
        if cursor < we:
            gaps.append((cursor, we))
        if not gaps:
            continue
        s, e = max(gaps, key=lambda g: g[1] - g[0])
        if e - s >= min_duration:
            out.append((s, e, pitch))
            added += 1
    out.sort()
    return out, added


def refine_boundaries(
    notes: list[Note],
    track: F0Track,
    *,
    search: float = 0.12,
    max_gap: float = 0.08,
) -> tuple[list[Note], int]:
    """Move each unequal-pitch joint to where F0 actually crosses over."""
    out = [list(n) for n in sorted(notes)]
    moved = 0
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        if b[0] - a[1] > max_gap or a[2] == b[2]:
            continue
        joint = b[0]
        lo = max(a[0] + 0.04, joint - search)
        hi = min(b[1] - 0.04, joint + search)
        if hi <= lo:
            continue
        times, midi = _voiced_midi(track, lo, hi)
        if midi.size < 3:
            continue
        closer_b = np.abs(midi - b[2]) < np.abs(midi - a[2])
        # smooth with a 3-frame majority, then take the first switch to b
        kernel = np.convolve(closer_b.astype(float), np.ones(3) / 3, mode="same")
        switched = np.nonzero(kernel >= 0.5)[0]
        if switched.size == 0:
            # F0 never reaches b's side inside the window; leave the joint.
            continue
        # If the window is already on b's side at its FIRST frame, the true
        # crossing happened earlier and t_star becomes the window edge (an
        # up-to-~120ms early snap). This was A/B/C-tested against "skip" and
        # "search backwards for the last a-side frame" on Chidori humangold +
        # byoushin (eval_postreview*.tsv): the edge snap wins 3 of 4 metrics,
        # because karaoke-guide bars lead the sung portamento — exactly the
        # convention the bar display targets. Deliberate, not an accident.
        t_star = float(times[switched[0]])
        if abs(t_star - joint) < 1e-3:
            continue
        t_star = float(np.clip(t_star, lo, hi))
        a[1] = t_star
        b[0] = t_star
        moved += 1
    return [tuple(n) for n in out if n[1] - n[0] > 0.02], moved


def absorb_shakuri(
    notes: list[Note],
    track: F0Track,
    line_bounds: list[tuple[float, float]],
    *,
    max_onset: float = 0.28,
    max_jump: int = 3,
) -> tuple[list[Note], int]:
    """Fold a short rising onset note into the longer target note above it."""

    def same_line(t1: float, t2: float) -> bool:
        return any(s - 0.05 <= t1 and t2 <= e + 0.05 for s, e in line_bounds)

    out = [list(n) for n in sorted(notes)]
    folded = 0
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        dur_a, dur_b = a[1] - a[0], b[1] - b[0]
        jump = b[2] - a[2]
        if not (
            b[0] - a[1] <= 0.10
            and dur_a < max_onset
            and dur_b >= 1.3 * dur_a
            and 1 <= jump <= max_jump
            and same_line(a[0], b[1])
        ):
            continue
        _, midi = _voiced_midi(track, a[0], a[1])
        if midi.size < 3:
            continue
        head = np.median(midi[: max(1, midi.size // 3)])
        tail = np.median(midi[-max(1, midi.size // 3):])
        rising = tail - head > 0.5
        lands = abs(tail - b[2]) <= 0.8
        if rising or lands:
            a[2] = b[2]
            folded += 1
    return [tuple(n) for n in out], folded


def chroma_weights(audio_path: Path) -> np.ndarray:
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    w = chroma.mean(axis=1)
    return w / w.max()


def chroma_snap(
    notes: list[Note],
    track: F0Track,
    weights: np.ndarray,
    *,
    min_dev: float = 0.25,
    min_ratio: float = 1.3,
) -> tuple[list[Note], int]:
    """Move a note one semitone toward a much stronger pitch class when the
    sung F0 already leans that way (the flat/sharp rounding family)."""
    out = []
    snapped = 0
    for s, e, p in notes:
        _, midi = _voiced_midi(track, s, e, edge_trim=0.04)
        new_p = p
        if midi.size >= 3:
            dev = float(np.median(midi)) - p
            if dev >= min_dev and weights[(p + 1) % 12] >= min_ratio * weights[p % 12]:
                new_p = p + 1
            elif dev <= -min_dev and weights[(p - 1) % 12] >= min_ratio * weights[p % 12]:
                new_p = p - 1
        snapped += int(new_p != p)
        out.append((s, e, new_p))
    return out, snapped


def extend_sustains(
    notes: list[Note],
    track: F0Track,
    *,
    tolerance: float = 0.7,
    max_extend: float = 1.5,
    max_holes: int = 3,
) -> tuple[list[Note], int]:
    """Push a note's end forward while F0 keeps holding its pitch."""
    out = [list(n) for n in sorted(notes)]
    hop = track.hop_seconds or 0.01
    extended = 0
    for i, note in enumerate(out):
        limit = out[i + 1][0] if i + 1 < len(out) else note[1] + max_extend
        limit = min(limit, note[1] + max_extend)
        if limit - note[1] < 2 * hop:
            continue
        mask = (track.times >= note[1]) & (track.times < limit)
        times = track.times[mask]
        midi = track.midi[mask]
        holes = 0
        new_end = note[1]
        for t, m in zip(times, midi):
            if np.isfinite(m) and abs(m - note[2]) <= tolerance:
                new_end = t + hop
                holes = 0
            else:
                holes += 1
                if holes > max_holes:
                    break
        if new_end - note[1] >= 0.06:
            note[1] = min(new_end, limit)
            extended += 1
    return [tuple(n) for n in out], extended


def capture_tail_falls(
    notes: list[Note],
    track: F0Track,
    *,
    min_gap: float = 0.35,
    min_jump: int = 1,
    max_jump: int = 4,
    min_plateau: float = 0.25,
    tolerance: float = 0.7,
    max_extend: float = 1.5,
) -> tuple[list[Note], int]:
    """Add the sustained note a phrase tail falls onto.

    Singers often drop the last mora onto a lower sustained pitch that the
    aligned lyric grid does not cover (verse る = F3 then Eb3; the 目眩晴れ
    tails = F4 then Eb4 — both confirmed by the Chidori sheet/ear gold).
    Eligible notes are those followed by a lyric gap of at least min_gap
    (phrase tails happen at sub-phrase pauses too, not only line ends): if
    voiced F0 continues past the note end and settles on a stable plateau
    1..max_jump semitones away, append a new note covering the plateau
    instead of stretching the old pitch over it.
    """
    out = sorted(notes)
    hop = track.hop_seconds or 0.01
    added = 0
    new_notes: list[Note] = []
    for i, (s, e, p) in enumerate(out):
        nxt = out[i + 1][0] if i + 1 < len(out) else e + max_extend + min_gap
        if nxt - e < min_gap:
            continue
        limit = min(e + max_extend, nxt)
        mask = (track.times >= e) & (track.times < limit)
        midi = track.midi[mask]
        times = track.times[mask]
        keep = np.isfinite(midi)
        midi, times = midi[keep], times[keep]
        if midi.size * hop < min_plateau:
            continue
        target = float(np.median(midi))
        target_p = int(round(target))
        if not (min_jump <= abs(target_p - p) <= max_jump):
            continue
        on_plateau = np.abs(midi - target_p) <= tolerance
        if np.sum(on_plateau) * hop < min_plateau:
            continue
        plateau_times = times[on_plateau]
        new_notes.append((float(plateau_times[0]), float(plateau_times[-1] + hop), target_p))
        added += 1
    return sorted(out + new_notes), added


def merge_same_pitch(notes: list[Note], max_gap: float = 0.025) -> list[Note]:
    merged: list[list[float | int]] = []
    for s, e, p in sorted(notes):
        if merged and p == merged[-1][2] and s - merged[-1][1] <= max_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e, p])
    return [tuple(n) for n in merged]


@click.command()
@click.option("--midi", "midi_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--f0", "f0_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option(
    "--aligned",
    "aligned_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Aligned lyric JSON. Optional: without it, --fill-morae is unavailable "
    "and shakuri absorption treats the whole song as one line.",
)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--chroma-audio", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--fill-morae/--no-fill-morae", default=False)
@click.option("--refine-boundaries/--no-refine-boundaries", "do_refine", default=False)
@click.option("--absorb-shakuri/--no-absorb-shakuri", "do_shakuri", default=False)
@click.option("--chroma-prior/--no-chroma-prior", default=False)
@click.option("--extend-sustains/--no-extend-sustains", "do_extend", default=False)
@click.option("--capture-tail-falls/--no-capture-tail-falls", "do_tail_falls", default=False)
def main(
    midi_path: str,
    f0_path: str,
    aligned_path: str,
    out_path: str,
    chroma_audio: str | None,
    fill_morae: bool,
    do_refine: bool,
    do_shakuri: bool,
    chroma_prior: bool,
    do_extend: bool,
    do_tail_falls: bool,
) -> None:
    track = F0Track.from_npz(f0_path)
    if aligned_path:
        aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
        windows = rmm._mora_windows(aligned)
        line_ids = sorted({line for *_rest, line in windows})
        line_bounds = [
            (
                min(s for s, _e, _k, l in windows if l == line),
                max(e for _s, e, _k, l in windows if l == line),
            )
            for line in line_ids
        ]
    else:
        if fill_morae:
            raise click.UsageError("--fill-morae requires --aligned")
        windows = []
        line_bounds = [(0.0, float(track.times[-1]) + 10.0 if track.times.size else 1e9)]

    notes: list[Note] = [(n.start, n.end, n.pitch) for n in read_midi_notes(Path(midi_path))]
    stats: dict[str, int] = {}

    if fill_morae:
        notes, stats["filled"] = fill_missing_morae(notes, windows, track)
    if do_refine:
        before = len(notes)
        notes, stats["boundaries_moved"] = refine_boundaries(notes, track)
        if before != len(notes):
            stats["refine_dropped_sub20ms"] = before - len(notes)
    if do_shakuri:
        notes, stats["shakuri_folded"] = absorb_shakuri(notes, track, line_bounds)
    if chroma_prior:
        if not chroma_audio:
            raise click.UsageError("--chroma-prior requires --chroma-audio")
        weights = chroma_weights(Path(chroma_audio))
        notes, stats["chroma_snapped"] = chroma_snap(notes, track, weights)
    if do_tail_falls:
        notes, stats["tail_falls_captured"] = capture_tail_falls(notes, track)
    if do_extend:
        notes, stats["sustains_extended"] = extend_sustains(notes, track)

    notes = merge_same_pitch(notes)
    tempo = read_first_tempo_bpm(Path(midi_path))
    _write_midi(notes, Path(out_path), tempo=tempo)
    print(f"[score-note-postfix] wrote {out_path} notes={len(notes)} {stats}")


if __name__ == "__main__":
    main()
