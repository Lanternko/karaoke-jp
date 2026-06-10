#!/usr/bin/env python3
"""Rebuild a melody guide with one note per sung mora.

This is a sidecar experiment for karaoke pitch bars.  The normal melody MIDI
comes from F0 segmentation, so long vowels may be under-segmented relative to
the lyric/mora timing.  This tool instead treats the aligned lyric timing as
the event grid and uses an F0 track only to choose the pitch for each mora.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import click
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import midi_timing  # noqa: E402

from karaoke_jp.melody import _write_midi  # noqa: E402
from karaoke_jp.pitch_eval import F0Track  # noqa: E402
from karaoke_jp.score_melody import MidiNote, read_first_tempo_bpm, read_midi_notes  # noqa: E402


def _mora_windows(aligned: list[dict]) -> list[tuple[float, float, str, int]]:
    """Return (start, end, kana, line_index) windows, one per sung mora."""
    windows: list[tuple[float, float, str, int]] = []
    for line_idx, line in enumerate(aligned):
        morae = midi_timing.expand_line_to_morae(line)
        grouped: dict[int, list[dict]] = defaultdict(list)
        for mora in morae:
            grouped[id(mora["char"])].append(mora)

        seen: set[int] = set()
        for mora in morae:
            char = mora["char"]
            cid = id(char)
            if cid in seen:
                continue
            seen.add(cid)
            group = grouped[cid]
            start = float(char["start"])
            end = float(char["end"])
            if end <= start:
                continue
            step = (end - start) / len(group)
            for idx, member in enumerate(group):
                s = start + step * idx
                e = start + step * (idx + 1)
                windows.append((s, e, str(member.get("kana", "")), line_idx))
    return windows


def _f0_midi_mask(
    track: F0Track,
    start: float,
    end: float,
    *,
    edge_trim: float,
) -> np.ndarray:
    if end <= start:
        return np.zeros(track.times.shape, dtype=bool)
    trim = min(edge_trim, (end - start) * 0.25)
    s = start + trim
    e = end - trim
    if e <= s:
        s, e = start, end
    midi = track.midi
    mask = (track.times >= s) & (track.times < e) & np.isfinite(midi)
    return mask


def _longest_contiguous_run(
    frame_times: np.ndarray,
    hop: float,
) -> int:
    """Return the length of the longest contiguous run of frames."""
    if frame_times.size == 0:
        return 0
    longest = 1
    current = 1
    for prev_t, cur_t in zip(frame_times, frame_times[1:], strict=False):
        if cur_t - prev_t <= hop * 1.75:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return max(longest, current)


def _pitch_from_f0(
    track: F0Track,
    start: float,
    end: float,
    *,
    edge_trim: float,
    min_voiced_frames: int,
    quantile: float,
    chooser: str,
    plateau_tolerance: float = 0.65,
) -> int | None:
    if end <= start:
        return None
    midi = track.midi
    mask = _f0_midi_mask(track, start, end, edge_trim=edge_trim)
    if int(np.count_nonzero(mask)) < min_voiced_frames:
        mask = (track.times >= start) & (track.times < end) & np.isfinite(midi)
    if int(np.count_nonzero(mask)) < min_voiced_frames:
        return None
    values = midi[mask]
    if chooser == "quantile":
        return int(round(float(np.quantile(values, quantile))))
    if chooser == "mode":
        rounded = np.rint(values).astype(np.int16)
        pitches, counts = np.unique(rounded, return_counts=True)
        max_count = int(np.max(counts))
        candidates = pitches[counts == max_count]
        if candidates.size == 1:
            return int(candidates[0])
        tie_break = float(np.quantile(values, quantile))
        return int(candidates[np.argmin(np.abs(candidates - tie_break))])
    if chooser == "plateau":
        idx = np.flatnonzero(mask)
        frame_times = track.times[idx]
        hop = track.hop_seconds or 0.01
        rounded = np.rint(values).astype(int)
        unique_pitches = np.unique(rounded)
        best_pitch: int | None = None
        best_plateau = 0
        for p in unique_pitches:
            close = np.abs(values - p) <= plateau_tolerance
            close_times = frame_times[close]
            run = _longest_contiguous_run(close_times, hop)
            if run > best_plateau or (run == best_plateau and best_pitch is not None and abs(p - float(np.quantile(values, quantile))) < abs(best_pitch - float(np.quantile(values, quantile)))):
                best_plateau = run
                best_pitch = int(p)
        return best_pitch
    raise ValueError(f"Unsupported pitch chooser: {chooser}")


def _fallback_pitch_from_midi(notes: list[MidiNote], start: float, end: float) -> int | None:
    best_pitch: int | None = None
    best_overlap = 0.0
    for note in notes:
        overlap = min(note.end, end) - max(note.start, start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_pitch = note.pitch
    return best_pitch


def _choose_pitch(
    *,
    source: str,
    f0_pitch: int | None,
    midi_pitch: int | None,
) -> int | None:
    if source == "midi":
        return midi_pitch if midi_pitch is not None else f0_pitch
    if source == "f0":
        return f0_pitch if f0_pitch is not None else midi_pitch
    if source == "consensus":
        if f0_pitch is None:
            return midi_pitch
        if midi_pitch is None:
            return f0_pitch
        # Keep the smoother MIDI prior for small disagreements; use F0 only for
        # clear octave-level corrections or large pitch-class misses.
        diff = f0_pitch - midi_pitch
        if abs(abs(diff) - 12) <= 1:
            return midi_pitch + (12 if diff > 0 else -12)
        if abs(diff) >= 5:
            return f0_pitch
        return midi_pitch
    raise ValueError(f"Unsupported pitch source: {source}")


def _fill_missing_pitches(notes: list[tuple[float, float, int | None]]) -> list[tuple[float, float, int]]:
    known = [(idx, pitch) for idx, (_s, _e, pitch) in enumerate(notes) if pitch is not None]
    if not known:
        raise ValueError("No pitch could be inferred from F0 or fallback MIDI.")
    filled: list[tuple[float, float, int]] = []
    for idx, (start, end, pitch) in enumerate(notes):
        if pitch is None:
            nearest_idx, nearest_pitch = min(known, key=lambda item: abs(item[0] - idx))
            _ = nearest_idx
            pitch = nearest_pitch
        filled.append((start, end, int(pitch)))
    return filled


def _pitch_evidence(
    track: F0Track,
    start: float,
    end: float,
    pitch: int,
    *,
    edge_trim: float,
    tolerance: float,
) -> tuple[float, float, int]:
    """Return (support_ratio, longest_plateau_seconds, voiced_frames)."""
    midi = track.midi
    mask = _f0_midi_mask(track, start, end, edge_trim=edge_trim)
    if not np.any(mask):
        mask = (track.times >= start) & (track.times < end) & np.isfinite(midi)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0.0, 0.0, 0

    close = np.abs(midi[idx] - pitch) <= tolerance
    support = float(np.count_nonzero(close) / idx.size)
    if not np.any(close):
        return support, 0.0, int(idx.size)

    close_times = track.times[idx[close]]
    hop = track.hop_seconds or 0.01
    longest = _longest_contiguous_run(close_times, hop)
    return support, longest * hop, int(idx.size)


def _merge_adjacent_same_pitch(
    notes: list[tuple[float, float, int]],
    *,
    max_gap: float,
) -> tuple[list[tuple[float, float, int]], int]:
    merged: list[tuple[float, float, int]] = []
    merges = 0
    for start, end, pitch in notes:
        if merged and pitch == merged[-1][2] and start <= merged[-1][1] + max_gap:
            prev_start, prev_end, prev_pitch = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_pitch)
            merges += 1
        else:
            merged.append((start, end, pitch))
    return merged, merges


def _validate_mora_notes(
    notes: list[tuple[float, float, int]],
    *,
    f0: F0Track,
    edge_trim: float,
    pitch_tolerance: float,
    min_plateau_duration: float,
    min_support_ratio: float,
    short_duration: float,
    hard_short_duration: float,
    merge_same_pitch_gap: float,
) -> tuple[list[tuple[float, float, int]], dict[str, int]]:
    """Suppress mora-level pitch fragments that lack stable F0 evidence.

    This deliberately does not cap notes per beat.  It starts from the lyric
    mora grid, then only absorbs very short or weakly supported pitch changes
    into neighboring stable notes.
    """
    if not notes:
        return [], {
            "aba_absorbed": 0,
            "hard_short_absorbed": 0,
            "island_absorbed": 0,
            "weak_neighbor_absorbed": 0,
            "same_pitch_merges": 0,
        }

    adjusted = [list(note) for note in notes]
    stats = {
        "aba_absorbed": 0,
        "hard_short_absorbed": 0,
        "island_absorbed": 0,
        "weak_neighbor_absorbed": 0,
        "same_pitch_merges": 0,
    }

    evidence = [
        _pitch_evidence(
            f0,
            start,
            end,
            pitch,
            edge_trim=edge_trim,
            tolerance=pitch_tolerance,
        )
        for start, end, pitch in notes
    ]

    # Pass 1: absorb weak notes (short / unvoiced / low-support).
    for idx, (start, end, pitch) in enumerate(notes):
        duration = end - start
        support, plateau, voiced = evidence[idx]
        weak = (
            voiced == 0
            or duration < hard_short_duration
            or (
                duration < short_duration
                and (plateau < min_plateau_duration or support < min_support_ratio)
            )
        )
        if not weak:
            continue

        prev_pitch = int(adjusted[idx - 1][2]) if idx > 0 else None
        next_pitch = int(adjusted[idx + 1][2]) if idx + 1 < len(adjusted) else None

        if (
            prev_pitch is not None
            and next_pitch is not None
            and abs(prev_pitch - next_pitch) <= 1
            and abs(int(pitch) - round((prev_pitch + next_pitch) / 2)) >= 2
        ):
            adjusted[idx][2] = int(round((prev_pitch + next_pitch) / 2))
            stats["aba_absorbed"] += 1
            continue

        if duration < hard_short_duration and (prev_pitch is not None or next_pitch is not None):
            if prev_pitch is None:
                adjusted[idx][2] = next_pitch
            elif next_pitch is None:
                adjusted[idx][2] = prev_pitch
            else:
                prev_dur = adjusted[idx - 1][1] - adjusted[idx - 1][0]
                next_dur = adjusted[idx + 1][1] - adjusted[idx + 1][0]
                adjusted[idx][2] = prev_pitch if prev_dur >= next_dur else next_pitch
            stats["hard_short_absorbed"] += 1
            continue

        if prev_pitch is not None and abs(int(pitch) - prev_pitch) <= 1:
            adjusted[idx][2] = prev_pitch
            stats["weak_neighbor_absorbed"] += 1
        elif next_pitch is not None and abs(int(pitch) - next_pitch) <= 1:
            adjusted[idx][2] = next_pitch
            stats["weak_neighbor_absorbed"] += 1

    # Pass 2: absorb pitch islands — notes whose pitch differs from both
    # neighbors by ≥2 semitones and lacks a stable F0 plateau.  This runs
    # after pass 1 so that already-corrected neighbors are used.
    for idx in range(1, len(adjusted) - 1):
        pitch = int(adjusted[idx][2])
        prev_pitch = int(adjusted[idx - 1][2])
        next_pitch = int(adjusted[idx + 1][2])
        _support, plateau, _voiced = evidence[idx]
        if (
            abs(pitch - prev_pitch) >= 2
            and abs(pitch - next_pitch) >= 2
            and plateau < min_plateau_duration
        ):
            closer = prev_pitch if abs(pitch - prev_pitch) <= abs(pitch - next_pitch) else next_pitch
            adjusted[idx][2] = closer
            stats["island_absorbed"] += 1

    adjusted_notes = [(float(s), float(e), int(p)) for s, e, p in adjusted]
    adjusted_notes, merges = _merge_adjacent_same_pitch(
        adjusted_notes,
        max_gap=merge_same_pitch_gap,
    )
    stats["same_pitch_merges"] = merges
    return adjusted_notes, stats


@click.command()
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--f0", "f0_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--fallback-midi", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--tempo", type=float, default=None)
@click.option("--edge-trim", type=float, default=0.04, show_default=True)
@click.option("--min-voiced-frames", type=int, default=2, show_default=True)
@click.option(
    "--f0-quantile",
    type=click.FloatRange(0.0, 1.0),
    default=0.5,
    show_default=True,
    help="Quantile of voiced F0 MIDI values inside each mora window. Used directly by quantile chooser and as a tie-breaker by mode chooser.",
)
@click.option(
    "--pitch-chooser",
    type=click.Choice(["quantile", "mode", "plateau"]),
    default="quantile",
    show_default=True,
    help="How to summarize voiced F0 inside one mora. mode: most frequent semitone. plateau: semitone with longest contiguous stable run.",
)
@click.option("--plateau-tolerance", type=float, default=0.65, show_default=True,
              help="Semitone tolerance for plateau chooser contiguity check.")
@click.option("--min-duration", type=float, default=0.04, show_default=True)
@click.option(
    "--pitch-source",
    type=click.Choice(["f0", "midi", "consensus"]),
    default="consensus",
    show_default=True,
    help="mora note pitch source: f0 raw median, fallback MIDI prior, or consensus.",
)
@click.option(
    "--validate-mora-notes/--no-validate-mora-notes",
    default=False,
    show_default=True,
    help="Absorb unstable short mora notes into neighboring stable pitches.",
)
@click.option("--validator-pitch-tolerance", type=float, default=0.65, show_default=True)
@click.option("--validator-min-plateau", type=float, default=0.10, show_default=True)
@click.option("--validator-min-support", type=float, default=0.35, show_default=True)
@click.option("--validator-short-duration", type=float, default=0.15, show_default=True)
@click.option("--validator-hard-short-duration", type=float, default=0.08, show_default=True)
@click.option("--merge-same-pitch-gap", type=float, default=0.025, show_default=True)
def main(
    aligned_path: str,
    f0_path: str,
    fallback_midi: str | None,
    out_path: str,
    tempo: float | None,
    edge_trim: float,
    min_voiced_frames: int,
    f0_quantile: float,
    pitch_chooser: str,
    plateau_tolerance: float,
    min_duration: float,
    pitch_source: str,
    validate_mora_notes: bool,
    validator_pitch_tolerance: float,
    validator_min_plateau: float,
    validator_min_support: float,
    validator_short_duration: float,
    validator_hard_short_duration: float,
    merge_same_pitch_gap: float,
) -> None:
    aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    windows = _mora_windows(aligned)
    f0 = F0Track.from_npz(f0_path)
    fallback_notes = read_midi_notes(fallback_midi) if fallback_midi else []

    raw_notes: list[tuple[float, float, int | None]] = []
    from_f0 = 0
    from_fallback = 0
    from_consensus = 0
    for start, end, _kana, _line_idx in windows:
        if end - start < min_duration:
            mid = 0.5 * (start + end)
            start = mid - min_duration / 2
            end = mid + min_duration / 2
        f0_pitch = _pitch_from_f0(
            f0,
            start,
            end,
            edge_trim=edge_trim,
            min_voiced_frames=min_voiced_frames,
            quantile=f0_quantile,
            chooser=pitch_chooser,
            plateau_tolerance=plateau_tolerance,
        )
        if f0_pitch is not None:
            from_f0 += 1
        midi_pitch = _fallback_pitch_from_midi(fallback_notes, start, end) if fallback_notes else None
        if midi_pitch is not None:
            from_fallback += 1
        pitch = _choose_pitch(source=pitch_source, f0_pitch=f0_pitch, midi_pitch=midi_pitch)
        if pitch is not None:
            from_consensus += 1
        raw_notes.append((start, end, pitch))

    filled_count = len(raw_notes) - from_consensus
    notes = _fill_missing_pitches(raw_notes)
    validation_stats: dict[str, int] | None = None
    if validate_mora_notes:
        notes, validation_stats = _validate_mora_notes(
            notes,
            f0=f0,
            edge_trim=edge_trim,
            pitch_tolerance=validator_pitch_tolerance,
            min_plateau_duration=validator_min_plateau,
            min_support_ratio=validator_min_support,
            short_duration=validator_short_duration,
            hard_short_duration=validator_hard_short_duration,
            merge_same_pitch_gap=merge_same_pitch_gap,
        )
    tempo_bpm = tempo
    if tempo_bpm is None and fallback_midi:
        tempo_bpm = read_first_tempo_bpm(fallback_midi)
    if tempo_bpm is None:
        tempo_bpm = 120.0

    out = Path(out_path)
    _write_midi(notes, out, tempo=tempo_bpm)
    click.echo(
        f"[mora-refit] wrote {len(notes)} notes -> {out} "
        f"(morae={len(windows)}, f0={from_f0}, fallback={from_fallback}, "
        f"chosen={from_consensus}, filled={filled_count}, "
        f"pitch_source={pitch_source}, pitch_chooser={pitch_chooser}, "
        f"f0_quantile={f0_quantile})"
    )
    if validation_stats is not None:
        click.echo(f"[mora-refit] validator={validation_stats}")


if __name__ == "__main__":
    main()
