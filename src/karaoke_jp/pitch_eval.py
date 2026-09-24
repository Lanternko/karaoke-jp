"""Pitch-alignment evaluation helpers.

The evaluator deliberately keeps estimator, pitch-fixer, and rendered MIDI
questions separate.  It compares a monophonic MIDI melody against one or more
frame-wise F0 tracks, while also using lyric windows to distinguish stable
in-character regions from transitions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Self

import numpy as np

from .score_melody import MidiNote

PUNCT_CHARS = set("、。,.!?！？()（）「」『』[]【】…・:：;； ")


@dataclass(frozen=True)
class F0Track:
    """Frame-wise F0 values with absolute frame times."""

    times: np.ndarray
    f0_hz: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path) -> Self:
        with np.load(path) as data:
            f0 = data["f0"].astype(np.float64)
            if "times" in data:
                times = data["times"].astype(np.float64)
            elif "hop_seconds" in data:
                hop = float(np.asarray(data["hop_seconds"]).reshape(-1)[0])
                times = np.arange(f0.size, dtype=np.float64) * hop
            else:
                raise ValueError(f"{path} must contain either times or hop_seconds")
        if times.shape != f0.shape:
            raise ValueError(f"{path} has mismatched f0/times shapes: {f0.shape} vs {times.shape}")
        return cls(times=times, f0_hz=f0)

    @cached_property
    def hop_seconds(self) -> float:
        if self.times.size < 2:
            return 0.0
        return float(np.median(np.diff(self.times)))

    @cached_property
    def midi(self) -> np.ndarray:
        # cached_property writes to instance __dict__ directly, which is
        # compatible with frozen dataclasses; callers access .midi once per
        # note/window, so recomputing the full-track log2 each time is O(N*M).
        midi = np.full(self.f0_hz.shape, np.nan, dtype=np.float64)
        voiced = np.isfinite(self.f0_hz) & (self.f0_hz > 0)
        midi[voiced] = 69.0 + 12.0 * np.log2(self.f0_hz[voiced] / 440.0)
        return midi


@dataclass(frozen=True)
class TimeWindow:
    start: float
    end: float
    label: str = ""
    line_index: int = -1

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class PitchMetrics:
    frames: int
    rpa: float
    rca: float
    octave_proxy: float
    octave_frames: int
    octave_rate: float
    non_octave_within_50c: float
    non_octave_50_150c: float
    non_octave_150_250c: float
    non_octave_gt_250c: float
    folded_median_abs_cents: float | None
    folded_p90_abs_cents: float | None
    note_cmp: int
    note_octave: int
    note_1_semitone: int
    note_2_semitone: int
    note_gt_250c: int


@dataclass(frozen=True)
class FragmentationMetrics:
    notes: int
    outside_lyric_window: int
    same_pitch_tiny_gap: int
    rapid_aba_jitter: int
    long_char_multi_note: int
    long_char_pitch_span_ge_2: int


def _is_sung_char(token: dict[str, Any], char_obj: dict[str, Any]) -> bool:
    ch = str(char_obj.get("char", ""))
    return bool(ch) and not bool(token.get("is_punct")) and ch not in PUNCT_CHARS


def lyric_line_windows(aligned: list[dict[str, Any]]) -> list[TimeWindow]:
    """Return visible lyric-line windows that contain at least one sung char."""
    windows: list[TimeWindow] = []
    for line_idx, line in enumerate(aligned):
        starts: list[float] = []
        ends: list[float] = []
        for token in line.get("tokens", []):
            for char_obj in token.get("chars", []):
                if not _is_sung_char(token, char_obj):
                    continue
                starts.append(float(char_obj["start"]))
                ends.append(float(char_obj["end"]))
        if starts and ends:
            windows.append(
                TimeWindow(
                    start=min(starts),
                    end=max(ends),
                    label=str(line.get("text", "")),
                    line_index=line_idx,
                )
            )
    return windows


def lyric_char_windows(aligned: list[dict[str, Any]]) -> list[TimeWindow]:
    """Return sung-character windows from an aligned sidecar."""
    windows: list[TimeWindow] = []
    for line_idx, line in enumerate(aligned):
        for token in line.get("tokens", []):
            for char_obj in token.get("chars", []):
                if not _is_sung_char(token, char_obj):
                    continue
                start = float(char_obj["start"])
                end = float(char_obj["end"])
                if end <= start:
                    continue
                windows.append(
                    TimeWindow(
                        start=start,
                        end=end,
                        label=str(char_obj.get("char", "")),
                        line_index=line_idx,
                    )
                )
    return windows


def stable_char_windows(
    char_windows: list[TimeWindow],
    *,
    trim_seconds: float = 0.05,
    min_duration: float = 0.16,
) -> list[TimeWindow]:
    """Trim char boundaries to approximate stable vowel-like interiors."""
    stable: list[TimeWindow] = []
    for window in char_windows:
        if window.duration < min_duration:
            continue
        trim = min(trim_seconds, window.duration * 0.25)
        start = window.start + trim
        end = window.end - trim
        if end > start:
            stable.append(
                TimeWindow(
                    start=start,
                    end=end,
                    label=window.label,
                    line_index=window.line_index,
                )
            )
    return stable


def transition_char_windows(
    char_windows: list[TimeWindow],
    *,
    trim_seconds: float = 0.05,
    min_duration: float = 0.16,
) -> list[TimeWindow]:
    """Return char-edge regions excluded from ``stable_char_windows``."""
    transition: list[TimeWindow] = []
    for window in char_windows:
        if window.duration < min_duration:
            transition.append(window)
            continue
        trim = min(trim_seconds, window.duration * 0.25)
        left_end = window.start + trim
        right_start = window.end - trim
        if left_end > window.start:
            transition.append(TimeWindow(window.start, left_end, window.label, window.line_index))
        if window.end > right_start:
            transition.append(TimeWindow(right_start, window.end, window.label, window.line_index))
    return transition


def mask_times(times: np.ndarray, windows: list[TimeWindow]) -> np.ndarray:
    mask = np.zeros(times.shape, dtype=bool)
    for window in windows:
        mask |= (times >= window.start) & (times < window.end)
    return mask


def folded_cents(diff_cents: np.ndarray) -> np.ndarray:
    """Fold pitch differences into one octave, preserving semitone error."""
    return ((diff_cents + 600.0) % 1200.0) - 600.0


def octave_like_mask(diff_cents: np.ndarray, *, tolerance_cents: float = 120.0) -> np.ndarray:
    return np.abs(np.abs(diff_cents) - 1200.0) <= tolerance_cents


def _empty_pitch_metrics() -> PitchMetrics:
    return PitchMetrics(
        frames=0,
        rpa=0.0,
        rca=0.0,
        octave_proxy=0.0,
        octave_frames=0,
        octave_rate=0.0,
        non_octave_within_50c=0.0,
        non_octave_50_150c=0.0,
        non_octave_150_250c=0.0,
        non_octave_gt_250c=0.0,
        folded_median_abs_cents=None,
        folded_p90_abs_cents=None,
        note_cmp=0,
        note_octave=0,
        note_1_semitone=0,
        note_2_semitone=0,
        note_gt_250c=0,
    )


def compare_notes_to_f0(
    notes: list[MidiNote],
    f0: F0Track,
    *,
    windows: list[TimeWindow] | None = None,
    octave_tolerance_cents: float = 120.0,
) -> PitchMetrics:
    """Compare MIDI notes against an F0 track.

    ``windows`` optionally restricts frame and note comparisons to a region
    such as stable char interiors or transitions.
    """
    if not notes:
        return _empty_pitch_metrics()

    f0_midi = f0.midi
    valid_window_mask = None if windows is None else mask_times(f0.times, windows)

    frame_diffs: list[np.ndarray] = []
    note_diffs: list[float] = []

    for note in notes:
        idx = (f0.times >= note.start) & (f0.times < note.end)
        if valid_window_mask is not None:
            idx &= valid_window_mask
        idx &= np.isfinite(f0_midi)
        if not np.any(idx):
            continue
        diffs = (f0_midi[idx] - float(note.pitch)) * 100.0
        frame_diffs.append(diffs)
        note_diffs.append(float(np.median(diffs)))

    if not frame_diffs:
        return _empty_pitch_metrics()

    diffs = np.concatenate(frame_diffs)
    folded = np.abs(folded_cents(diffs))
    octave_mask = octave_like_mask(diffs, tolerance_cents=octave_tolerance_cents)
    non_oct = ~octave_mask
    non_oct_count = int(np.count_nonzero(non_oct))

    rpa = float(np.mean(np.abs(diffs) <= 50.0))
    rca = float(np.mean(folded <= 50.0))
    octave_rate = float(np.mean(octave_mask))

    if non_oct_count:
        non_folded = folded[non_oct]
        within_50 = float(np.mean(non_folded <= 50.0))
        one = float(np.mean((non_folded > 50.0) & (non_folded <= 150.0)))
        two = float(np.mean((non_folded > 150.0) & (non_folded <= 250.0)))
        gt = float(np.mean(non_folded > 250.0))
        median_abs = float(np.median(non_folded))
        p90_abs = float(np.percentile(non_folded, 90))
    else:
        within_50 = one = two = gt = 0.0
        median_abs = p90_abs = None

    note_array = np.array(note_diffs, dtype=np.float64)
    note_oct = octave_like_mask(note_array, tolerance_cents=octave_tolerance_cents)
    note_folded = np.abs(folded_cents(note_array))
    note_non_oct = ~note_oct

    return PitchMetrics(
        frames=int(diffs.size),
        rpa=rpa,
        rca=rca,
        octave_proxy=max(0.0, rca - rpa),
        octave_frames=int(np.count_nonzero(octave_mask)),
        octave_rate=octave_rate,
        non_octave_within_50c=within_50,
        non_octave_50_150c=one,
        non_octave_150_250c=two,
        non_octave_gt_250c=gt,
        folded_median_abs_cents=median_abs,
        folded_p90_abs_cents=p90_abs,
        note_cmp=int(note_array.size),
        note_octave=int(np.count_nonzero(note_oct)),
        note_1_semitone=int(np.count_nonzero(note_non_oct & (note_folded > 50.0) & (note_folded <= 150.0))),
        note_2_semitone=int(np.count_nonzero(note_non_oct & (note_folded > 150.0) & (note_folded <= 250.0))),
        note_gt_250c=int(np.count_nonzero(note_non_oct & (note_folded > 250.0))),
    )


def count_notes_outside_windows(notes: list[MidiNote], windows: list[TimeWindow]) -> int:
    if not windows:
        return len(notes)
    outside = 0
    for note in notes:
        overlaps = any(note.end > window.start and note.start < window.end for window in windows)
        if not overlaps:
            outside += 1
    return outside


def count_same_pitch_tiny_gaps(notes: list[MidiNote], *, max_gap: float = 0.08) -> int:
    count = 0
    for prev, note in zip(notes, notes[1:], strict=False):
        gap = note.start - prev.end
        if prev.pitch == note.pitch and 0.0 <= gap <= max_gap:
            count += 1
    return count


def count_rapid_aba_jitter(
    notes: list[MidiNote],
    *,
    max_span: float = 0.80,
    max_gap: float = 0.12,
    max_middle_jump: int = 3,
) -> int:
    count = 0
    for left, mid, right in zip(notes, notes[1:], notes[2:], strict=False):
        if left.pitch != right.pitch or mid.pitch == left.pitch:
            continue
        if abs(mid.pitch - left.pitch) > max_middle_jump:
            continue
        if right.end - left.start > max_span:
            continue
        if mid.start - left.end > max_gap or right.start - mid.end > max_gap:
            continue
        count += 1
    return count


def long_char_fragmentation(
    notes: list[MidiNote],
    char_windows: list[TimeWindow],
    *,
    min_char_duration: float = 0.45,
    min_note_overlap: float = 0.03,
    pitch_span_threshold: int = 2,
) -> tuple[int, int]:
    multi_note = 0
    wide_span = 0
    for window in char_windows:
        if window.duration < min_char_duration:
            continue
        pitches: list[int] = []
        for note in notes:
            overlap = min(note.end, window.end) - max(note.start, window.start)
            if overlap >= min_note_overlap:
                pitches.append(note.pitch)
        if len(pitches) < 2:
            continue
        multi_note += 1
        if max(pitches) - min(pitches) >= pitch_span_threshold:
            wide_span += 1
    return multi_note, wide_span


def fragmentation_metrics(
    notes: list[MidiNote],
    *,
    line_windows: list[TimeWindow],
    char_windows: list[TimeWindow],
) -> FragmentationMetrics:
    multi_note, wide_span = long_char_fragmentation(notes, char_windows)
    return FragmentationMetrics(
        notes=len(notes),
        outside_lyric_window=count_notes_outside_windows(notes, line_windows),
        same_pitch_tiny_gap=count_same_pitch_tiny_gaps(notes),
        rapid_aba_jitter=count_rapid_aba_jitter(notes),
        long_char_multi_note=multi_note,
        long_char_pitch_span_ge_2=wide_span,
    )


def merge_adjacent_same_pitch_notes(
    notes: list[MidiNote],
    *,
    max_gap: float = 0.08,
) -> list[MidiNote]:
    if not notes:
        return []
    merged = [notes[0]]
    for note in notes[1:]:
        prev = merged[-1]
        gap = note.start - prev.end
        if note.pitch == prev.pitch and gap <= max_gap:
            merged[-1] = MidiNote(start=prev.start, end=max(prev.end, note.end), pitch=prev.pitch)
        else:
            merged.append(note)
    return merged


def _window_pitch_span(
    notes: list[MidiNote],
    window: TimeWindow,
    *,
    replace_index: int | None = None,
    replacement_pitch: int | None = None,
    min_note_overlap: float = 0.03,
) -> int:
    pitches: list[int] = []
    for idx, note in enumerate(notes):
        overlap = min(note.end, window.end) - max(note.start, window.start)
        if overlap < min_note_overlap:
            continue
        if replace_index is not None and idx == replace_index and replacement_pitch is not None:
            pitches.append(replacement_pitch)
        else:
            pitches.append(note.pitch)
    if len(pitches) < 2:
        return 0
    return max(pitches) - min(pitches)


def median_f0_diff_for_note(note: MidiNote, f0: F0Track) -> float | None:
    f0_midi = f0.midi
    idx = (f0.times >= note.start) & (f0.times < note.end) & np.isfinite(f0_midi)
    if not np.any(idx):
        return None
    return float(np.median((f0_midi[idx] - float(note.pitch)) * 100.0))


def shift_octave_notes_by_f0_consensus(
    notes: list[MidiNote],
    *,
    primary: F0Track,
    veto: F0Track | None = None,
    span_guard_windows: list[TimeWindow] | None = None,
    span_guard_min_duration: float = 0.45,
    octave_tolerance_cents: float = 120.0,
    veto_current_close_cents: float = 250.0,
) -> tuple[list[MidiNote], int]:
    """Shift only obvious octave errors, optionally vetoed by a second F0 track."""
    shifted: list[MidiNote] = []
    changes = 0
    for idx, note in enumerate(notes):
        primary_diff = median_f0_diff_for_note(note, primary)
        if primary_diff is None or not octave_like_mask(
            np.array([primary_diff]), tolerance_cents=octave_tolerance_cents
        )[0]:
            shifted.append(note)
            continue

        direction = 12 if primary_diff > 0 else -12
        veto_diff = median_f0_diff_for_note(note, veto) if veto is not None else None
        if veto_diff is not None and abs(veto_diff) <= veto_current_close_cents:
            shifted.append(note)
            continue

        candidate_pitch = note.pitch + direction
        if span_guard_windows:
            increases_span = False
            for window in span_guard_windows:
                if window.duration < span_guard_min_duration:
                    continue
                if min(note.end, window.end) - max(note.start, window.start) <= 0.0:
                    continue
                before = _window_pitch_span(notes, window)
                after = _window_pitch_span(
                    notes,
                    window,
                    replace_index=idx,
                    replacement_pitch=candidate_pitch,
                )
                if after > before:
                    increases_span = True
                    break
            if increases_span:
                shifted.append(note)
                continue

        shifted.append(MidiNote(note.start, note.end, candidate_pitch))
        changes += 1
    return shifted, changes


def metrics_to_dict(metrics: PitchMetrics | FragmentationMetrics) -> dict[str, Any]:
    return asdict(metrics)
