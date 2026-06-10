from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from karaoke_jp.pitch_eval import (
    F0Track,
    TimeWindow,
    compare_notes_to_f0,
    merge_adjacent_same_pitch_notes,
    shift_octave_notes_by_f0_consensus,
    stable_char_windows,
    transition_char_windows,
)
from karaoke_jp.score_melody import MidiNote


def _hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def test_compare_notes_to_f0_reports_rpa_rca_octave_gap() -> None:
    notes = [MidiNote(0.0, 0.4, 72)]
    track = F0Track(
        times=np.array([0.0, 0.1, 0.2, 0.3]),
        f0_hz=np.array([_hz(60), _hz(60), _hz(60), _hz(60)]),
    )

    metrics = compare_notes_to_f0(notes, track)

    assert metrics.frames == 4
    assert metrics.rpa == pytest.approx(0.0)
    assert metrics.rca == pytest.approx(1.0)
    assert metrics.octave_proxy == pytest.approx(1.0)
    assert metrics.note_octave == 1


def test_stable_and_transition_windows_split_char_edges() -> None:
    windows = [TimeWindow(1.0, 1.4, "あ", 0), TimeWindow(2.0, 2.08, "い", 0)]

    stable = stable_char_windows(windows, trim_seconds=0.05, min_duration=0.16)
    transition = transition_char_windows(windows, trim_seconds=0.05, min_duration=0.16)

    assert stable == [TimeWindow(1.05, 1.3499999999999999, "あ", 0)]
    assert transition == [
        TimeWindow(1.0, 1.05, "あ", 0),
        TimeWindow(1.3499999999999999, 1.4, "あ", 0),
        TimeWindow(2.0, 2.08, "い", 0),
    ]


def test_shift_octave_notes_by_f0_consensus_lowers_high_midi_note() -> None:
    notes = [MidiNote(0.0, 0.4, 72)]
    primary = F0Track(
        times=np.array([0.0, 0.1, 0.2, 0.3]),
        f0_hz=np.array([_hz(60), _hz(60), _hz(60), _hz(60)]),
    )
    veto = F0Track(
        times=np.array([0.0, 0.1, 0.2, 0.3]),
        f0_hz=np.array([_hz(60), _hz(60), _hz(60), _hz(60)]),
    )

    shifted, changes = shift_octave_notes_by_f0_consensus(notes, primary=primary, veto=veto)

    assert changes == 1
    assert shifted == [MidiNote(0.0, 0.4, 60)]


def test_shift_octave_notes_by_f0_consensus_vetoes_when_second_estimator_matches_current() -> None:
    notes = [MidiNote(0.0, 0.4, 72)]
    primary = F0Track(
        times=np.array([0.0, 0.1, 0.2, 0.3]),
        f0_hz=np.array([_hz(60), _hz(60), _hz(60), _hz(60)]),
    )
    veto = F0Track(
        times=np.array([0.0, 0.1, 0.2, 0.3]),
        f0_hz=np.array([_hz(72), _hz(72), _hz(72), _hz(72)]),
    )

    shifted, changes = shift_octave_notes_by_f0_consensus(notes, primary=primary, veto=veto)

    assert changes == 0
    assert shifted == notes


def test_shift_octave_notes_by_f0_consensus_vetoes_long_char_span_increase() -> None:
    notes = [
        MidiNote(0.0, 0.2, 72),
        MidiNote(0.2, 0.4, 72),
        MidiNote(0.4, 0.6, 72),
    ]
    primary = F0Track(
        times=np.array([0.45, 0.50, 0.55]),
        f0_hz=np.array([_hz(60), _hz(60), _hz(60)]),
    )
    guard = [TimeWindow(0.0, 0.6, "長", 0)]

    shifted, changes = shift_octave_notes_by_f0_consensus(
        notes,
        primary=primary,
        span_guard_windows=guard,
    )

    assert changes == 0
    assert shifted == notes


def test_consensus_unguarded_fixes_octave_cluster_that_guard_would_block() -> None:
    """Canonical octave fix is *unguarded* (2026-06-06 ablation).

    A run of octave-high notes sharing one long char window must be fully
    corrected, not left half-shifted.  The opt-in span guard instead blocks
    every shift, leaving the whole cluster an octave off (the chidori failure
    mode: residuals float an octave above the F0 contour).
    """
    notes = [
        MidiNote(0.0, 0.2, 72),
        MidiNote(0.2, 0.4, 72),
        MidiNote(0.4, 0.6, 72),
    ]
    f0 = F0Track(
        times=np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55]),
        f0_hz=np.array([_hz(60)] * 6),
    )
    long_window = [TimeWindow(0.0, 0.6, "長", 0)]

    # default (unguarded): the whole cluster comes down an octave and merges
    unguarded, n_unguarded = shift_octave_notes_by_f0_consensus(notes, primary=f0, veto=f0)
    assert n_unguarded == 3
    assert [n.pitch for n in unguarded] == [60, 60, 60]
    assert merge_adjacent_same_pitch_notes(unguarded) == [MidiNote(0.0, 0.6, 60)]

    # opt-in span guard: every shift blocked -> cluster stays an octave off
    guarded, n_guarded = shift_octave_notes_by_f0_consensus(
        notes, primary=f0, veto=f0, span_guard_windows=long_window
    )
    assert n_guarded == 0
    assert [n.pitch for n in guarded] == [72, 72, 72]


def test_merge_adjacent_same_pitch_notes_closes_tiny_gaps() -> None:
    notes = [
        MidiNote(0.0, 0.2, 64),
        MidiNote(0.24, 0.4, 64),
        MidiNote(0.6, 0.8, 64),
    ]

    merged = merge_adjacent_same_pitch_notes(notes, max_gap=0.08)

    assert merged == [
        MidiNote(0.0, 0.4, 64),
        MidiNote(0.6, 0.8, 64),
    ]
