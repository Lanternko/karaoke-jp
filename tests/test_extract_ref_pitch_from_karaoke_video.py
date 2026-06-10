from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_ref_pitch_from_karaoke_video import (  # noqa: E402
    Sample,
    _enforce_monophonic_notes,
    _sort_unique_samples,
)


def test_sort_unique_samples_orders_by_time_and_drops_overlapped_frames() -> None:
    samples = [
        Sample(time=1.1, y=20.0, playhead_x=12, frame_path="b/frame_00004.png"),
        Sample(time=1.0, y=10.0, playhead_x=10, frame_path="b/frame_00001.png"),
        Sample(time=1.0000003, y=11.0, playhead_x=11, frame_path="a/frame_00001.png"),
    ]

    unique = _sort_unique_samples(samples)

    assert [(s.time, s.y, s.frame_path) for s in unique] == [
        (1.0, 10.0, "b/frame_00001.png"),
        (1.1, 20.0, "b/frame_00004.png"),
    ]


def test_enforce_monophonic_notes_trims_overlap_before_midi_serialization() -> None:
    notes = [
        (0.00, 0.50, 63),
        (0.45, 0.90, 70),
        (0.94, 1.20, 70),
    ]

    cleaned = _enforce_monophonic_notes(notes, min_duration=0.06)

    assert cleaned == [
        (0.00, 0.45, 63),
        (0.45, 1.20, 70),
    ]


def test_enforce_monophonic_notes_drops_too_short_previous_fragment() -> None:
    notes = [
        (0.00, 0.05, 63),
        (0.04, 0.30, 70),
    ]

    cleaned = _enforce_monophonic_notes(notes, min_duration=0.06)

    assert cleaned == [(0.04, 0.30, 70)]
