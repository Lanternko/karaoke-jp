from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from karaoke_jp.score_melody import (
    MidiNote,
    build_score_chroma,
    extract_top_voice_notes,
    map_notes_to_audio,
)


def test_extract_top_voice_notes_uses_highest_note_per_onset() -> None:
    notes = [
        MidiNote(start=0.0, end=2.0, pitch=60),   # accompaniment tail
        MidiNote(start=0.0, end=0.5, pitch=76),   # melody
        MidiNote(start=0.5, end=1.0, pitch=64),   # accompaniment only
        MidiNote(start=1.0, end=1.5, pitch=77),   # next melody note
        MidiNote(start=1.0, end=2.0, pitch=67),   # chord under melody
    ]

    top_voice = extract_top_voice_notes(notes)

    assert top_voice == [
        MidiNote(start=0.0, end=0.5, pitch=76),
        MidiNote(start=0.5, end=1.0, pitch=64),
        MidiNote(start=1.0, end=1.5, pitch=77),
    ]


def test_build_score_chroma_marks_pitch_classes_across_frames() -> None:
    chroma, origin = build_score_chroma(
        [
            MidiNote(start=1.0, end=1.5, pitch=60),  # C
            MidiNote(start=1.5, end=2.0, pitch=67),  # G
        ],
        hop_seconds=0.25,
    )

    assert origin == pytest.approx(1.0)
    assert chroma.shape == (12, 5)
    assert chroma[0, 0] == pytest.approx(1.0)
    assert chroma[0, 1] == pytest.approx(1.0)
    assert chroma[7, 2] == pytest.approx(1.0)
    assert chroma[7, 3] == pytest.approx(1.0)


def test_map_notes_to_audio_interpolates_monotone_times() -> None:
    notes = [MidiNote(start=0.0, end=1.0, pitch=72)]
    frame_map = np.array([0.0, 2.0, 4.0], dtype=np.float64)

    aligned = map_notes_to_audio(
        notes,
        score_origin=0.0,
        frame_map=frame_map,
        score_hop_seconds=0.5,
        audio_hop_seconds=0.25,
        audio_offset_seconds=1.0,
    )

    assert aligned == [MidiNote(start=1.0, end=2.0, pitch=72)]
