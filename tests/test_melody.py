from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from karaoke_jp.melody import segment_f0_to_notes


def _midi_to_hz(midi_note: int) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def test_segment_f0_to_notes_merges_short_vibrato_jump() -> None:
    f0 = [_midi_to_hz(60)] * 10 + [_midi_to_hz(61)] * 2 + [_midi_to_hz(60)] * 10

    notes = segment_f0_to_notes(
        f0_hz=np.array(f0, dtype=float),
        hop_seconds=0.01,
    )

    assert notes == pytest.approx([(0.0, 0.22, 60)])


def test_segment_f0_to_notes_splits_stable_pitch_change() -> None:
    f0 = [_midi_to_hz(60)] * 10 + [_midi_to_hz(62)] * 10

    notes = segment_f0_to_notes(
        f0_hz=np.array(f0, dtype=float),
        hop_seconds=0.01,
    )

    assert notes == pytest.approx([(0.0, 0.1, 60), (0.1, 0.2, 62)])


def test_segment_f0_to_notes_bridges_short_unvoiced_gap() -> None:
    f0 = [_midi_to_hz(60)] * 10 + [0.0, 0.0] + [_midi_to_hz(60)] * 10

    notes = segment_f0_to_notes(
        f0_hz=np.array(f0, dtype=float),
        hop_seconds=0.01,
    )

    assert notes == pytest.approx([(0.0, 0.22, 60)])


def test_segment_f0_to_notes_drops_brief_noise_burst() -> None:
    f0 = [0.0] * 5 + [_midi_to_hz(72)] * 2 + [0.0] * 5

    notes = segment_f0_to_notes(
        f0_hz=np.array(f0, dtype=float),
        hop_seconds=0.01,
    )

    assert notes == []
