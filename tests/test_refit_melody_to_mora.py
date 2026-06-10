from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import refit_melody_to_mora  # noqa: E402
from karaoke_jp.pitch_eval import F0Track  # noqa: E402


def test_mode_pitch_chooser_prefers_persistent_plateau_over_high_quantile() -> None:
    times = np.arange(0.0, 1.0, 0.01)
    midi = np.full(times.shape, 60.0)
    midi[times >= 0.70] = 67.0
    f0 = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
    track = F0Track(times=times, f0_hz=f0)

    quantile_pitch = refit_melody_to_mora._pitch_from_f0(
        track,
        0.0,
        1.0,
        edge_trim=0.0,
        min_voiced_frames=1,
        quantile=0.8,
        chooser="quantile",
    )
    mode_pitch = refit_melody_to_mora._pitch_from_f0(
        track,
        0.0,
        1.0,
        edge_trim=0.0,
        min_voiced_frames=1,
        quantile=0.8,
        chooser="mode",
    )

    assert quantile_pitch == 67
    assert mode_pitch == 60


def test_validator_absorbs_short_aba_jitter() -> None:
    times = np.arange(0.0, 0.5, 0.01)
    f0 = np.full(times.shape, 261.625565)  # C4 / MIDI 60
    track = F0Track(times=times, f0_hz=f0)
    notes = [
        (0.00, 0.20, 60),
        (0.20, 0.27, 64),
        (0.27, 0.50, 60),
    ]

    validated, stats = refit_melody_to_mora._validate_mora_notes(
        notes,
        f0=track,
        edge_trim=0.0,
        pitch_tolerance=0.65,
        min_plateau_duration=0.10,
        min_support_ratio=0.35,
        short_duration=0.15,
        hard_short_duration=0.08,
        merge_same_pitch_gap=0.025,
    )

    assert stats["aba_absorbed"] == 1
    assert stats["same_pitch_merges"] == 2
    assert validated == [(0.0, 0.5, 60)]


def test_validator_keeps_stable_short_plateau() -> None:
    times = np.arange(0.0, 0.5, 0.01)
    midi = np.full(times.shape, 60.0)
    midi[(times >= 0.20) & (times < 0.34)] = 64.0
    f0 = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
    track = F0Track(times=times, f0_hz=f0)
    notes = [
        (0.00, 0.20, 60),
        (0.20, 0.34, 64),
        (0.34, 0.50, 67),
    ]

    validated, stats = refit_melody_to_mora._validate_mora_notes(
        notes,
        f0=track,
        edge_trim=0.0,
        pitch_tolerance=0.65,
        min_plateau_duration=0.10,
        min_support_ratio=0.35,
        short_duration=0.15,
        hard_short_duration=0.08,
        merge_same_pitch_gap=0.025,
    )

    assert stats["aba_absorbed"] == 0
    assert stats["hard_short_absorbed"] == 0
    assert validated == notes


def test_plateau_chooser_picks_longest_stable_run() -> None:
    """70 frames G4 then 30 frames A#4 — plateau picks G4 (longer run)."""
    times = np.arange(0.0, 1.0, 0.01)
    midi = np.full(times.shape, 67.0)  # G4
    midi[times >= 0.70] = 70.0  # A#4
    f0 = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
    track = F0Track(times=times, f0_hz=f0)

    pitch = refit_melody_to_mora._pitch_from_f0(
        track, 0.0, 1.0,
        edge_trim=0.0, min_voiced_frames=1, quantile=0.8,
        chooser="plateau", plateau_tolerance=0.65,
    )
    assert pitch == 67


def test_plateau_chooser_ignores_scattered_frames() -> None:
    """A#4 appears in two separate 15-frame bursts (total 30) vs G4 contiguous 25 frames."""
    times = np.arange(0.0, 0.70, 0.01)
    midi = np.full(times.shape, 67.0)  # G4 baseline
    midi[0:15] = 70.0   # A#4 burst 1
    midi[40:55] = 70.0  # A#4 burst 2 (not contiguous with burst 1)
    f0 = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
    track = F0Track(times=times, f0_hz=f0)

    pitch = refit_melody_to_mora._pitch_from_f0(
        track, 0.0, 0.70,
        edge_trim=0.0, min_voiced_frames=1, quantile=0.8,
        chooser="plateau", plateau_tolerance=0.65,
    )
    # G4 has a 25-frame contiguous block (15..40) vs A#4's longest 15-frame block
    assert pitch == 67


def test_validator_absorbs_island_spike() -> None:
    """[F4, A#4, G4] where A#4 has no F0 support and is long enough to not be
    weak — neighbors differ by 2 semitones so ABA doesn't fire, but island does."""
    times = np.arange(0.0, 0.8, 0.01)
    f0 = np.full(times.shape, 392.0)  # G4 = MIDI 67
    track = F0Track(times=times, f0_hz=f0)
    notes = [
        (0.00, 0.25, 65),  # F4
        (0.25, 0.50, 70),  # A#4 — island, no F0 plateau at 70
        (0.50, 0.80, 67),  # G4
    ]

    validated, stats = refit_melody_to_mora._validate_mora_notes(
        notes,
        f0=track,
        edge_trim=0.0,
        pitch_tolerance=0.65,
        min_plateau_duration=0.10,
        min_support_ratio=0.35,
        short_duration=0.15,
        hard_short_duration=0.08,
        merge_same_pitch_gap=0.025,
    )

    assert stats["island_absorbed"] == 1
    # A#4 absorbed to closer neighbor G4 (|70-67|=3 < |70-65|=5)
    assert validated[1][2] == 67
