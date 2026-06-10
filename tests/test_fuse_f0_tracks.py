from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fuse_f0_tracks  # noqa: E402


def test_primary_fill_only_fills_unvoiced_primary_frames() -> None:
    primary = np.array([60.0, np.nan, 62.0, np.nan])
    secondary = np.array([67.0, 64.0, 65.0, np.nan])

    fused = fuse_f0_tracks.fuse_midi_tracks(
        primary,
        secondary,
        strategy="primary-fill",
        agreement_tolerance=2.0,
    )

    np.testing.assert_allclose(fused[:3], [60.0, 64.0, 62.0])
    assert np.isnan(fused[3])


def test_agreement_average_keeps_primary_for_far_disagreement() -> None:
    primary = np.array([60.0, 60.0, np.nan])
    secondary = np.array([61.0, 67.0, 64.0])

    fused = fuse_f0_tracks.fuse_midi_tracks(
        primary,
        secondary,
        strategy="agree-avg-else-primary",
        agreement_tolerance=2.0,
    )

    np.testing.assert_allclose(fused, [60.5, 60.0, 64.0])


def test_agreement_average_can_prefer_secondary_for_far_disagreement() -> None:
    primary = np.array([60.0, 60.0, np.nan])
    secondary = np.array([61.0, 67.0, 64.0])

    fused = fuse_f0_tracks.fuse_midi_tracks(
        primary,
        secondary,
        strategy="agree-avg-else-secondary",
        agreement_tolerance=2.0,
    )

    np.testing.assert_allclose(fused, [60.5, 67.0, 64.0])


def test_resample_midi_to_target_times() -> None:
    source_times = np.array([0.00, 0.10, 0.20])
    source_midi = np.array([60.0, np.nan, 64.0])
    target_times = np.array([0.01, 0.11, 0.19])

    aligned = fuse_f0_tracks._resample_midi_to_times(
        source_times,
        source_midi,
        target_times,
        max_distance=0.04,
    )

    assert aligned[0] == 60.0
    assert np.isnan(aligned[1])
    assert aligned[2] == 64.0
