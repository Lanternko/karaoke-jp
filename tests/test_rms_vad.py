"""RMS-VAD pre-segmentation core — deterministic, no audio I/O, no GPU.

Guards the Cut & Merge invariants that make the segments safe for Whisper:
phrases separated by short silences fuse, long spans get cut under the 30 s
window at a low-energy valley, and known-bad windows stay covered.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from rms_vad_segments import (  # noqa: E402
    cut_long,
    merge_runs,
    pad_and_clamp,
    segment_audio,
    voiced_runs,
)


def _tone(sr: int, *spans: tuple[float, float], total: float) -> np.ndarray:
    """Build a waveform that is a 220 Hz tone inside each (start, end) span and
    silence elsewhere — a controllable stand-in for sung phrases."""
    y = np.zeros(int(total * sr), dtype=np.float32)
    t = np.arange(len(y)) / sr
    for s, e in spans:
        mask = (t >= s) & (t < e)
        y[mask] = 0.5 * np.sin(2 * np.pi * 220 * t[mask])
    return y


def test_voiced_runs_recovers_two_phrases() -> None:
    sr = 16000
    y = _tone(sr, (1.0, 3.0), (5.0, 7.0), total=8.0)
    segs = segment_audio(y, sr, pad=0.0, merge_gap=0.3)
    assert len(segs) == 2
    s0, e0 = segs[0]
    # The 128 ms RMS frame smears each edge by up to ~one frame; onsets bleed
    # slightly early (harmless — we pad anyway), so allow one frame of slack.
    assert s0 == pytest.approx(1.0, abs=0.15)
    assert e0 == pytest.approx(3.0, abs=0.15)


def test_merge_gap_fuses_close_phrases() -> None:
    sr = 16000
    # two tones 0.4 s apart → merged at merge_gap=0.6, split at merge_gap=0.2
    y = _tone(sr, (1.0, 2.0), (2.4, 3.4), total=5.0)
    assert len(segment_audio(y, sr, pad=0.0, merge_gap=0.6)) == 1
    assert len(segment_audio(y, sr, pad=0.0, merge_gap=0.2)) == 2


def test_short_blip_dropped() -> None:
    sr = 16000
    y = _tone(sr, (1.0, 1.05), (3.0, 5.0), total=6.0)  # 50 ms blip + real phrase
    # The blip smears to ~0.16 s under the 128 ms frame, so threshold above it.
    segs = segment_audio(y, sr, pad=0.0, min_voiced_dur=0.25)
    assert len(segs) == 1
    assert segs[0][0] == pytest.approx(3.0, abs=0.15)


def test_cut_long_respects_max_len() -> None:
    # one 40 s run must be cut into pieces each <= max_len
    times = np.arange(0, 40, 0.032)
    rms_db = np.full(len(times), -5.0)
    pieces = cut_long([(0.0, 40.0)], rms_db, times, max_len=28.0, min_cut_frac=0.4)
    assert all(e - s <= 28.0 + 1e-6 for s, e in pieces)
    assert pieces[0][0] == 0.0 and pieces[-1][1] == pytest.approx(40.0, abs=0.05)
    # pieces are contiguous (no gaps, no overlaps)
    for (_, e), (s, _) in zip(pieces, pieces[1:]):
        assert s == pytest.approx(e, abs=1e-6)


def test_cut_prefers_low_energy_valley() -> None:
    times = np.arange(0, 40, 0.032)
    rms_db = np.full(len(times), -5.0)
    valley_t = 20.0
    rms_db[int(valley_t / 0.032)] = -60.0  # deep valley within the cut window
    pieces = cut_long([(0.0, 40.0)], rms_db, times, max_len=28.0, min_cut_frac=0.4)
    assert pieces[0][1] == pytest.approx(valley_t, abs=0.05)


def test_pad_merges_overlaps() -> None:
    runs = [(1.0, 2.0), (2.2, 3.0)]
    padded = pad_and_clamp(runs, pad=0.25, duration=5.0)
    assert padded == [(0.75, 3.25)]  # padding closes the 0.2 s gap


def test_empty_audio_is_safe() -> None:
    assert merge_runs([], 0.5) == []
    assert voiced_runs(np.array([-80.0, -80.0]), np.array([0.0, 0.032]), 40.0, 0.1) == []
