import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import score_note_postfix as snp  # noqa: E402

from karaoke_jp.pitch_eval import F0Track  # noqa: E402


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _track(midi_curve: list[float], hop: float = 0.01) -> F0Track:
    times = np.arange(len(midi_curve)) * hop
    f0 = np.array([_hz(m) if m > 0 else 0.0 for m in midi_curve])
    return F0Track(times=times, f0_hz=f0)


def test_refine_boundaries_moves_joint_to_f0_crossing() -> None:
    # F0 sits on pitch 60 until 0.30s, then on 67; the joint is misplaced at 0.40s.
    curve = [60.0] * 30 + [67.0] * 50
    track = _track(curve)
    notes = [(0.0, 0.40, 60), (0.40, 0.80, 67)]
    fixed, moved = snp.refine_boundaries(notes, track)
    assert moved == 1
    assert abs(fixed[0][1] - 0.30) < 0.05
    assert fixed[0][1] == fixed[1][0]


def test_absorb_shakuri_folds_rising_onset_into_target() -> None:
    # short onset note rises from 65 toward 67, then a long 67 note follows
    curve = [65.0 + 2.0 * (i / 14) for i in range(15)] + [67.0] * 60
    track = _track(curve)
    notes = [(0.0, 0.15, 65), (0.15, 0.75, 67)]
    fixed, folded = snp.absorb_shakuri(notes, track, [(0.0, 1.0)])
    assert folded == 1
    # the onset is re-labeled and then merged into one 67 bar
    merged = snp.merge_same_pitch(fixed)
    assert merged == [(0.0, 0.75, 67)]


def test_extend_sustains_follows_held_pitch() -> None:
    # note ends at 0.30s but F0 keeps holding pitch 63 until 0.60s
    curve = [63.0] * 60 + [0.0] * 20
    track = _track(curve)
    notes = [(0.0, 0.30, 63)]
    fixed, extended = snp.extend_sustains(notes, track)
    assert extended == 1
    assert abs(fixed[0][1] - 0.60) < 0.06


def test_fill_missing_morae_adds_supported_note() -> None:
    curve = [0.0] * 30 + [70.0] * 40 + [0.0] * 10
    track = _track(curve)
    windows = [(0.30, 0.70, "あ", 0)]
    notes, added = snp.fill_missing_morae([], windows, track)
    assert added == 1
    assert notes[0][2] == 70


def test_fill_missing_morae_avoids_interior_note_overlap() -> None:
    # an existing note ENTIRELY INSIDE the window must split it; the filled
    # note goes into the largest free gap instead of overlapping
    curve = [0.0] * 30 + [70.0] * 40 + [0.0] * 10
    track = _track(curve)
    windows = [(0.30, 0.70, "あ", 0)]
    existing = [(0.45, 0.50, 65)]
    notes, added = snp.fill_missing_morae(existing, windows, track)
    assert added == 1
    new = [n for n in notes if n not in existing]
    assert len(new) == 1
    s, e, _p = new[0]
    assert abs(s - 0.50) < 1e-6 and abs(e - 0.70) < 1e-6
    # no pair of notes may overlap
    ordered = sorted(notes)
    assert all(b[0] >= a[1] - 1e-9 for a, b in zip(ordered, ordered[1:]))


def test_refine_boundaries_skips_when_f0_never_crosses() -> None:
    # F0 stays on pitch a through the whole search window: no crossing
    # evidence, the joint must NOT be moved.
    curve = [60.0] * 80
    track = _track(curve)
    notes = [(0.0, 0.40, 60), (0.40, 0.80, 67)]
    fixed, moved = snp.refine_boundaries(notes, track)
    assert moved == 0
    assert fixed == notes


def test_refine_boundaries_snaps_early_when_crossing_precedes_window() -> None:
    # F0 is already on b's side at the window's first frame: deliberate
    # early snap toward the window edge (karaoke bars lead the portamento).
    curve = [67.0] * 80
    track = _track(curve)
    notes = [(0.0, 0.40, 60), (0.40, 0.80, 67)]
    fixed, moved = snp.refine_boundaries(notes, track)
    assert moved == 1
    assert fixed[0][1] == fixed[1][0]
    assert 0.04 <= fixed[1][0] < 0.40


def test_chroma_snap_only_moves_toward_stronger_pitch_class() -> None:
    # sung median is 30 cents above pitch 69; pitch class of 70 dominates
    curve = [69.3] * 50
    track = _track(curve)
    weights = np.full(12, 0.1)
    weights[70 % 12] = 1.0
    fixed, snapped = snp.chroma_snap([(0.0, 0.5, 69)], track, weights)
    assert snapped == 1 and fixed[0][2] == 70
    # without pitch-class support it must stay put
    flat = np.full(12, 0.5)
    fixed2, snapped2 = snp.chroma_snap([(0.0, 0.5, 69)], track, flat)
    assert snapped2 == 0 and fixed2[0][2] == 69


def test_classify_row_matches_scalar_reference() -> None:
    import importlib.util

    rlk_path = ROOT / "tmp" / "reference" / "chidori" / "sheet_video" / "read_lit_keys.py"
    if not rlk_path.exists():
        import pytest

        pytest.skip("targeted detector not present")
    spec = importlib.util.spec_from_file_location("rlk", rlk_path)
    rlk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rlk)

    rng = np.random.default_rng(0)
    row = rng.integers(0, 256, size=(512, 3), dtype=np.uint8)
    labels = rlk.classify_row(row)
    want = {None: 0, "R": 1, "L": 2}
    for x in range(row.shape[0]):
        assert labels[x] == want[rlk.classify_color(*row[x][:3])], x


def test_capture_tail_falls_adds_plateau_note() -> None:
    # a phrase tail sings 65, then F0 falls to a stable 63 plateau in the
    # following lyric gap: a NEW note must cover the plateau.
    curve = [65.0] * 30 + [63.0] * 50
    track = _track(curve)
    fixed, added = snp.capture_tail_falls([(0.0, 0.30, 65)], track)
    assert added == 1
    tail = fixed[-1]
    assert tail[2] == 63
    assert abs(tail[0] - 0.30) < 0.05 and tail[1] > 0.70


def test_capture_tail_falls_requires_a_lyric_gap() -> None:
    # back-to-back notes (no gap) must not trigger the tail capture
    curve = [65.0] * 30 + [63.0] * 50
    track = _track(curve)
    fixed, added = snp.capture_tail_falls(
        [(0.0, 0.30, 65), (0.40, 0.80, 63)], track
    )
    assert added == 0


def test_melody_union_fills_gaps_without_overlap() -> None:
    import melody_union as mu

    primary = [(1.0, 2.0, 70), (3.0, 4.0, 67)]
    fallback = [(0.0, 5.0, 65)]
    merged = mu.union(primary, fallback)
    # primary survives untouched
    assert all(n in merged for n in primary)
    # fallback pieces fill the three gaps, guarded, no overlaps
    ordered = sorted(merged)
    assert all(b[0] >= a[1] - 1e-9 for a, b in zip(ordered, ordered[1:]))
    pieces = [n for n in merged if n not in primary]
    assert len(pieces) == 3 and all(p == 65 for _s, _e, p in pieces)
    # a too-small gap must not produce a sliver
    merged2 = mu.union([(0.0, 1.0, 70), (1.05, 2.0, 67)], [(0.0, 2.0, 65)])
    assert len(merged2) == 2
