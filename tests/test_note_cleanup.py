"""note_cleanup: the three whale failure families + their guard rails.

Every scenario is a minimal synthetic replica of an ear-confirmed whale case
(times shifted to small numbers): scoop mis-split (の @5.97), same-pitch
shatter (か @18.84), phantom tails/orphans (に @66.92 / interlude 57 @240),
RMVPE octave-up outro (81 @250.9), plus the guards — a real descending
melisma must NOT merge, a supported soft tail must NOT drop, a mid-register
pYIN subharmonic must NOT repitch.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import note_cleanup as nc  # noqa: E402


# ---------- helpers ----------

def _aligned(char_spans, text="x"):
    """One aligned line whose chars sit at the given (start, end) spans."""
    return [{
        "text": text,
        "start": char_spans[0][0],
        "end": char_spans[-1][1],
        "tokens": [{"chars": [{"char": "あ", "start": s, "end": e}
                              for s, e in char_spans]}],
    }]


def _evidence(dur=20.0, *, db=0.0, rmvpe=None, pyin=None, regions=()):
    """Flat evidence at `db`/`rmvpe`/`pyin`, overridden per (lo, hi, field, value)
    region. NaN pitch = unvoiced."""
    t = np.arange(0.0, dur, 0.01)
    rms_db = np.full_like(t, db)
    rm = np.full_like(t, np.nan if rmvpe is None else float(rmvpe))
    py = np.full_like(t, np.nan if pyin is None else float(pyin))
    for lo, hi, field, value in regions:
        sel = (t >= lo) & (t < hi)
        {"db": rms_db, "rmvpe": rm, "pyin": py}[field][sel] = value
    return nc.AcousticEvidence(t, rms_db, rm, 0.01, t, py)


# ---------- failure mode 1: scoop mis-split (の @5.97: 65+67 -> 67) ----------

def test_scoop_onset_merges_into_sustain():
    aligned = _aligned([(0.9, 1.0), (1.0, 1.8)])  # 2nd char = the mora
    notes = [(1.02, 1.21, 65), (1.21, 1.79, 67)]
    out, stats, _ = nc.cleanup(notes, aligned, None)
    assert out == [(1.02, 1.79, 67)]
    assert stats["scoops_merged"] == 1


def test_mora_primary_prefers_tracker_supported_plateau():
    # whale く @232.15: attack plateau 70 (0.39s, both trackers pinned)
    # vs a slightly LONGER 69 drift (0.42s, trackers wandering 67-69).
    # Raw duration picked 69; support-weighted dominance must pick 70.
    aligned = _aligned([(1.0, 1.81)])
    notes = [(1.00, 1.39, 70), (1.39, 1.81, 69)]
    ev = _evidence(dur=3.0, db=-5.0, regions=[
        (1.00, 1.39, "rmvpe", 70.0), (1.00, 1.39, "pyin", 70.0),
        (1.39, 1.81, "rmvpe", 67.5), (1.39, 1.81, "pyin", 67.5),  # drift, not 69
    ])
    out, stats, _ = nc.cleanup(notes, aligned, ev)
    assert out == [(1.00, 1.81, 70)]
    assert stats["mora_primary_merged"] == 1


def test_close_descending_variant_is_one_mora_pitch():
    # A nearby 69->67 contour on one mora defaults to the longer 67 plateau.
    # It is not a scoop, but the broader mora-primary rule still consolidates it.
    aligned = _aligned([(1.0, 2.6)])
    notes = [(1.00, 1.24, 69), (1.24, 1.68, 67)]
    out, stats, _ = nc.cleanup(notes, aligned, None)
    assert stats["scoops_merged"] == 0
    assert out == [(1.00, 1.68, 67)]


def test_wide_melisma_is_preserved_as_special_case():
    aligned = _aligned([(1.0, 2.6)])
    notes = [(1.00, 1.40, 69), (1.40, 2.30, 64)]  # five-semitone move
    out, stats, _ = nc.cleanup(notes, aligned, None)
    assert [p for _s, _e, p in out] == [69, 64]
    assert stats["mora_primary_merged"] == 0


def test_wide_melisma_exception_requires_supported_plateaus_with_acoustics():
    aligned = _aligned([(1.0, 2.6)])
    notes = [(1.00, 1.80, 53), (1.82, 1.96, 65)]
    ev = _evidence(dur=3.0, db=-70.0, regions=[
        (1.0, 1.8, "db", -4.0), (1.0, 1.8, "rmvpe", 53.0),
        (1.0, 1.8, "pyin", 53.0),
        (1.82, 1.96, "db", -44.0),  # unsupported octave/bleed blip
    ])
    out, stats, _ = nc.cleanup(notes, aligned, ev)
    assert out == [(1.00, 1.80, 53)]
    assert stats["melisma_weak_dropped"] == 1


def test_long_first_note_is_not_a_scoop():
    aligned = _aligned([(1.0, 2.6)])
    notes = [(1.00, 1.40, 65), (1.40, 2.10, 67)]  # 0.4s > SCOOP_MAX_DUR
    out, stats, _ = nc.cleanup(notes, aligned, None)
    assert stats["scoops_merged"] == 0


def test_scoop_never_crosses_mora_boundary():
    # short 65 in char A, sustain 67 in char B: two morae, keep both bars
    aligned = _aligned([(1.0, 1.2), (1.2, 2.0)])
    notes = [(1.00, 1.19, 65), (1.21, 1.90, 67)]
    out, stats, _ = nc.cleanup(notes, aligned, None)
    assert stats["scoops_merged"] == 0
    assert len(out) == 2


# ---------- failure mode 2: same-pitch shatter (か @18.84: 67|67, 20ms gap) --

def test_same_pitch_fragments_merge_within_mora():
    aligned = _aligned([(1.0, 2.1)])
    notes = [(1.05, 1.86, 67), (1.88, 2.01, 67)]
    out, stats, _ = nc.cleanup(notes, aligned, None)
    assert out == [(1.05, 2.01, 67)]
    assert stats["same_pitch_merged"] == 1


def test_same_pitch_does_not_merge_across_moras():
    # legato same-pitch morae are SPLIT on purpose (one bar per mora);
    # the merge must not undo that
    aligned = _aligned([(1.0, 1.5), (1.5, 2.0)])
    notes = [(1.00, 1.50, 67), (1.50, 2.00, 67)]
    out, stats, _ = nc.cleanup(notes, aligned, None)
    assert len(out) == 2
    assert stats["same_pitch_merged"] == 0


def test_same_pitch_does_not_merge_over_wide_gap():
    aligned = _aligned([(1.0, 3.0)])
    notes = [(1.00, 1.50, 67), (1.80, 2.20, 67)]  # 0.3s gap = re-attack
    out, _stats, _ = nc.cleanup(notes, aligned, None)
    assert len(out) == 2


# ---------- failure mode 3: phantom tails / orphans (に @66.92, 57 @240) ----

def test_reverb_tail_without_pitch_support_drops():
    # main note loud at pitch; tail fragment has energy (reverb) but no
    # tracker at its pitch (whale に: trackers elsewhere / unvoiced)
    aligned = _aligned([(1.0, 2.9)])
    notes = [(1.00, 2.60, 65), (2.76, 2.92, 65)]
    ev = _evidence(db=-80.0, regions=[
        (1.0, 2.6, "db", -5.0), (1.0, 2.6, "rmvpe", 65.0), (1.0, 2.6, "pyin", 65.0),
        (2.6, 3.0, "db", -32.0),  # reverb tail energy, no pitch
    ])
    out, stats, _ = nc.cleanup(notes, aligned, ev)
    assert out == [(1.00, 2.60, 65)]
    assert stats["tails_dropped"] == 1


def test_supported_soft_tail_survives():
    # whale ら outro echoes @252.6/253.1: quiet but BOTH trackers agree
    aligned = _aligned([(1.0, 4.0)])
    notes = [(1.00, 1.40, 69), (2.60, 2.76, 67)]
    ev = _evidence(db=-80.0, regions=[
        (1.0, 1.4, "db", -26.0), (1.0, 1.4, "rmvpe", 69.0), (1.0, 1.4, "pyin", 69.0),
        (2.6, 2.8, "db", -44.0), (2.6, 2.8, "rmvpe", 67.0), (2.6, 2.8, "pyin", 67.0),
    ])
    out, stats, _ = nc.cleanup(notes, aligned, ev)
    assert len(out) == 2
    assert stats["tails_dropped"] == 0


def test_dead_silent_orphan_drops_despite_tracker_blip():
    # whale 240.03: RMVPE reports 57 inside a -89 dB interlude — f0>0 alone
    # must not keep a note alive
    aligned = _aligned([(1.0, 1.5)])
    notes = [(1.00, 1.50, 65), (5.00, 5.16, 57)]
    ev = _evidence(dur=8.0, db=-85.0, regions=[
        (1.0, 1.5, "db", -5.0), (1.0, 1.5, "rmvpe", 65.0), (1.0, 1.5, "pyin", 65.0),
        (5.0, 5.2, "rmvpe", 57.0),
    ])
    out, stats, _ = nc.cleanup(notes, aligned, ev)
    assert out == [(1.00, 1.50, 65)]
    assert stats["orphans_dropped"] == 1


def test_supported_orphan_still_drops_because_lyrics_are_truth():
    aligned = _aligned([(1.0, 1.5)])
    notes = [(1.0, 1.5, 65), (2.0, 2.3, 69)]
    ev = _evidence(dur=3.0, db=-5.0, rmvpe=69.0, pyin=69.0, regions=[
        (1.0, 1.5, "rmvpe", 65.0), (1.0, 1.5, "pyin", 65.0),
    ])
    out, stats, _ = nc.cleanup(notes, aligned, ev)
    assert out == [(1.0, 1.5, 65)]
    assert stats["orphans_dropped"] == 1


def test_no_acoustics_means_no_drops():
    # the destructive rule must not fire on geometry alone
    aligned = _aligned([(1.0, 1.5)])
    notes = [(1.00, 1.50, 65), (5.00, 5.16, 57)]
    out, stats, _ = nc.cleanup(notes, aligned, None)
    assert len(out) == 2
    assert stats["orphans_dropped"] == stats["tails_dropped"] == 0


# ---------- octave fix (81 @250.9 -> 69) and its guard ----------

def test_rmvpe_octave_up_outlier_repitched():
    aligned = _aligned([(i, i + 0.4) for i in range(1, 11)])
    notes = [(float(i), i + 0.4, 67) for i in range(1, 10)]
    notes.append((10.0, 10.4, 81))  # way above the song register
    ev = _evidence(dur=12.0, db=-20.0, rmvpe=67.0, pyin=67.0, regions=[
        (10.0, 10.4, "rmvpe", 81.0), (10.0, 10.4, "pyin", 69.0),
    ])
    out, stats, _ = nc.cleanup(notes, aligned, ev)
    assert stats["octave_fixed"] == 1
    assert out[-1][2] == 69


def test_pyin_subharmonic_does_not_repitch_midrange():
    # whale 65.1-66.8s: GAME+RMVPE say 65, pYIN reads 53 (subharmonic).
    # Transcription outranks a single tracker — the note must stay 65.
    aligned = _aligned([(i, i + 0.4) for i in range(1, 11)])
    notes = [(float(i), i + 0.4, 65) for i in range(1, 11)]
    ev = _evidence(dur=12.0, db=-10.0, rmvpe=65.0, pyin=53.0)
    out, stats, _ = nc.cleanup(notes, aligned, ev)
    assert stats["octave_fixed"] == 0
    assert all(p == 65 for _s, _e, p in out)


def test_local_octave_island_repitches_inside_global_range():
    # Whale 僕ら波を待ってる: the bad 70 is not a global high outlier, but
    # 60 -> 70 -> 57 becomes locally smooth 60 -> 58 -> 57 after -12 and
    # pYIN explicitly supports 58.
    aligned = _aligned([(1.0, 1.3), (1.3, 1.7), (1.7, 2.1)])
    notes = [(1.00, 1.30, 60), (1.30, 1.70, 70), (1.70, 2.10, 57)]
    ev = _evidence(dur=3.0, db=-5.0, rmvpe=60.0, pyin=60.0, regions=[
        (1.30, 1.70, "rmvpe", 70.0), (1.30, 1.70, "pyin", 58.0),
        (1.70, 2.10, "rmvpe", 57.0), (1.70, 2.10, "pyin", 57.0),
    ])
    out, stats, _ = nc.cleanup(notes, aligned, ev)
    assert stats["octave_fixed"] == 1
    assert [p for _s, _e, p in out] == [60, 58, 57]


# ---------- true mora ownership + one-pitch default ----------

def test_multimora_kanji_partitions_pitch_coherently():
    # 静（しず） is one surface char but two reading morae.  60 belongs to
    # し; the nearby 67->69 approach belongs to ず and must become one 69.
    aligned = [{
        "text": "静",
        "start": 1.0,
        "end": 2.2,
        "tokens": [{
            "surface": "静",
            "reading": "しず",
            "kana_only": False,
            "chars": [{"char": "静", "start": 1.0, "end": 2.2}],
        }],
    }]
    notes = [(1.00, 1.24, 60), (1.24, 1.41, 67), (1.41, 2.12, 69)]
    out, stats, _ = nc.cleanup(notes, aligned, None)
    assert [p for _s, _e, p in out] == [60, 69]
    assert out[1][:2] == (1.24, 2.12)
    assert stats["scoops_merged"] == 1


def test_close_pitch_mora_defaults_to_one_primary_pitch():
    # A close 69->67->69 contour is intonation by default, not three score
    # notes.  The longest plateau (67) owns the mora.
    aligned = _aligned([(1.0, 2.5)])
    notes = [(1.00, 1.24, 69), (1.24, 2.10, 67), (2.10, 2.40, 69)]
    out, stats, _ = nc.cleanup(notes, aligned, None)
    assert out == [(1.00, 2.40, 67)]
    assert stats["mora_primary_merged"] == 2


def test_different_pitch_spill_mostly_after_mora_drops_without_extending():
    # Whale 波に横たえながら: the real 65 plateau ends at 2.20; a 68 event
    # begins just before the aligned mora end but mostly lives beyond it.
    # It is accompaniment/tail leakage, not a second pitch for ら.
    aligned = _aligned([(1.0, 2.6)])
    notes = [(1.00, 2.20, 65), (2.27, 3.38, 68)]
    out, stats, _ = nc.cleanup(notes, aligned, None)
    assert out == [(1.00, 2.20, 65)]
    assert stats["mora_spills_dropped"] == 1


# ---------- recut propagation to the aligned source ----------

def test_apply_recut_to_aligned_rewrites_chars_and_line():
    aligned = _aligned([(231.2, 232.0), (234.4, 248.1), (263.9, 264.2)])
    patch = [{"lyric_recut": 231.22,
              "chars": [[231.3, 232.1], [234.3, 235.4], [250.6, 253.3]]}]
    applied = nc.apply_recut_to_aligned(aligned, patch)
    assert applied == 1
    chars = [ch for tok in aligned[0]["tokens"] for ch in tok["chars"]]
    assert [ch["start"] for ch in chars] == [231.3, 234.3, 250.6]
    assert aligned[0]["start"] == 231.3 and aligned[0]["end"] == 253.3


def test_apply_recut_char_count_mismatch_raises():
    aligned = _aligned([(10.0, 11.0), (11.0, 12.0)])
    patch = [{"lyric_recut": 10.0, "chars": [[10.0, 11.0]]}]
    with pytest.raises(ValueError):
        nc.apply_recut_to_aligned(aligned, patch)


# ---------- mora assignment ----------

def test_assign_moras_unpadded_overlap_wins():
    # whale の: the scoop note must belong to の, not tie-break into た
    chars = [(5.681, 5.841), (5.941, 6.742)]
    moras = nc.assign_moras([(5.97, 6.16, 65), (6.16, 6.74, 67)], chars)
    assert moras == [1, 1]


def test_assign_moras_orphan_vs_near_tail():
    chars = [(1.0, 2.0)]
    near = (2.1, 2.3, 60)   # 0.1s past the char: its tail
    far = (3.0, 3.2, 60)    # 1.0s past: orphan
    assert nc.assign_moras([near, far], chars) == [0, -1]
