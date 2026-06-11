import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import make_display_grid as mdg  # noqa: E402

QUARTER = 0.5
GAP = 0.25 * QUARTER
PGAP = 1.25 * QUARTER
LEAD = 0.5 * QUARTER
SPAN = 16 * QUARTER


def _width_fn(qnotes):
    def width(ph):
        return sum(qnotes[i][1] - qnotes[i][0] for i in ph) + GAP * (len(ph) - 1)
    return width


# ---------- split_oversized ----------

def test_split_prefers_line_boundary_over_largest_gap() -> None:
    # 6 notes, 2 lyric lines (3+3). The largest acoustic gap (0.30s) sits
    # MID-LINE between notes 1-2; the line boundary gap (0.10s) is smaller.
    notes = [
        (0.0, 1.0, 60), (1.05, 2.0, 62), (2.30, 3.3, 64),   # line 0 (gap 0.30 inside)
        (3.40, 4.4, 65), (4.45, 5.4, 67), (5.45, 6.4, 69),  # line 1
    ]
    qnotes = [(s, s + 2 * QUARTER, p) for s, e, p in notes]  # width 6.25q each part
    width = _width_fn(qnotes)
    line_of = [0, 0, 0, 1, 1, 1]
    budget = 8 * QUARTER  # whole cluster (13.25q) over budget, halves fit
    parts = mdg.split_oversized(list(range(6)), notes, width, budget, line_of)
    assert parts == [[0, 1, 2], [3, 4, 5]]


def test_split_falls_back_to_largest_gap_without_lines() -> None:
    notes = [
        (0.0, 1.0, 60), (1.05, 2.0, 62), (2.30, 3.3, 64),
        (3.40, 4.4, 65), (4.45, 5.4, 67), (5.45, 6.4, 69),
    ]
    qnotes = [(s, s + 2 * QUARTER, p) for s, e, p in notes]
    width = _width_fn(qnotes)
    line_of = [0] * 6  # no line info
    budget = 8 * QUARTER
    parts = mdg.split_oversized(list(range(6)), notes, width, budget, line_of)
    # largest gap is between notes 1 and 2 (0.30s)
    assert parts[0] == [0, 1]


def test_split_oversized_single_line_recurses_to_fit() -> None:
    notes = [(i * 1.0, i * 1.0 + 0.9, 60) for i in range(8)]
    qnotes = [(s, s + 2 * QUARTER, p) for s, e, p in notes]
    width = _width_fn(qnotes)
    parts = mdg.split_oversized(list(range(8)), notes, width, 5 * QUARTER, [0] * 8)
    assert all(width(ph) <= 5 * QUARTER for ph in parts)
    assert sorted(i for ph in parts for i in ph) == list(range(8))


# ---------- assign_lines ----------

def test_assign_lines_tolerates_early_onsets() -> None:
    notes = [(9.95, 10.5, 60), (10.6, 11.0, 62)]  # first starts 0.05 early
    assert mdg.assign_lines(notes, [0.0, 10.0]) == [1, 1]


def test_assign_lines_without_lines_is_all_zero() -> None:
    assert mdg.assign_lines([(0.0, 1.0, 60)], []) == [0]


# ---------- pack_pages ----------

def test_pack_moves_whole_phrase_to_next_page() -> None:
    qnotes = [(0, 7 * QUARTER, 60), (8, 8 + 7 * QUARTER, 62), (16, 16 + 7 * QUARTER, 64)]
    width = _width_fn(qnotes)
    pages = mdg.pack_pages([[0], [1], [2]], width, span=SPAN, lead=LEAD,
                           pgap=PGAP, quarter=QUARTER)
    # 7q + pgap 1.25q + 7q = 15.25q fits (<= 15.75q); third phrase moves whole
    assert [len(pg) for pg in pages] == [2, 1]


# ---------- layout_pages ----------

def _layout_simple(notes, qnotes, pages, **kw):
    return mdg.layout_pages(pages, notes, qnotes, span=SPAN, lead=LEAD,
                            gap=GAP, pgap=PGAP, quarter=QUARTER, **kw)


def test_layout_keeps_fixed_gaps_and_no_overlap() -> None:
    notes = [(20.0, 20.4, 60), (20.4, 20.8, 62), (20.8, 21.2, 64)]
    qnotes = mdg.quantize(notes, quarter=QUARTER)
    disp, _r, _d = _layout_simple(notes, qnotes, [[[0, 1, 2]]])
    for a, b in zip(disp, disp[1:]):
        assert b[0] - a[1] == pytest.approx(GAP)


def test_layout_warp_strictly_monotonic() -> None:
    notes = [(20.0, 20.4, 60), (20.5, 21.0, 62), (40.0, 41.0, 64)]
    qnotes = mdg.quantize(notes, quarter=QUARTER)
    _disp, real, disp = _layout_simple(notes, qnotes, [[[0, 1]], [[2]]])
    assert all(b > a for a, b in zip(real, real[1:]))
    assert all(b > a for a, b in zip(disp, disp[1:]))


def test_layout_count_in_parks_cursor_after_long_rest() -> None:
    # page 1 ends at 21s; page 2 first note at 40s -> long rest
    notes = [(20.0, 21.0, 60), (40.0, 41.0, 64)]
    qnotes = mdg.quantize(notes, quarter=QUARTER)
    _disp, real, disp = _layout_simple(notes, qnotes, [[[0]], [[1]]],
                                       count_in_quarters=4.0, flip_delay=0.5)
    import numpy as np
    warp = lambda t: float(np.interp(t, real, disp))  # noqa: E731
    page2 = SPAN
    # shortly after the flip the display is already on page 2 ...
    assert warp(21.5) >= page2
    # ... and parked at its left edge until the count-in starts (40 - 4q = 38)
    assert warp(37.9) == pytest.approx(page2, abs=0.01)
    # during the count-in the cursor sweeps the lead toward the first note
    assert page2 + 0.001 < warp(39.0) < page2 + LEAD
    assert warp(40.0) == pytest.approx(page2 + LEAD, abs=1e-6)


def test_layout_no_count_in_anchor_when_rest_is_short() -> None:
    # first note close to t=0 (no intro count-in), short rest between pages
    notes = [(1.0, 2.0, 60), (3.0, 4.0, 64)]
    qnotes = mdg.quantize(notes, quarter=QUARTER)
    _disp, real, _d = _layout_simple(notes, qnotes, [[[0]], [[1]]])
    # only 0.0 + 2 anchors per note + closing anchor: no flip/park pair
    assert len(real) == 1 + 4 + 1


# ---------- renderer BPM contract ----------

def test_bpm_2dp_survives_renderer_roundtrip() -> None:
    # MID2BAR's mid2csv recovers BPM as round(60e6 / tempo_us, 2); the grid
    # must emit a tempo whose roundtrip lands on the SAME 2dp BPM, or page
    # boundaries drift ~5ms by mid-song (v9 right-edge-parked-cursor bug)
    import mido
    for bpm in (117.4538, 117.45, 120.0, 89.731, 203.99):
        b2 = round(bpm, 2)
        tempo_us = mido.bpm2tempo(b2)
        assert round(60 / (tempo_us * 1e-6), 2) == b2


# ---------- flat bar skin contract ----------

def test_flat_skin_respects_padding_contract(tmp_path) -> None:
    PIL = pytest.importorskip("PIL")  # noqa: F841
    from click.testing import CliRunner
    import make_flat_bar_skin as skin

    res = CliRunner().invoke(skin.main, ["--out-dir", str(tmp_path)])
    assert res.exit_code == 0, res.output

    from PIL import Image
    left = Image.open(tmp_path / "back_left.png")
    right = Image.open(tmp_path / "back_right.png")
    pad, mid_row = skin.PADDING, (skin.BODY_TOP + skin.BODY_BOTTOM) // 2
    # outer PADDING columns must be fully transparent (renderer blits the cap
    # at x - PADDING and anything opaque there bleeds outside the note span)
    assert all(left.getpixel((x, mid_row))[3] == 0 for x in range(pad - 1))
    assert all(right.getpixel((skin.SEG_W - 1 - x, mid_row))[3] == 0
               for x in range(pad - 1))
    # body starts right at the padding edge
    assert left.getpixel((pad + skin.RADIUS, mid_row))[3] == 255
    assert right.getpixel((skin.SEG_W - 1 - pad - skin.RADIUS, mid_row))[3] == 255
