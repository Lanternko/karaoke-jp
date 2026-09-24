from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import midi_timing


def _kana_line() -> dict:
    return {
        "text": "あい",
        "start": 10.0,
        "end": 10.6,
        "tokens": [
            {
                "surface": "あい",
                "reading": "あい",
                "kana_only": True,
                "start": 10.0,
                "end": 10.6,
                "chars": [
                    {"char": "あ", "start": 10.0, "end": 10.2},
                    {"char": "い", "start": 10.4, "end": 10.6},
                ],
            }
        ],
    }


def test_allocate_notes_splits_single_note_across_multiple_morae() -> None:
    spans, last_idx = midi_timing._allocate_notes(
        [(1.0, 2.0, 60)],
        [1.00, 1.10, 1.20],
    )

    assert last_idx == 0
    assert len(spans) == 3
    assert spans[0][0] == pytest.approx(1.0)
    assert spans[0][1] == pytest.approx(4 / 3)
    assert spans[1][0] == pytest.approx(4 / 3)
    assert spans[1][1] == pytest.approx(5 / 3)
    assert spans[2][0] == pytest.approx(5 / 3)
    assert spans[2][1] == pytest.approx(2.0)


def test_apply_char_timing_skips_trailing_punctuation_when_matching() -> None:
    lines = [
        {
            "text": "声に、",
            "start": 1.0,
            "end": 2.0,
            "tokens": [
                {
                    "surface": "声に、",
                    "reading": "こえに",
                    "kana_only": False,
                    "start": 1.0,
                    "end": 2.0,
                    "chars": [
                        {"char": "声", "start": 1.0, "end": 1.2},
                        {"char": "に", "start": 1.2, "end": 1.4},
                        {"char": "、", "start": 1.4, "end": 1.5},
                    ],
                }
            ],
        }
    ]

    updated, kept = midi_timing.apply_char_timing(
        lines,
        [(1.0, 1.2, 60), (1.3, 1.5, 62)],
    )

    assert (updated, kept) == (1, 0)
    chars = lines[0]["tokens"][0]["chars"]
    assert chars[0]["start"] == pytest.approx(1.0)
    assert chars[0]["end"] == pytest.approx(1.2)
    assert chars[1]["start"] == pytest.approx(1.3)
    assert chars[1]["end"] == pytest.approx(1.5)
    assert chars[2]["start"] == pytest.approx(1.5)
    assert chars[2]["end"] == pytest.approx(1.5)
    assert lines[0]["start"] == pytest.approx(1.0)
    assert lines[0]["end"] == pytest.approx(1.5)


def test_apply_mora_timing_retimes_leading_and_trailing_quotes() -> None:
    lines = [
        {
            "text": "「あ」",
            "start": 10.0,
            "end": 11.0,
            "tokens": [
                {
                    "surface": "「",
                    "reading": "",
                    "kana_only": False,
                    "start": 10.0,
                    "end": 10.1,
                    "chars": [{"char": "「", "start": 10.0, "end": 10.1}],
                },
                {
                    "surface": "あ",
                    "reading": "あ",
                    "kana_only": True,
                    "start": 10.1,
                    "end": 10.9,
                    "chars": [{"char": "あ", "start": 10.1, "end": 10.9}],
                },
                {
                    "surface": "」",
                    "reading": "",
                    "kana_only": False,
                    "start": 10.9,
                    "end": 11.0,
                    "chars": [{"char": "」", "start": 10.9, "end": 11.0}],
                },
            ],
        }
    ]

    updated, morae, used = midi_timing.apply_mora_timing(
        lines,
        [(2.0, 2.5, 60)],
        margin=20.0,
    )

    assert (updated, morae, used) == (1, 1, 1)
    open_quote = lines[0]["tokens"][0]["chars"][0]
    sung = lines[0]["tokens"][1]["chars"][0]
    close_quote = lines[0]["tokens"][2]["chars"][0]
    assert open_quote["start"] == pytest.approx(2.0)
    assert open_quote["end"] == pytest.approx(2.0)
    assert sung["start"] == pytest.approx(2.0)
    assert sung["end"] == pytest.approx(2.5)
    assert close_quote["start"] == pytest.approx(2.5)
    assert close_quote["end"] == pytest.approx(2.5)
    assert lines[0]["tokens"][0]["start"] == pytest.approx(2.0)
    assert lines[0]["tokens"][0]["end"] == pytest.approx(2.0)
    assert lines[0]["tokens"][2]["start"] == pytest.approx(2.5)
    assert lines[0]["tokens"][2]["end"] == pytest.approx(2.5)


def test_first_mora_min_delay_is_opt_in_boundary_prior() -> None:
    notes = [(9.95, 10.25, 60), (10.30, 10.70, 62)]
    default_lines = [_kana_line()]
    gated_lines = [copy.deepcopy(_kana_line())]

    midi_timing.apply_mora_timing(default_lines, notes, margin=0.4)
    midi_timing.apply_mora_timing(
        gated_lines,
        notes,
        margin=0.4,
        first_mora_min_delay=0.0,
    )

    default_chars = default_lines[0]["tokens"][0]["chars"]
    gated_chars = gated_lines[0]["tokens"][0]["chars"]
    assert default_chars[0]["start"] == pytest.approx(9.95)
    assert gated_chars[0]["start"] == pytest.approx(10.30)
    assert gated_lines[0]["start"] == pytest.approx(10.30)


def test_first_mora_min_delay_guard_keeps_nearby_legal_lead_note() -> None:
    notes = [(9.95, 10.25, 60), (10.30, 10.70, 62)]
    lines = [_kana_line()]

    midi_timing.apply_mora_timing(
        lines,
        notes,
        margin=0.4,
        first_mora_min_delay=0.0,
        first_mora_gate_lead_tolerance=0.10,
    )

    chars = lines[0]["tokens"][0]["chars"]
    assert chars[0]["start"] == pytest.approx(9.95)
    assert lines[0]["start"] == pytest.approx(9.95)


def test_first_mora_min_delay_guard_rejects_implausibly_early_note() -> None:
    notes = [(9.70, 10.00, 60), (10.30, 10.70, 62)]
    lines = [_kana_line()]

    midi_timing.apply_mora_timing(
        lines,
        notes,
        margin=0.4,
        first_mora_min_delay=0.0,
        first_mora_gate_lead_tolerance=0.10,
    )

    chars = lines[0]["tokens"][0]["chars"]
    assert chars[0]["start"] == pytest.approx(10.30)
    assert lines[0]["start"] == pytest.approx(10.30)


def test_absorb_trailing_notes_extends_final_mora_as_melisma() -> None:
    lines = [
        {
            "text": "あ",
            "start": 10.0,
            "end": 10.4,
            "tokens": [
                {
                    "surface": "あ",
                    "reading": "あ",
                    "kana_only": True,
                    "start": 10.0,
                    "end": 10.4,
                    "chars": [{"char": "あ", "start": 10.0, "end": 10.4}],
                }
            ],
        }
    ]

    updated, morae, used = midi_timing.apply_mora_timing(
        lines,
        [(10.0, 10.3, 60), (10.3, 10.8, 62)],
        margin=0.5,
        absorb_trailing_notes=True,
    )

    sung = lines[0]["tokens"][0]["chars"][0]
    assert (updated, morae, used) == (1, 1, 2)
    assert sung["start"] == pytest.approx(10.0)
    assert sung["end"] == pytest.approx(10.8)
    assert lines[0]["end"] == pytest.approx(10.8)


def test_next_line_hint_guard_keeps_boundary_note_for_next_line() -> None:
    lines = [
        {
            "text": "あい",
            "start": 9.0,
            "end": 10.0,
            "tokens": [
                {
                    "surface": "あい",
                    "reading": "あい",
                    "kana_only": True,
                    "start": 9.0,
                    "end": 10.0,
                    "chars": [
                        {"char": "あ", "start": 9.0, "end": 9.5},
                        {"char": "い", "start": 9.5, "end": 10.0},
                    ],
                }
            ],
        },
        {
            "text": "う",
            "start": 10.05,
            "end": 10.7,
            "tokens": [
                {
                    "surface": "う",
                    "reading": "う",
                    "kana_only": True,
                    "start": 10.05,
                    "end": 10.7,
                    "chars": [{"char": "う", "start": 10.05, "end": 10.7}],
                }
            ],
        },
    ]

    midi_timing.apply_mora_timing(
        lines,
        [(9.0, 9.3, 60), (10.3, 10.6, 62)],
        margin=0.4,
        next_line_hint_guard=0.20,
    )

    first_line_chars = lines[0]["tokens"][0]["chars"]
    next_line_char = lines[1]["tokens"][0]["chars"][0]
    assert first_line_chars[-1]["end"] == pytest.approx(9.3)
    assert next_line_char["start"] == pytest.approx(10.3)
    assert lines[1]["start"] == pytest.approx(10.3)


def test_next_line_hint_guard_min_delay_keeps_safe_boundary_unchanged() -> None:
    guarded = [
        {
            "text": "あい",
            "start": 9.0,
            "end": 10.0,
            "tokens": [
                {
                    "surface": "あい",
                    "reading": "あい",
                    "kana_only": True,
                    "start": 9.0,
                    "end": 10.0,
                    "chars": [
                        {"char": "あ", "start": 9.0, "end": 9.5},
                        {"char": "い", "start": 9.5, "end": 10.0},
                    ],
                }
            ],
        },
        {
            "text": "う",
            "start": 10.05,
            "end": 10.7,
            "tokens": [
                {
                    "surface": "う",
                    "reading": "う",
                    "kana_only": True,
                    "start": 10.05,
                    "end": 10.7,
                    "chars": [{"char": "う", "start": 10.05, "end": 10.7}],
                }
            ],
        },
    ]
    baseline = copy.deepcopy(guarded)
    notes = [(9.0, 9.3, 60), (10.3, 10.6, 62)]

    midi_timing.apply_mora_timing(baseline, notes, margin=0.4)
    midi_timing.apply_mora_timing(
        guarded,
        notes,
        margin=0.4,
        next_line_hint_guard=0.20,
        next_line_hint_min_start_delay=2.0,
    )

    assert guarded == baseline


def test_dp_allocator_can_group_multiple_notes_under_one_mora() -> None:
    spans, last_idx = midi_timing._allocate_notes_dp(
        [(10.0, 10.2, 60), (10.2, 10.5, 62), (10.8, 11.0, 64)],
        [10.0, 10.8],
        skip_penalty=0.30,
        extra_note_penalty=0.03,
        max_notes_per_mora=3,
    )

    assert last_idx == 2
    assert spans == pytest.approx([(10.0, 10.5), (10.8, 11.0)])


def test_apply_mora_timing_dp_allocator_is_opt_in() -> None:
    lines = [
        {
            "text": "あ",
            "start": 10.0,
            "end": 10.4,
            "tokens": [
                {
                    "surface": "あ",
                    "reading": "あ",
                    "kana_only": True,
                    "start": 10.0,
                    "end": 10.4,
                    "chars": [{"char": "あ", "start": 10.0, "end": 10.4}],
                }
            ],
        }
    ]

    midi_timing.apply_mora_timing(
        lines,
        [(10.0, 10.2, 60), (10.2, 10.5, 62)],
        margin=0.5,
        allocator="dp",
        dp_skip_penalty=0.30,
        dp_extra_note_penalty=0.03,
    )

    sung = lines[0]["tokens"][0]["chars"][0]
    assert sung["start"] == pytest.approx(10.0)
    assert sung["end"] == pytest.approx(10.5)
