from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import midi_timing


def test_match_notes_to_chars_splits_single_note_across_multiple_chars() -> None:
    spans = midi_timing._match_notes_to_chars(
        [(1.0, 2.0, 60)],
        [1.00, 1.10, 1.20],
    )

    assert spans is not None
    assert len(spans) == 3
    assert spans[0][0] == pytest.approx(1.0)
    assert spans[0][1] == pytest.approx(4 / 3)
    assert spans[1][0] == pytest.approx(4 / 3)
    assert spans[1][1] == pytest.approx(5 / 3)
    assert spans[2][0] == pytest.approx(5 / 3)
    assert spans[2][1] == pytest.approx(2.0)


def test_apply_midi_timing_skips_trailing_punctuation_when_matching() -> None:
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

    updated, kept = midi_timing.apply_midi_timing(
        lines,
        [(1.0, 1.2, 60), (1.3, 1.5, 62)],
    )

    assert (updated, kept) == (1, 0)
    chars = lines[0]["tokens"][0]["chars"]
    assert chars[0]["start"] == pytest.approx(1.0)
    assert chars[0]["end"] == pytest.approx(1.3)
    assert chars[1]["start"] == pytest.approx(1.3)
    assert chars[1]["end"] == pytest.approx(1.5)
    assert chars[2]["start"] == pytest.approx(1.5)
    assert chars[2]["end"] == pytest.approx(1.5)
    assert lines[0]["start"] == pytest.approx(1.0)
    assert lines[0]["end"] == pytest.approx(1.5)


def test_apply_midi_timing_retimes_leading_and_trailing_quotes() -> None:
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

    updated, kept = midi_timing.apply_midi_timing(
        lines,
        [(2.0, 2.5, 60)],
        window_margin=20.0,
    )

    assert (updated, kept) == (1, 0)
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
