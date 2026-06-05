from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "line_end_repair.py"
SPEC = importlib.util.spec_from_file_location("line_end_repair", SCRIPT)
line_end_repair = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(line_end_repair)


def _line(start: float, end: float, text: str = "あ") -> dict:
    return {
        "text": text,
        "start": start,
        "end": end,
        "tokens": [
            {
                "surface": text,
                "is_punct": False,
                "start": start,
                "end": end,
                "chars": [{"surface": text, "start": start, "end": end}],
            },
            {
                "surface": "。",
                "is_punct": True,
                "start": end,
                "end": end,
                "chars": [{"surface": "。", "start": end, "end": end}],
            },
        ],
    }


def test_repair_extends_last_sung_char_but_not_trailing_punctuation() -> None:
    lines = [_line(0.0, 1.0), _line(2.0, 2.5)]
    rms_db = np.array([-80.0] * 10 + [-12.0] * 6 + [-80.0] * 20)

    changes = line_end_repair.repair(
        lines,
        rms_db,
        hop_s=0.1,
        duration=4.0,
        tail_top_db=30.0,
        max_extend=2.0,
        next_guard=0.1,
        tail_gap=0.2,
    )

    assert changes == [(0, 1.0, 1.6)]
    assert lines[0]["end"] == 1.6
    assert lines[0]["tokens"][0]["end"] == 1.6
    assert lines[0]["tokens"][0]["chars"][0]["end"] == 1.6
    assert lines[0]["tokens"][1]["chars"][0]["end"] == 1.0


def test_repair_respects_next_line_guard() -> None:
    lines = [_line(0.0, 1.0), _line(1.25, 2.0)]
    rms_db = np.array([-80.0] * 10 + [-12.0] * 20)

    line_end_repair.repair(
        lines,
        rms_db,
        hop_s=0.1,
        duration=3.0,
        tail_top_db=30.0,
        max_extend=2.0,
        next_guard=0.12,
        tail_gap=0.2,
    )

    assert lines[0]["end"] == 1.13


def test_repair_does_not_extend_after_initial_silence() -> None:
    lines = [_line(0.0, 1.0), _line(3.0, 3.5)]
    rms_db = np.array([-80.0] * 13 + [-12.0] * 10)

    changes = line_end_repair.repair(
        lines,
        rms_db,
        hop_s=0.1,
        duration=4.0,
        tail_top_db=30.0,
        max_extend=2.0,
        next_guard=0.1,
        tail_gap=0.2,
    )

    assert changes == []
    assert lines[0]["end"] == 1.0
