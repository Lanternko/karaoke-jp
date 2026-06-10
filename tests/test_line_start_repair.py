from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "line_start_repair.py"
SPEC = importlib.util.spec_from_file_location("line_start_repair", SCRIPT)
line_start_repair = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = line_start_repair
SPEC.loader.exec_module(line_start_repair)


def _line() -> dict:
    return {
        "text": "「あい",
        "start": 2.0,
        "end": 2.8,
        "tokens": [
            {
                "surface": "「",
                "is_punct": True,
                "start": 2.0,
                "end": 2.0,
                "chars": [{"char": "「", "start": 2.0, "end": 2.0}],
            },
            {
                "surface": "あい",
                "is_punct": False,
                "start": 2.0,
                "end": 2.8,
                "chars": [
                    {"char": "あ", "start": 2.0, "end": 2.4},
                    {"char": "い", "start": 2.4, "end": 2.8},
                ],
            },
        ],
    }


def test_rms_onset_between_returns_sustained_voiced_frame_after_valley() -> None:
    rms_db = np.array([-60, -55, -58, -42, -12, -11, -10, -45], dtype=np.float32)

    onset = line_start_repair.rms_onset_between(
        rms_db,
        0.1,
        0.0,
        0.7,
        top_db=18.0,
        sustain=0.3,
    )

    assert onset == pytest.approx(0.4)


def test_repair_blends_start_toward_rms_onset_and_clamps_leading_punct() -> None:
    lines = [_line()]
    hint_lines = [{"start": 1.0}]
    rms_db = np.array(
        [-60, -60, -60, -60, -60, -60, -60, -60, -60, -60, -55, -50, -12, -11, -10, -9, -9, -9, -9, -9, -9],
        dtype=np.float32,
    )

    changes = line_start_repair.repair(
        lines,
        hint_lines,
        rms_db,
        0.1,
        max_shift=1.2,
        min_late=0.45,
        onset_top_db=18.0,
        onset_sustain=0.2,
        prev_guard=0.04,
        blend=0.5,
        min_move=0.25,
        skip_first_line=False,
    )

    assert changes == [(0, 2.0, 1.6)]
    punct = lines[0]["tokens"][0]["chars"][0]
    sung = lines[0]["tokens"][1]["chars"][0]
    assert lines[0]["start"] == pytest.approx(1.6)
    assert punct["start"] == pytest.approx(1.6)
    assert punct["end"] == pytest.approx(1.6)
    assert sung["start"] == pytest.approx(1.6)


def test_repair_respects_min_move_and_skip_first_line() -> None:
    hint_lines = [{"start": 1.0}]
    rms_db = np.array([-60] * 18 + [-12, -11, -10, -9], dtype=np.float32)

    min_move_lines = [_line()]
    min_move_changes = line_start_repair.repair(
        min_move_lines,
        hint_lines,
        rms_db,
        0.1,
        max_shift=1.2,
        min_late=0.45,
        onset_top_db=18.0,
        onset_sustain=0.2,
        prev_guard=0.04,
        blend=0.9,
        min_move=0.25,
        skip_first_line=False,
    )
    assert min_move_changes == []
    assert min_move_lines[0]["start"] == pytest.approx(2.0)

    skip_lines = [_line()]
    skip_changes = line_start_repair.repair(
        skip_lines,
        hint_lines,
        rms_db,
        0.1,
        max_shift=1.2,
        min_late=0.45,
        onset_top_db=18.0,
        onset_sustain=0.2,
        prev_guard=0.04,
        blend=0.0,
        min_move=0.0,
        skip_first_line=True,
    )
    assert skip_changes == []
    assert skip_lines[0]["start"] == pytest.approx(2.0)


def test_repair_handles_missing_token_start() -> None:
    lines = [_line()]
    lines[0]["tokens"][1]["start"] = None
    hint_lines = [{"start": 1.0}]
    rms_db = np.array([-60] * 12 + [-12, -11, -10, -9], dtype=np.float32)

    changes = line_start_repair.repair(
        lines,
        hint_lines,
        rms_db,
        0.1,
        max_shift=1.2,
        min_late=0.45,
        onset_top_db=18.0,
        onset_sustain=0.2,
        prev_guard=0.04,
        blend=0.5,
        min_move=0.25,
        skip_first_line=False,
    )

    assert changes == [(0, 2.0, 1.6)]
    assert lines[0]["tokens"][1]["start"] == pytest.approx(1.6)
