from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_alignment.py"
SPEC = importlib.util.spec_from_file_location("eval_alignment", SCRIPT)
eval_alignment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = eval_alignment
SPEC.loader.exec_module(eval_alignment)


def test_evaluate_reports_boundary_metrics_and_stress_rows() -> None:
    lines = [
        {
            "text": "あい",
            "start": 1.0,
            "end": 2.0,
            "tokens": [
                {
                    "chars": [
                        {"char": "あ", "start": 1.0, "end": 1.4},
                        {"char": "い", "start": 1.4, "end": 2.0},
                    ]
                }
            ],
        },
        {
            "text": "うえ",
            "start": 2.0,
            "end": 3.0,
            "tokens": [
                {
                    "chars": [
                        {"char": "う", "start": 2.0, "end": 2.4},
                        {"char": "え", "start": 2.4, "end": 3.0},
                    ]
                }
            ],
        },
    ]
    gold = [
        eval_alignment.GoldRow("song", 0, "あい", 1.0, 2.0),
        eval_alignment.GoldRow("song", 1, "うえ", 2.5, 3.0),
    ]

    summary, rows = eval_alignment.evaluate(
        lines,
        gold,
        stress_error_threshold=0.4,
        stress_gap_threshold=0.25,
    )

    assert summary["n"] == 2
    assert summary["start"]["mae"] == pytest.approx(0.25)
    assert summary["start"]["bias"] == pytest.approx(-0.25)
    assert summary["end"]["mae"] == pytest.approx(0.0)
    assert summary["stress_n"] == 1
    assert rows[1].signed_start_error == pytest.approx(-0.5)
    assert rows[1].stress_reason == "start_error,pred_butted"


def test_invariants_count_only_sung_zero_duration_as_sung_invalid() -> None:
    lines = [
        {
            "text": "あ？",
            "start": 1.0,
            "end": 1.0,
            "tokens": [
                {
                    "chars": [
                        {"char": "あ", "start": 1.0, "end": 1.0},
                        {"char": "？", "start": 1.0, "end": 1.0},
                    ]
                }
            ],
        }
    ]

    inv = eval_alignment.invariants(lines)

    assert inv["zero_duration_lines"] == 1
    assert inv["zero_duration_chars"] == 2
    assert inv["zero_duration_sung_chars"] == 1
