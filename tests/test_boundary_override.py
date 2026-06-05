from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "boundary_override.py"
SPEC = importlib.util.spec_from_file_location("boundary_override", SCRIPT)
boundary_override = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(boundary_override)


def test_apply_override_resnaps_leading_cascade_and_stops_at_slack() -> None:
    punct = {"char": "(", "start": 0.0, "end": 0.0}
    chars = [
        {"char": "た", "start": 0.0, "end": 1.0},
        {"char": "い", "start": 1.0, "end": 2.0},
        {"char": "せ", "start": 2.0, "end": 3.0},
        {"char": "つ", "start": 5.0, "end": 6.0},
    ]
    line = {
        "text": "(たいせつ",
        "start": 0.0,
        "end": 6.0,
        "tokens": [
            {"surface": "(", "is_punct": True, "chars": [punct]},
            {"surface": "たいせつ", "is_punct": False, "chars": chars},
        ],
    }

    moved = boundary_override.apply_override(
        line,
        1.0,
        [1.0, 2.0, 3.0, 4.0],
    )

    assert moved == 3
    assert line["start"] == 1.0
    assert punct == {"char": "(", "start": 1.0, "end": 1.0}
    assert [ch["start"] for ch in chars] == [1.0, 2.0, 3.0, 5.0]
    assert [ch["end"] for ch in chars] == [2.0, 3.0, 5.0, 6.0]
    assert all(ch["end"] > ch["start"] for ch in chars)
