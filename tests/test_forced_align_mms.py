"""forced_align_mms.py pure parts — romanization attribution, skeleton, guard.

The aligner itself needs torch + the HF checkpoint; these tests cover the
deterministic glue where regressions would silently corrupt timing: which
letters belong to which mora (small kana, sokuon doubling, long vowels),
the ASR-free skeleton builder, and the re-entry guard's do-no-harm rules.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from forced_align_mms import (
    apply_reentry_guard,
    records_to_words,
    skeleton_from_tokens,
)


def _recs(*kana: str) -> list[dict]:
    return [{"kana": k, "char": {}, "hint": 0.0} for k in kana]


def test_plain_morae_one_word_each() -> None:
    words, lpr = records_to_words(_recs("ど", "よ", "め", "き"))
    assert words == ["do", "yo", "me", "ki"]
    assert lpr == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_small_ya_combines_with_host() -> None:
    words, lpr = records_to_words(_recs("ち", "ょ", "う"))
    assert words == ["cho", "u"]
    assert lpr[0] == [0, 1]
    assert lpr[1] == [2]
    assert lpr[2] == [3]


def test_sokuon_doubles_next_consonant() -> None:
    words, lpr = records_to_words(_recs("び", "た", "っ", "た"))
    assert words == ["bi", "ta", "tta"]
    assert lpr[2] == [4]
    assert lpr[3] == [5, 6]


def test_long_vowel_mark_repeats_previous_vowel() -> None:
    words, _ = records_to_words(_recs("り", "ー"))
    assert words == ["ri", "i"]


def test_latin_run_groups_into_one_word() -> None:
    words, lpr = records_to_words(_recs("T", "u", "t", "u"))
    assert words == ["tutu"]
    assert [p for lst in lpr for p in lst] == [0, 1, 2, 3]


def test_skeleton_from_tokens(tmp_path: Path) -> None:
    tokens = [{
        "text": "響めき",
        "tokens": [
            {"surface": "響", "reading": "どよ", "kana_only": False,
             "pos": "名詞", "is_punct": False},
            {"surface": "めき", "reading": None, "kana_only": True,
             "pos": "接尾辞", "is_punct": False},
        ],
    }]
    p = tmp_path / "tokens.json"
    p.write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
    lines = skeleton_from_tokens(p)
    assert lines[0]["start"] == 0.0
    chars = [c["char"] for t in lines[0]["tokens"] for c in t["chars"]]
    assert chars == ["響", "め", "き"]


def _line(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end,
            "tokens": [{"surface": text,
                        "chars": [{"char": text[0], "start": start, "end": end}]}]}


def test_reentry_guard_snaps_only_post_gap_outliers(tmp_path: Path) -> None:
    seg = tmp_path / "rms_segments.json"
    seg.write_text(json.dumps({
        "params": {"pad": 0.25},
        "segments": [{"start": 9.45, "end": 14.0}, {"start": 19.8, "end": 24.0}],
    }), encoding="utf-8")
    lines = [
        _line("あ", 1.0, 4.0),
        _line("い", 10.4, 13.0),
        _line("う", 13.2, 16.0),
    ]
    moved = apply_reentry_guard(lines, seg)
    assert moved == 1
    assert abs(lines[1]["start"] - 9.7) < 1e-6
    assert lines[2]["start"] == 13.2

def test_f0_reentry_guard_snaps_post_gap_and_respects_quiet_gate(tmp_path: Path) -> None:
    import numpy as np

    from forced_align_mms import f0_reentry_guard

    hop = 0.01
    f0 = np.zeros(3000, dtype=np.float32)
    f0[0:400] = 220.0
    f0[1062:1500] = 220.0
    f0[1510:2000] = 230.0
    npz = tmp_path / "f0.npz"
    np.savez(npz, f0=f0, hop_seconds=np.array([hop], dtype=np.float32))

    def line(text, start, end):
        return {"text": text, "start": start, "end": end,
                "tokens": [{"surface": text,
                            "chars": [{"char": text[0], "start": start, "end": end}]}]}

    lines = [
        line("あ", 0.5, 4.0),
        line("い", 11.2, 14.9),
        line("う", 15.1, 19.9),
    ]
    moved = f0_reentry_guard(lines, npz)
    assert moved == 1
    assert abs(lines[1]["start"] - 10.62) < 0.02
    assert lines[2]["start"] == 15.1
