"""romaji_overrides.py — deterministic romaji->kana and reading extraction.

The generator audits fugashi readings against a user-provided romaji
transcript (gikun like 響めき=どよめき only exist there). The fast path —
fullmatch with all fugashi readings inlined — must swallow clean lines so
lazy-capture truncation artifacts never produce false overrides.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from romaji_overrides import build_line_pattern, kana_norm, romaji_to_kana


@pytest.mark.parametrize(
    ("romaji", "kana"),
    [
        ("doyomeki", "どよめき"),
        ("kirameki", "きらめき"),
        ("iribitatta", "いりびたった"),
        ("kawaranai ne", "かわらないね"),
        ("shite", "して"),
        ("chirakaru", "ちらかる"),
        ("futari kizamou", "ふたりきざもう"),
        ("MYUUJIKKU", "みゅうじっく"),
        ("n'", "ん"),
        ("ashita", "あした"),
    ],
)
def test_romaji_to_kana(romaji: str, kana: str) -> None:
    assert romaji_to_kana(romaji) == kana


def test_kana_norm_long_vowel_and_katakana() -> None:
    assert kana_norm("メロディー") == "めろでぃい"
    assert kana_norm("コーヒー") == "こおひい"
    assert kana_norm("メモリー") == "めもりい"


def _tok(surface: str, reading: str | None = None, *, punct: bool = False) -> dict:
    return {"surface": surface, "reading": reading, "is_punct": punct,
            "kana_only": reading is None}


def test_fast_path_swallows_clean_line() -> None:
    tokens = [_tok("入り浸っ", "いりびたっ"), _tok("た"), _tok("散らかる", "ちらかる"),
              _tok("部屋", "へや"), _tok("も")]
    literal, _, _ = build_line_pattern(tokens, literal_readings=True)
    assert literal.match(kana_norm(romaji_to_kana("iribitatta chirakaru heya mo")))


def test_extraction_catches_gikun() -> None:
    tokens = [_tok("響", "ひびき"), _tok("めき"), _tok("煌めき", "きらめき"),
              _tok("と"), _tok("君", "きみ"), _tok("も")]
    literal, _, _ = build_line_pattern(tokens, literal_readings=True)
    line = kana_norm(romaji_to_kana("doyomeki kirameki to kimi mo"))
    assert not literal.match(line)
    pattern, kanji_idx, ambiguous = build_line_pattern(tokens, literal_readings=False)
    match = pattern.match(line)
    assert match and match.group("g0") == "どよ"
    assert not ambiguous


def test_trailing_punct_cannot_eat_final_capture() -> None:
    tokens = [_tok("も"), _tok('"', punct=True), _tok("踊ろう", "おどろう"),
              _tok('"', punct=True)]
    literal, _, _ = build_line_pattern(tokens, literal_readings=True)
    assert literal.match(kana_norm(romaji_to_kana('mo "Odorou"')))


def test_adjacent_kanji_flagged_ambiguous() -> None:
    tokens = [_tok("二人", "ふたり"), _tok("刻もう", "きざもう")]
    pattern, kanji_idx, ambiguous = build_line_pattern(tokens, literal_readings=False)
    assert set(kanji_idx) == ambiguous == {0, 1}
