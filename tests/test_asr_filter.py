"""Whisper hallucination filter — guard against the "ねえねえねえ..." class
of repetition artefacts that Whisper emits over instrumental breaks.

Without this filter, a 33s "ねえ"×100 segment leaks into asr.json, where the
aligner finds zero overlap with lyrics.txt, leaving the next real verse with
no usable Whisper anchors → midi_timing collapses every char in that verse
onto a single timestamp. Symptom seen on haru-hikage 2:14-2:18.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_asr import is_hallucinated_segment


@pytest.mark.parametrize(
    "text",
    [
        "ねえ" * 100,  # entropy gate (2 unique / 200)
        "ねえ" * 11,  # 2-gram run = 11 ≥ 10
        "あぁ" * 10,  # 2-gram run = 10 ≥ 10 (boundary)
        "ーー" * 50,  # 1 unique / 100 — entropy
    ],
)
def test_hallucinations_dropped(text: str) -> None:
    assert is_hallucinated_segment(text)


@pytest.mark.parametrize(
    "text",
    [
        "悴んだ心　ふるえる眼差し",
        "雲間をぬって　きらりきらり",  # legitimate 2-gram repeat (×2)
        "ずっと　ずっと　離さないでいて",  # legitimate word repeat (×2)
        "あぁぁぁぁ",  # too short to flag on entropy
        "突然降る夕立　あぁ傘もないや嫌",
        "縁を結んでは　ほどきほどかれ",
        "",  # empty
        "ら",  # single char
    ],
)
def test_legitimate_kept(text: str) -> None:
    assert not is_hallucinated_segment(text)


def test_long_chorus_with_repeat_kept() -> None:
    # Realistic chorus: kirari-kirari + line follow-on, ~30 chars total.
    text = "雲間をぬってきらりきらり心満たしては溢れ"
    assert not is_hallucinated_segment(text)


def test_short_repetition_kept() -> None:
    # Less than 30 chars and ngram-run < 10: should slip through both gates.
    assert not is_hallucinated_segment("ねえねえねえ")  # 3 runs


def test_threshold_boundary() -> None:
    # Exactly 10 consecutive 2-gram repeats should fire.
    assert is_hallucinated_segment("ねえ" * 10)
    # 9 should not.
    assert not is_hallucinated_segment("ねえ" * 9)


@pytest.mark.parametrize(
    "text",
    [
        "ご視聴ありがとうございました",  # NIGHT DANCER 95.5s: emitted on a breathy ad-lib blip
        "ご視聴ありがとうございます",
        "チャンネル登録お願いします",
        "コメント欄で教えてください",
    ],
)
def test_stock_phrase_hallucinations_dropped(text: str) -> None:
    assert is_hallucinated_segment(text)


@pytest.mark.parametrize(
    "text",
    [
        "ありがとう さよなら",  # plain thanks in a lyric is NOT the stock phrase
        "登録した記憶を辿って",  # 登録 alone isn't チャンネル登録
    ],
)
def test_stock_phrase_near_misses_kept(text: str) -> None:
    assert not is_hallucinated_segment(text)
