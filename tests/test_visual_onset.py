import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import visual_onset as vo  # noqa: E402

DELTAS = vo.DEFAULT_DELTAS


def test_consonant_classes():
    assert vo.consonant_class("か") == "stop"      # k
    assert vo.consonant_class("ぱ") == "stop"      # p
    assert vo.consonant_class("さ") == "fric"      # s
    assert vo.consonant_class("ち") == "fric"      # ch affricate
    assert vo.consonant_class("つ") == "fric"      # ts affricate
    assert vo.consonant_class("が") == "voiced"    # g
    assert vo.consonant_class("ま") == "sonorant"  # m
    assert vo.consonant_class("な") == "sonorant"  # n-onset
    assert vo.consonant_class("あ") == "zero"      # vowel
    assert vo.consonant_class("ん") == "zero"      # moraic nasal
    assert vo.consonant_class("っ") == "zero"      # geminate
    assert vo.consonant_class("ー") == "zero"      # long mark


def test_push_only_forward_and_capped():
    # voiceless stop, ample duration -> pushed by ~40ms
    ns = vo.push_char(1.0, 1.5, "か", DELTAS, cap_frac=0.5, min_vowel=0.04)
    assert abs(ns - 1.04) < 1e-6
    # vowel-initial -> never moved
    assert vo.push_char(1.0, 1.5, "あ", DELTAS, cap_frac=0.5, min_vowel=0.04) == 1.0
    # tiny char -> capped so vowel core preserved (never past end - min_vowel)
    ns = vo.push_char(1.0, 1.05, "か", DELTAS, cap_frac=0.5, min_vowel=0.04)
    assert ns <= 1.05 - 0.04 + 1e-9 and ns >= 1.0


def test_apply_updates_line_span():
    lines = [{"tokens": [{"chars": [
        {"char": "か", "start": 1.0, "end": 1.5},
        {"char": "あ", "start": 1.5, "end": 2.0}]}],
        "start": 1.0, "end": 2.0}]
    moved = vo.apply_visual_onset(lines, DELTAS, cap_frac=0.5, min_vowel=0.04)
    assert moved == 1  # only か moved
    assert abs(lines[0]["start"] - 1.04) < 1e-6  # line start follows pushed first char
