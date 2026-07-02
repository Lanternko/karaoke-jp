#!/usr/bin/env python3
"""UltraSinger syllable text -> first-mora romaji label (COnPOff+L L axis).

Chain (PLAN Phase 4.2):
  syllable text (Whisper output, may contain kanji/kana/marks)
    -> reading kana   (fugashi + UniDic, NEVER pykakasi)
    -> phones         (kiritan japanese.table, the SAME kana->phone set as GT
                       mono_label -> guarantees identical phone spelling)
    -> mora label     (conpoff_l.group_morae -> take the FIRST mora)

Un-convertible syllables (blank / punctuation / non-Japanese) -> "?" (auto L
fail, honest tax). Each UltraStar note gets one (onset, label) row.

Run standalone smoke test:
  ~/venvs/karaoke-jp-lyrics/bin/python syllable_to_mora.py --selftest
(fugashi lives in karaoke-jp-lyrics; group_morae import needs no mir_eval.)
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # benchmarks/kiritan for conpoff_l

# group_morae is a pure function that never touches mir_eval, but conpoff_l does
# a top-level `from mir_eval.util import _bipartite_match`. Phase-4 runs in the
# fugashi venv (karaoke-jp-lyrics), which has no mir_eval. Shim the import so we
# reuse group_morae verbatim (PLAN: do not copy-paste harness logic) without the
# unused dependency; the real eval (Phase 5) runs in the mir_eval venv.
import types  # noqa: E402
if "mir_eval.util" not in sys.modules:
    _me = types.ModuleType("mir_eval")
    _me_util = types.ModuleType("mir_eval.util")
    _me_util._bipartite_match = None  # never invoked here
    _me.util = _me_util
    sys.modules.setdefault("mir_eval", _me)
    sys.modules.setdefault("mir_eval.util", _me_util)

from conpoff_l import group_morae  # noqa: E402  (pure fn, no mir_eval touch)

KIRITAN = Path.home() / "side_projects/kiritan/kiritan_singing"
TABLE_PATH = KIRITAN / "japanese.table"

# katakana -> hiragana shift (same block offset, U+30A1..U+30F6 -> U+3041..)
_KATA_LO, _KATA_HI = 0x30A1, 0x30F6


def _kata_to_hira(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if _KATA_LO <= o <= _KATA_HI:
            out.append(chr(o - 0x60))
        elif ch == "ヴ":  # already covered by range but explicit
            out.append("ゔ")
        else:
            out.append(ch)
    return "".join(out)


def load_table(path: Path = TABLE_PATH) -> dict[str, list[str]]:
    """kana (hiragana key) -> list of phones, e.g. 'きゃ' -> ['ky','a']."""
    table: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        kana, phones = parts[0], parts[1:]
        table[kana] = phones
    return table


_TABLE = load_table()
_MAX_KANA = max(len(k) for k in _TABLE)  # longest key (2, e.g. きゃ/ふぁ)

# small kana that combine with the previous kana into a digraph the table lists
# directly (ゃゅょぁぃぅぇぉ); the greedy longest-match handles these because the
# table has きゃ etc. as 2-char keys. Long-vowel mark ー = hold previous vowel.

# fugashi tagger (lazy: only if we actually need readings)
_TAGGER = None


def _tagger():
    global _TAGGER
    if _TAGGER is None:
        import fugashi  # type: ignore
        _TAGGER = fugashi.Tagger()
    return _TAGGER


def _reading_kana(text: str) -> str:
    """Get katakana reading of a (possibly kanji) syllable via fugashi+UniDic.
    Falls back to the surface itself for tokens with no reading (already kana)."""
    tagger = _tagger()
    out = []
    for w in tagger(text):
        # UniDic features: pron/kana reading fields vary; prefer .feature.kana
        kana = None
        feat = w.feature
        for attr in ("kana", "pron", "reading"):
            v = getattr(feat, attr, None)
            if v and v != "*":
                kana = v
                break
        out.append(kana if kana else w.surface)
    return "".join(out)


def _hira_to_phones(hira: str) -> list[str] | None:
    """Greedy longest-match kana run -> phones. Returns None if any run char is
    un-mappable (so caller can decide '?')."""
    # ゔ (U+3094) is written う゛ in the kiritan table; normalize to match.
    hira = hira.replace("ゔ", "う゛")
    phones: list[str] = []
    i = 0
    n = len(hira)
    while i < n:
        ch = hira[i]
        if ch == "ー":  # long-vowel: repeat last vowel phone
            if phones and phones[-1] in "aiueo":
                phones.append(phones[-1])
            i += 1
            continue
        # try longest kana key first
        matched = False
        for span in range(min(_MAX_KANA, n - i), 0, -1):
            key = hira[i:i + span]
            if key in _TABLE:
                phones.extend(_TABLE[key])
                i += span
                matched = True
                break
        if not matched:
            # small vowels alone / stray marks -> map bare vowel if possible
            solo = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
                    "ゃ": None, "ゅ": None, "ょ": None}
            if ch in solo and solo[ch]:
                phones.append(solo[ch])
                i += 1
                continue
            return None
    return phones


def syllable_to_mora(text: str) -> str:
    """UltraStar syllable text -> first-mora romaji label, or '?' on failure."""
    if text is None:
        return "?"
    # strip UltraStar word-separator spaces and NFKC-normalize
    t = unicodedata.normalize("NFKC", text).strip()
    if not t:
        return "?"
    # drop leading/trailing latin punctuation & spaces that Whisper may attach
    t = t.strip(" 　.,!?！？、。「」『』()（）~〜ー…-\t")
    if not t:
        return "?"
    # already-kana fast path: if all chars are kana/marks skip fugashi
    if re.fullmatch(r"[ぁ-ゖァ-ヺ゛゜ー]+", t):
        hira = _kata_to_hira(t)
    else:
        try:
            reading = _reading_kana(t)
        except Exception:
            return "?"
        hira = _kata_to_hira(reading)
    phones = _hira_to_phones(hira)
    if not phones:
        return "?"
    # frame group_morae expects (on, off, ph) rows; times irrelevant here
    rows = [(float(k), float(k), ph) for k, ph in enumerate(phones)]
    morae = group_morae(rows)
    if not morae:
        return "?"
    return morae[0][1]


def _selftest():
    cases = {
        "き": "ki", "きゃ": "kya", "しゃ": "sha", "つ": "tsu", "ん": "N",
        "し": "shi", "ち": "chi", "ふぁ": "fa", "を": "wo", "ー": "?",
        "そら": "so", "東京": "to", "": "?", ".": "?", "hello": "?",
        "きょう": "kyo", "ヴ": "vu", "ぎゅ": "gyu",
    }
    ok = 0
    for src, exp in cases.items():
        got = syllable_to_mora(src)
        flag = "OK " if got == exp else "XX "
        ok += got == exp
        print(f"  {flag} {src!r:12s} -> {got!r:6s} (exp {exp!r})")
    print(f"selftest: {ok}/{len(cases)} passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        for arg in sys.argv[1:]:
            print(f"{arg!r} -> {syllable_to_mora(arg)!r}")
