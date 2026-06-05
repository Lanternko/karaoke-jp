from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from karaoke_jp.ruby import _tokenize_line


class FakeTagger:
    """Tiny context-sensitive tagger for phrase-boundary regression tests."""

    def __call__(self, text: str):
        if " " in text or "　" in text:
            # Simulate the UniDic failure mode: when a whole line is tokenized,
            # the second phrase's 君 is treated as a suffix.
            rows = [
                ("僕", "ボク", "代名詞"),
                ("の", "ノ", "助詞"),
                ("ため", "タメ", "名詞"),
                ("君", "クン", "接尾辞"),
                ("の", "ノ", "助詞"),
                ("ため", "タメ", "名詞"),
            ]
        elif text == "君のため":
            rows = [("君", "キミ", "代名詞"), ("の", "ノ", "助詞"), ("ため", "タメ", "名詞")]
        elif text == "僕のため":
            rows = [("僕", "ボク", "代名詞"), ("の", "ノ", "助詞"), ("ため", "タメ", "名詞")]
        elif text == "行こう海を":
            rows = [("行こう", "イコウ", "動詞"), ("海", "カイ", "接尾辞"), ("を", "ヲ", "助詞")]
        elif text == "海":
            rows = [("海", "ウミ", "名詞")]
        elif text == "山田君":
            rows = [("山田", "ヤマダ", "名詞"), ("君", "クン", "接尾辞")]
        elif text == "君":
            rows = [("君", "キミ", "代名詞")]
        else:
            rows = [(text, text, "名詞")]

        return [
            SimpleNamespace(
                surface=surface,
                feature=SimpleNamespace(pos1=pos, kana=kana, pron=kana),
            )
            for surface, kana, pos in rows
        ]


def test_tokenize_line_treats_whitespace_as_phrase_boundary() -> None:
    for sep in (" ", "　"):
        line = _tokenize_line(FakeTagger(), f"僕のため{sep}君のため", {})

        assert [(tok.surface, tok.reading, tok.pos) for tok in line.tokens] == [
            ("僕", "ぼく", "代名詞"),
            ("の", None, "助詞"),
            ("ため", None, "名詞"),
            ("君", "きみ", "代名詞"),
            ("の", None, "助詞"),
            ("ため", None, "名詞"),
        ]


def test_tokenize_line_retags_false_suffix_after_verb_as_standalone() -> None:
    line = _tokenize_line(FakeTagger(), "行こう海を", {})

    assert [(tok.surface, tok.reading, tok.pos) for tok in line.tokens] == [
        ("行こう", "いこう", "動詞"),
        ("海", "うみ", "名詞"),
        ("を", None, "助詞"),
    ]


def test_tokenize_line_keeps_true_suffix_after_noun() -> None:
    line = _tokenize_line(FakeTagger(), "山田君", {})

    assert [(tok.surface, tok.reading, tok.pos) for tok in line.tokens] == [
        ("山田", "やまだ", "名詞"),
        ("君", "くん", "接尾辞"),
    ]
