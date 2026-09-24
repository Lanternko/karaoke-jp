from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from karaoke_jp.align import lyrics_tokens_to_kana_units


def test_lyrics_tokens_to_kana_units_preserves_override_reading() -> None:
    lyrics_lines = [
        {
            "text": "笑顔の下では",
            "tokens": [
                {"surface": "笑顔", "reading": "えがお", "kana_only": False},
                {"surface": "の", "reading": None, "kana_only": True},
                {"surface": "下", "reading": "した", "kana_only": False},
                {"surface": "で", "reading": None, "kana_only": True},
                {"surface": "は", "reading": None, "kana_only": True},
            ],
        }
    ]

    kanas = lyrics_tokens_to_kana_units(lyrics_lines)

    assert [kana.kana for kana in kanas] == ["え", "が", "お", "の", "し", "た", "で", "は"]
    assert [(kana.src_start, kana.src_end) for kana in kanas] == [
        (0, 2),
        (0, 2),
        (0, 2),
        (2, 3),
        (3, 4),
        (3, 4),
        (4, 5),
        (5, 6),
    ]


def test_lyrics_tokens_to_kana_units_keeps_particle_pronunciation_from_tagger() -> None:
    class FakeTagger:
        def __call__(self, text: str):
            mapping = {
                "は": "わ",
                "へ": "え",
                "を": "お",
            }
            reading = mapping.get(text, text)
            return [
                SimpleNamespace(
                    surface=text,
                    feature=SimpleNamespace(kana=reading, pron=reading),
                )
            ]

    lyrics_lines = [
        {
            "text": "はへを",
            "tokens": [
                {"surface": "は", "reading": None, "kana_only": True},
                {"surface": "へ", "reading": None, "kana_only": True},
                {"surface": "を", "reading": None, "kana_only": True},
            ],
        }
    ]

    kanas = lyrics_tokens_to_kana_units(lyrics_lines, FakeTagger())

    assert [kana.kana for kana in kanas] == ["わ", "え", "お"]
