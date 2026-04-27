"""Furigana annotation for Japanese lyrics.

Pipeline (M3 v1):
    fugashi + UniDic-lite -> per-token reading
    + per-song override JSON (gikun fixes)
    -> token sequence with ruby annotations

Future (M3 v2):
    + Yomikata BERT for heteronym disambiguation (運命/本気/etc.)
    + sub-token ruby (only on the kanji part of mixed kanji+kana words)
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

# Range of CJK Unified Ideographs we treat as "needs ruby".
_KANJI_RE = re.compile(r"[㐀-鿿豈-﫿]")
# Hiragana / Katakana ranges (no ruby needed).
_KANA_RE = re.compile(r"[぀-ゟ゠-ヿ]")
# Whitespace (incl. full-width).
_WS_RE = re.compile(r"[\s　]+")


def has_kanji(s: str) -> bool:
    return bool(_KANJI_RE.search(s))


def kata_to_hira(s: str) -> str:
    out = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


@dataclass
class Token:
    surface: str
    reading: str | None  # hiragana, None if no ruby
    kana_only: bool
    pos: str
    is_punct: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Line:
    text: str
    tokens: list[Token]

    def to_dict(self) -> dict:
        return {"text": self.text, "tokens": [t.to_dict() for t in self.tokens]}


def _tokenize_line(tagger, text: str, override: dict[str, str]) -> Line:
    tokens: list[Token] = []
    for word in tagger(text):
        surface = word.surface
        if not surface or _WS_RE.fullmatch(surface):
            continue

        feat = word.feature
        pos = feat.pos1 or "?"
        is_punct = pos in {"補助記号", "記号"}

        # Override wins (gikun).
        if surface in override:
            reading_hira = override[surface]
            kana_only = not has_kanji(surface)
        elif has_kanji(surface):
            kana = getattr(feat, "kana", None) or getattr(feat, "pron", None)
            reading_hira = kata_to_hira(kana) if kana else None
            kana_only = False
        else:
            reading_hira = None
            kana_only = True

        tokens.append(
            Token(
                surface=surface,
                reading=reading_hira,
                kana_only=kana_only,
                pos=pos,
                is_punct=is_punct,
            )
        )
    return Line(text=text, tokens=tokens)


def annotate_lyrics(
    lyrics_path: str | Path,
    *,
    override_path: str | Path | None = None,
) -> list[Line]:
    """Tokenize a lyrics.txt and return Line objects with ruby annotations."""
    import fugashi

    lyrics_path = Path(lyrics_path)
    raw = lyrics_path.read_text(encoding="utf-8")
    raw = unicodedata.normalize("NFKC", raw)

    override: dict[str, str] = {}
    if override_path:
        override_path = Path(override_path)
        if override_path.exists():
            override = json.loads(override_path.read_text(encoding="utf-8"))

    tagger = fugashi.Tagger()

    lines: list[Line] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Full-width 　 inside a visual line is a phrase break, not a line
        # break. fugashi naturally drops whitespace tokens, so keep the line
        # whole and let alignment / rendering decide whether to pause there.
        lines.append(_tokenize_line(tagger, line, override))
    return lines


def dump_json(lines: list[Line], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([ln.to_dict() for ln in lines], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
