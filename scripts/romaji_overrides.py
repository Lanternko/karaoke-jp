"""Generate per-song reading overrides from a user-provided romaji transcript.

fugashi+UniDic picks dictionary readings, but song lyrics use gikun and rare
readings (響めき=どよめき, 東風=こち) that only the romaji transcript — which
records what the singer actually sings — gets right. This script extracts
kanji-token readings from the romaji deterministically (no LLM):

    lyrics tokens   響(capture) めき(anchor) 煌めき(capture) と(anchor) ...
    romaji line     doyomeki kirameki to kimi mo  ->  どよめきき らめきときみも
    regex fullmatch anchors the kana runs; captures = sung readings

Extractions that disagree with fugashi become entries in
``overrides/<song>.json`` (the existing tokenize --override mechanism).
Anything ambiguous — unpaired lines, adjacent kanji tokens with no kana
anchor between them, inconsistent repeats — is reported for human review
instead of written. Existing override entries always win (human > auto).

Romaji transcripts carry their own errors (typos, hearing mistakes), so the
report prints every written entry for a human scan; the override file stays
the single human-vetoable source of truth.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import click

_DIGRAPHS = {
    "kya": "きゃ", "kyu": "きゅ", "kyo": "きょ",
    "gya": "ぎゃ", "gyu": "ぎゅ", "gyo": "ぎょ",
    "sha": "しゃ", "shu": "しゅ", "sho": "しょ", "shi": "し",
    "ja": "じゃ", "ju": "じゅ", "jo": "じょ", "ji": "じ",
    "cha": "ちゃ", "chu": "ちゅ", "cho": "ちょ", "chi": "ち",
    "nya": "にゃ", "nyu": "にゅ", "nyo": "にょ",
    "hya": "ひゃ", "hyu": "ひゅ", "hyo": "ひょ",
    "bya": "びゃ", "byu": "びゅ", "byo": "びょ",
    "pya": "ぴゃ", "pyu": "ぴゅ", "pyo": "ぴょ",
    "mya": "みゃ", "myu": "みゅ", "myo": "みょ",
    "rya": "りゃ", "ryu": "りゅ", "ryo": "りょ",
    "tsu": "つ", "fu": "ふ",
    "ti": "てぃ", "di": "でぃ", "tu": "とぅ", "du": "どぅ",
    "fa": "ふぁ", "fi": "ふぃ", "fe": "ふぇ", "fo": "ふぉ",
    "wi": "うぃ", "we": "うぇ",
}

_BASIC = {
    "k": "かきくけこ", "g": "がぎぐげご", "s": "さしすせそ", "z": "ざじずぜぞ",
    "t": "たちつてと", "d": "だぢづでど", "n": "なにぬねの", "h": "はひふへほ",
    "b": "ばびぶべぼ", "p": "ぱぴぷぺぽ", "m": "まみむめも", "y": "やいゆえよ",
    "r": "らりるれろ", "w": "わゐうゑを",
}
_VOWELS = {"a": 0, "i": 1, "u": 2, "e": 3, "o": 4}
_PLAIN = {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}

_KANA_CLASS = "[ぁ-ゖ]"
_HAS_KANJI = re.compile(r"[一-鿿々]")
_PARTICLE_EQUIV = {"は": "[はわ]", "へ": "[へえ]", "を": "[をお]"}


def romaji_to_kana(text: str) -> str:
    """Deterministic Hepburn(+loanword di/ti) -> hiragana. Unknown chars drop."""
    s = unicodedata.normalize("NFKC", text).lower()
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if not ch.isalpha():
            i += 1
            continue
        if ch == "n" and (i + 1 >= len(s) or s[i + 1] == "'" or
                          (s[i + 1].isalpha() and s[i + 1] not in "aiueoy")):
            out.append("ん")
            i += 2 if i + 1 < len(s) and s[i + 1] == "'" else 1
            continue
        if (ch not in "aiueon" and i + 1 < len(s) and s[i + 1] == ch):
            out.append("っ")
            i += 1
            continue
        matched = False
        for ln in (3, 2):
            chunk = s[i:i + ln]
            if chunk in _DIGRAPHS:
                out.append(_DIGRAPHS[chunk])
                i += ln
                matched = True
                break
        if matched:
            continue
        if ch in _VOWELS:
            out.append(_PLAIN[ch])
            i += 1
            continue
        if ch in _BASIC and i + 1 < len(s) and s[i + 1] in _VOWELS:
            out.append(_BASIC[ch][_VOWELS[s[i + 1]]])
            i += 2
            continue
        i += 1
    return "".join(out)


_VOWEL_OF = {"ぁ": "あ", "ぃ": "い", "ぅ": "う", "ぇ": "え", "ぉ": "お",
             "ゃ": "あ", "ゅ": "う", "ょ": "お", "ん": "ん"}
for _v, _group in [("あ", "あかがさざただなはばぱまやらわ"),
                   ("い", "いきぎしじちぢにひびぴみり"),
                   ("う", "うくぐすずっつづぬふぶぷむゆる"),
                   ("え", "えけげせぜてでねへべぺめれ"),
                   ("お", "おこごそぞとどのほぼぽもよろを")]:
    for _g in _group:
        _VOWEL_OF[_g] = _v


def kana_norm(text: str) -> str:
    """katakana -> hiragana, 長音 -> repeated vowel, drop non-kana."""
    s = unicodedata.normalize("NFKC", text or "")
    out: list[str] = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            ch = chr(code - 0x60)
        if ch == "ー" and out:
            out.append(_VOWEL_OF.get(out[-1], "あ"))
            continue
        if "ぁ" <= ch <= "ゖ":
            out.append(ch)
    return "".join(out)


def _is_latin_or_junk(surface: str) -> bool:
    return not _HAS_KANJI.search(surface) and not re.search(r"[ぁ-ゖァ-ヶー]", surface)


def build_line_pattern(
    tokens: list[dict],
    *,
    literal_readings: bool,
) -> tuple[re.Pattern | None, list[int], set[int]]:
    """Regex over one line's tokens.

    ``literal_readings=True`` replaces every kanji capture with the fugashi
    reading verbatim — the fast path: a fullmatch means the whole line agrees
    with the dictionary and needs no extraction (this also avoids lazy-capture
    truncation artifacts on lines that were never in conflict).

    Returns (pattern, kanji token indices, ambiguous indices — kanji tokens
    adjacent with no kana anchor between them).
    """
    parts: list[str] = []
    kanji_idx: list[int] = []
    ambiguous: set[int] = set()
    prev_was_capture = False
    for i, tok in enumerate(tokens):
        surface = tok["surface"]
        if tok.get("is_punct"):
            # quotes/brackets carry no sound; contributing a wildcard here
            # lets line-final captures leak their tail into it
            prev_was_capture = False
            continue
        if _is_latin_or_junk(surface):
            parts.append(f"{_KANA_CLASS}{{0,6}}?")
            prev_was_capture = False
            continue
        if _HAS_KANJI.search(surface):
            if literal_readings:
                parts.append("".join(_PARTICLE_EQUIV.get(c, re.escape(c))
                                     for c in kana_norm(tok.get("reading") or "")))
            else:
                if prev_was_capture and kanji_idx:
                    ambiguous.add(kanji_idx[-1])
                    ambiguous.add(i)
                parts.append(f"(?P<g{i}>{_KANA_CLASS}{{1,14}}?)")
            kanji_idx.append(i)
            prev_was_capture = True
        else:
            lit = kana_norm(surface)
            if not lit:
                prev_was_capture = False
                continue
            parts.append("".join(_PARTICLE_EQUIV.get(c, re.escape(c)) for c in lit))
            prev_was_capture = False
    if not kanji_idx:
        return None, [], set()
    return re.compile("".join(parts) + "$"), kanji_idx, ambiguous


@click.command()
@click.option("--tokens", "tokens_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="tokens.json from tokenize_lyrics.py (fugashi readings to audit).")
@click.option("--romaji", "romaji_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="User-provided romaji transcript, roughly line-paired with lyrics.txt.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True,
              help="overrides/<song>.json — existing (human) entries always win.")
@click.option("--dry-run", is_flag=True, help="Report only; do not write the override file.")
def main(tokens_path: str, romaji_path: str, out_path: str, dry_run: bool) -> None:
    lines = json.loads(Path(tokens_path).read_text(encoding="utf-8"))
    romaji_lines = [ln.strip() for ln in Path(romaji_path).read_text(encoding="utf-8").splitlines()
                    if ln.strip()]
    romaji_kana = [kana_norm(romaji_to_kana(ln)) for ln in romaji_lines]

    candidates: dict[str, set[str]] = {}
    manual: list[str] = []
    unpaired: list[str] = []
    cursor = 0
    for line in lines:
        literal, _, _ = build_line_pattern(line["tokens"], literal_readings=True)
        pattern, kanji_idx, ambiguous = build_line_pattern(line["tokens"], literal_readings=False)
        if pattern is None:
            continue
        match = None
        clean = False
        for offset in range(3):
            j = cursor + offset
            if j >= len(romaji_kana):
                break
            if literal is not None and literal.match(romaji_kana[j]):
                cursor = j + 1
                clean = True
                break
            match = pattern.match(romaji_kana[j])
            if match:
                cursor = j + 1
                break
        if clean:
            continue
        if not match:
            unpaired.append(line["text"])
            continue
        for i in kanji_idx:
            tok = line["tokens"][i]
            extracted = match.group(f"g{i}")
            expected = kana_norm(tok.get("reading") or "")
            if i in ambiguous:
                if extracted != expected:
                    manual.append(f"{tok['surface']}: fugashi={expected} romaji~{extracted} (相鄰漢字無錨,人工裁決)")
                continue
            if extracted and extracted != expected:
                candidates.setdefault(tok["surface"], set()).add(extracted)

    existing: dict[str, str] = {}
    out_p = Path(out_path)
    if out_p.exists():
        existing = json.loads(out_p.read_text(encoding="utf-8"))

    written: dict[str, str] = {}
    for surface, readings in sorted(candidates.items()):
        if len(readings) > 1:
            manual.append(f"{surface}: 多行萃取不一致 {sorted(readings)},人工裁決")
            continue
        reading = next(iter(readings))
        if surface in existing:
            if existing[surface] != reading:
                manual.append(f"{surface}: 既有 override={existing[surface]} vs romaji={reading},人工保留既有")
            continue
        written[surface] = reading

    click.echo(f"[romaji-overrides] {len(lines)} lyric lines, {len(unpaired)} unpaired, "
               f"{len(written)} new override(s), {len(manual)} for human review")
    for surface, reading in written.items():
        fug = next((kana_norm(t.get("reading") or "") for ln in lines for t in ln["tokens"]
                    if t["surface"] == surface), "?")
        click.echo(f"  + {surface}: {fug} -> {reading}")
    for item in manual:
        click.echo(f"  ! {item}")
    for text in unpaired:
        click.echo(f"  ~ unpaired: {text}")

    if dry_run:
        return
    merged = {**existing, **written}
    if not merged:
        click.echo("[romaji-overrides] nothing to write")
        return
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    click.echo(f"[romaji-overrides] wrote {out_p} ({len(merged)} entries)")


if __name__ == "__main__":
    main()
