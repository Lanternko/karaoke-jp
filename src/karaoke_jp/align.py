"""Kana-aware alignment of faster-whisper ASR to a known lyrics.txt.

The ASR output has the right *timing* but wrong *characters* (Whisper
hallucinates kanji or transcribes phonetically as kana mid-word). The lyrics
file has the right *characters* but no timing. We need to make every lyrics
char inherit an ASR timestamp.

**Why we don't just NW on the raw character streams**: ASR may emit
``ぜったいれいど`` while the lyrics has ``絶対零度`` — those share zero
characters but are pronounced identically. A naive Levenshtein call drops
all four kanji as indels and pushes their timestamps to the next matching
chunk, causing visible drift in the karaoke wipe.

**The fix**: normalize both streams to a hiragana stream via fugashi+UniDic
(reading per token, surface fallback for kana-only tokens), DTW on those,
then map the ASR-kana times back to original lyrics char positions.

A kana-aware path also gives us a future SOFA upgrade slot at M3 v2 — SOFA
emits phoneme timestamps which sit one level below kana, so the same
``kana → original char`` back-map applies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .ruby import kata_to_hira

# Whisper hallucinations on Japanese audio. Drop entire segments whose text
# matches any of these.
HALLUCINATION_SUBSTRINGS = [
    "ご視聴ありがとう",
    "ご清聴ありがとう",
    "編曲 初音ミク",
    "作詞 初音ミク",
    "Thanks for watching",
    "thanks for watching",
]


@dataclass
class AsrChar:
    char: str
    start: float
    end: float
    prob: float


@dataclass
class LyricsChar:
    char: str
    line_idx: int
    char_idx_in_line: int


@dataclass
class KanaUnit:
    """One hiragana char with a back-pointer to its source-char range.

    For ASR units we additionally carry the time interval that this kana was
    sung over (linearly interpolated within the originating word's per-char
    timestamps).
    """
    kana: str
    src_start: int  # inclusive index into the originating char stream
    src_end: int    # exclusive
    t_start: float = 0.0
    t_end: float = 0.0


def _read_kana(feat) -> str:
    """Pull a hiragana reading off a fugashi token feature, falling back gracefully."""
    raw = getattr(feat, "kana", None) or getattr(feat, "pron", None) or ""
    return kata_to_hira(raw) if raw else ""


def text_to_kana_units(text: str, tagger) -> list[KanaUnit]:
    """Tokenize a flat text and emit one ``KanaUnit`` per kana of the reading.

    Mixed-script text such as ``絶対零度`` becomes ``[ぜ, っ, た, い, れ, い, ど]``,
    each one carrying ``src_start..src_end`` pointing at the surface span it
    came from (so ぜっ・たい・れい・ど all share the [0..4) span).
    """
    out: list[KanaUnit] = []
    src_pos = 0
    for word in tagger(text):
        surface = word.surface
        if not surface:
            continue
        if surface.isspace() or surface == "　":
            src_pos += len(surface)
            continue
        kana_hira = _read_kana(word.feature)
        if not kana_hira:
            # Punctuation / symbols / unrecognised: emit the surface chars
            # themselves so they still anchor in the alignment.
            kana_hira = surface
        src_end = src_pos + len(surface)
        for kc in kana_hira:
            out.append(KanaUnit(kana=kc, src_start=src_pos, src_end=src_end))
        src_pos = src_end
    return out


def asr_chars_to_kana_units(asr_chars: list[AsrChar], tagger) -> list[KanaUnit]:
    """Same as ``text_to_kana_units`` but each kana also gets a time slice
    interpolated within its originating surface span."""
    text = "".join(c.char for c in asr_chars)
    out: list[KanaUnit] = []
    src_pos = 0
    for word in tagger(text):
        surface = word.surface
        if not surface:
            continue
        if surface.isspace() or surface == "　":
            src_pos += len(surface)
            continue
        kana_hira = _read_kana(word.feature)
        if not kana_hira:
            kana_hira = surface
        src_end = min(src_pos + len(surface), len(asr_chars))
        if src_pos >= len(asr_chars):
            break
        anchor_start = asr_chars[src_pos].start
        anchor_end = asr_chars[src_end - 1].end
        span = max(anchor_end - anchor_start, 1e-3)
        n = len(kana_hira)
        for i, kc in enumerate(kana_hira):
            t_s = anchor_start + span * i / n
            t_e = anchor_start + span * (i + 1) / n
            out.append(
                KanaUnit(
                    kana=kc,
                    src_start=src_pos,
                    src_end=src_end,
                    t_start=t_s,
                    t_end=t_e,
                )
            )
        src_pos = src_end
    return out


def load_asr_chars(asr_path: Path) -> list[AsrChar]:
    """Flatten ASR words into per-character events. Drops obvious hallucinations."""
    data = json.loads(asr_path.read_text(encoding="utf-8"))
    out: list[AsrChar] = []
    dropped = 0
    for seg in data["segments"]:
        text = seg["text"].strip()
        if any(p in text for p in HALLUCINATION_SUBSTRINGS):
            dropped += 1
            continue
        for w in seg["words"]:
            word = w["word"].strip()
            if not word:
                continue
            dur = max(w["end"] - w["start"], 1e-3)
            per = dur / len(word)
            for i, c in enumerate(word):
                if c.isspace():
                    continue
                out.append(
                    AsrChar(
                        char=c,
                        start=w["start"] + i * per,
                        end=w["start"] + (i + 1) * per,
                        prob=w["probability"],
                    )
                )
    if dropped:
        print(f"[align] dropped {dropped} hallucinated segments")
    return out


def load_lyrics_chars(tokens_json_path: Path) -> tuple[list[LyricsChar], list[dict]]:
    """Flatten the per-line tokenized lyrics into per-character with line metadata."""
    lines = json.loads(tokens_json_path.read_text(encoding="utf-8"))
    out: list[LyricsChar] = []
    for li, line in enumerate(lines):
        for ci, ch in enumerate(line["text"]):
            if ch.isspace() or ch == "　":
                continue
            out.append(LyricsChar(char=ch, line_idx=li, char_idx_in_line=ci))
    return out, lines


def needleman_wunsch_kana(
    asr_kanas: list[KanaUnit], lyr_kanas: list[KanaUnit]
) -> list[tuple[int | None, int | None]]:
    """Global alignment on the two kana streams. Match=0, sub=1, indel=1.

    Returns pairs of (asr_kana_idx, lyrics_kana_idx); either side may be None
    for an indel.
    """
    n, m = len(asr_kanas), len(lyr_kanas)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = asr_kanas[i - 1].kana
        row = dp[i]
        prev_row = dp[i - 1]
        for j in range(1, m + 1):
            sub = 0 if ai == lyr_kanas[j - 1].kana else 1
            row[j] = min(
                prev_row[j - 1] + sub,
                prev_row[j] + 1,
                row[j - 1] + 1,
            )

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 and j > 0:
        sub = 0 if asr_kanas[i - 1].kana == lyr_kanas[j - 1].kana else 1
        if dp[i][j] == dp[i - 1][j - 1] + sub:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    while i > 0:
        pairs.append((i - 1, None))
        i -= 1
    while j > 0:
        pairs.append((None, j - 1))
        j -= 1
    pairs.reverse()
    return pairs


def assign_kana_aware_timestamps(
    lyr_chars: list[LyricsChar],
    asr_kanas: list[KanaUnit],
    lyr_kanas: list[KanaUnit],
    pairs: list[tuple[int | None, int | None]],
) -> list[tuple[float, float]]:
    """For each lyrics char, return ``(start, end)`` derived from the kana
    alignment.

    Each lyrics kana that maps back to char index ``c`` contributes its
    aligned ASR kana's time interval. We take the union (min start, max end)
    across all contributing kana so kanji compounds like ``絶対`` get the full
    ぜったい sung-time range, not just the first kana.

    Lyrics chars with no aligned kana fall back to linear interpolation
    between known neighbours, same as before.
    """
    n_chars = len(lyr_chars)

    # For each lyr_kana: which asr_kana is it aligned to (if any)?
    asr_for_lyr: list[int | None] = [None] * len(lyr_kanas)
    for ai, li in pairs:
        if ai is not None and li is not None:
            asr_for_lyr[li] = ai

    # For each lyrics char: which lyr_kana indices map back to it?
    char_to_kanas: list[list[int]] = [[] for _ in range(n_chars)]
    for k_idx, lk in enumerate(lyr_kanas):
        for c_idx in range(lk.src_start, min(lk.src_end, n_chars)):
            char_to_kanas[c_idx].append(k_idx)

    raw: list[tuple[float, float] | None] = [None] * n_chars
    for c_idx in range(n_chars):
        spans = []
        for k_idx in char_to_kanas[c_idx]:
            a_idx = asr_for_lyr[k_idx]
            if a_idx is not None:
                spans.append((asr_kanas[a_idx].t_start, asr_kanas[a_idx].t_end))
        if spans:
            raw[c_idx] = (min(s[0] for s in spans), max(s[1] for s in spans))

    out: list[tuple[float, float]] = []
    for c_idx in range(n_chars):
        if raw[c_idx] is not None:
            out.append(raw[c_idx])
            continue
        prev_idx = next((k for k in range(c_idx - 1, -1, -1) if raw[k] is not None), None)
        next_idx = next((k for k in range(c_idx + 1, n_chars) if raw[k] is not None), None)
        if prev_idx is None and next_idx is None:
            out.append((0.0, 0.0))
        elif prev_idx is None:
            t = raw[next_idx][0]
            out.append((t, t))
        elif next_idx is None:
            t = raw[prev_idx][1]
            out.append((t, t))
        else:
            t0 = raw[prev_idx][1]
            t1 = raw[next_idx][0]
            span = next_idx - prev_idx
            offset = c_idx - prev_idx
            t = t0 + (t1 - t0) * offset / span
            out.append((t, t))
    return out


def build_aligned_lines(
    lyrics_lines: list[dict],
    lyrics_chars: list[LyricsChar],
    timestamps: list[tuple[float, float]],
) -> list[dict]:
    """Group per-char timestamps back into per-line records with rich tokens."""
    chars_by_line: dict[int, list[tuple[LyricsChar, tuple[float, float]]]] = {}
    for lc, ts in zip(lyrics_chars, timestamps, strict=False):
        chars_by_line.setdefault(lc.line_idx, []).append((lc, ts))

    out: list[dict] = []
    for li, line in enumerate(lyrics_lines):
        chars = chars_by_line.get(li, [])
        # Walk tokens in order, consume the right number of non-space chars
        # per token surface to attach token-level timing.
        char_iter = iter(chars)
        rich_tokens = []
        for tok in line["tokens"]:
            surface = tok["surface"]
            n_real = sum(1 for ch in surface if not ch.isspace() and ch != "　")
            grabbed: list[tuple[LyricsChar, tuple[float, float]]] = []
            for _ in range(n_real):
                try:
                    grabbed.append(next(char_iter))
                except StopIteration:
                    break
            if grabbed:
                t_start = grabbed[0][1][0]
                t_end = grabbed[-1][1][1]
                char_times = [
                    {"char": g[0].char, "start": round(g[1][0], 3), "end": round(g[1][1], 3)}
                    for g in grabbed
                ]
            else:
                t_start = t_end = 0.0
                char_times = []
            rich_tokens.append(
                {
                    **tok,
                    "start": round(t_start, 3),
                    "end": round(t_end, 3),
                    "chars": char_times,
                }
            )
        if rich_tokens:
            line_start = rich_tokens[0]["start"]
            line_end = max(t["end"] for t in rich_tokens)
        else:
            line_start = line_end = 0.0
        out.append(
            {
                "text": line["text"],
                "start": round(line_start, 3),
                "end": round(line_end, 3),
                "tokens": rich_tokens,
            }
        )
    return out


def emit_enhanced_lrc(aligned_lines: list[dict], out_path: Path) -> None:
    """Emit an enhanced-LRC with per-character `<mm:ss.ff>` markers.

    No ruby in this format; M4 renderer will read the sidecar JSON for
    furigana data. This LRC is a fallback for any LRC-aware player.
    """

    def fmt(t: float) -> str:
        m = int(t // 60)
        s = t - m * 60
        return f"{m:02d}:{s:05.2f}"

    out_lines = []
    for line in aligned_lines:
        if not line["tokens"]:
            continue
        head = f"[{fmt(line['start'])}]"
        body_parts = []
        for tok in line["tokens"]:
            for ch in tok["chars"]:
                body_parts.append(f"<{fmt(ch['start'])}>{ch['char']}")
        out_lines.append(head + "".join(body_parts))
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
