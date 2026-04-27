"""Char-level alignment of faster-whisper ASR to a known lyrics.txt.

The ASR output has the right *timing* but wrong *characters* (Whisper makes
mistakes singing). The lyrics file has the right *characters* but no timing.
We DTW-align the two character streams so each lyrics char inherits an ASR
timestamp.

Why char-level: faster-whisper on Japanese already emits per-char timestamps
(the tokenizer is BPE-on-bytes), so we get fine granularity for free without
needing a separate forced-alignment model. SOFA at M3 v2 will improve
accuracy on melismas and quiet entries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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


def needleman_wunsch(
    asr: list[AsrChar], lyr: list[LyricsChar]
) -> list[tuple[int | None, int | None]]:
    """Global alignment. Match=0, sub=1, indel=1.

    Returns pairs of (asr_idx, lyrics_idx); either side may be None for an
    indel.
    """
    n, m = len(asr), len(lyr)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = asr[i - 1].char
        row = dp[i]
        prev_row = dp[i - 1]
        for j in range(1, m + 1):
            sub = 0 if ai == lyr[j - 1].char else 1
            row[j] = min(
                prev_row[j - 1] + sub,
                prev_row[j] + 1,
                row[j - 1] + 1,
            )

    pairs: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 and j > 0:
        sub = 0 if asr[i - 1].char == lyr[j - 1].char else 1
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


def assign_timestamps(
    asr: list[AsrChar],
    lyr: list[LyricsChar],
    pairs: list[tuple[int | None, int | None]],
) -> list[tuple[float, float]]:
    """For each lyrics char, return (start, end) inferred from the alignment.

    Unaligned lyrics chars (insertions w.r.t. ASR) interpolate linearly
    between known neighbors.
    """
    n = len(lyr)
    raw: list[tuple[float, float] | None] = [None] * n
    for ai, li in pairs:
        if li is not None and ai is not None and raw[li] is None:
            raw[li] = (asr[ai].start, asr[ai].end)

    out: list[tuple[float, float]] = []
    for li in range(n):
        if raw[li] is not None:
            out.append(raw[li])
            continue
        # Find prev / next known.
        prev_idx = None
        for k in range(li - 1, -1, -1):
            if raw[k] is not None:
                prev_idx = k
                break
        next_idx = None
        for k in range(li + 1, n):
            if raw[k] is not None:
                next_idx = k
                break

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
            offset = li - prev_idx
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
