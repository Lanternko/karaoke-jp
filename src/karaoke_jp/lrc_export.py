"""Emit MID2BAR-Player flavored LRC from our aligned.json.

Format reverse-engineered from `third_party/MID2BAR-Player/sample/*.lrc` and
`third_party/MID2BAR-Player/lyrics/parse_ruby.py`:

* Time tags are ``[mm:ss:cs]`` — colon-separated **centiseconds**, not the
  standard ``[mm:ss.ms]`` dot-separated milliseconds.
* Per-character body: one ``[mm:ss:cs]char`` per char, with a trailing
  ``[mm:ss:cs]`` marking the line-end time.
* Furigana lives in a header section: ``@RubyN=base,ruby,[start],[end]``.
  The parser scans the body for each ``base`` and applies ``ruby`` only if
  the occurrence's leading time tag falls inside ``[start, end]``. So the
  same surface can appear N times at different verses, each with its own
  ``@RubyN`` whose range is just wide enough to cover that occurrence.

Furigana scope: a ``@Ruby`` is emitted per **kanji run** within a token,
not per token. Otherwise mixed-script tokens like ``ぶつけ合っ`` (verb,
reading ``ぶつけあっ``) would paint reading glyphs on top of the leading
``ぶつけ`` hiragana — which is wrong, only ``合`` should carry ``あ``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_KANJI_RE = re.compile(r"[㐀-鿿豈-﫿]")


def _is_kanji(ch: str) -> bool:
    return bool(_KANJI_RE.match(ch))


def split_furigana(surface: str, reading: str) -> list[tuple[str, str | None, int, int]]:
    """Split a token's surface into (segment, reading_or_None, char_start, char_end_exclusive).

    Hiragana / katakana characters in ``surface`` should appear verbatim at
    the corresponding position in ``reading`` (UniDic readings of okurigana
    match the surface). The reading slice between consecutive kana anchors
    is the furigana for the kanji run that lies between them.

    On any alignment mismatch we fall back to a single segment covering the
    whole surface so the caller emits the legacy single ``@Ruby`` rather than
    silently dropping ruby for the token.
    """
    if not surface or not reading:
        return [(surface, reading or None, 0, len(surface))]

    # 1. Group surface chars into runs of (kanji | not-kanji).
    runs: list[tuple[str, bool, int, int]] = []
    i = 0
    while i < len(surface):
        is_k = _is_kanji(surface[i])
        j = i + 1
        while j < len(surface) and _is_kanji(surface[j]) == is_k:
            j += 1
        runs.append((surface[i:j], is_k, i, j))
        i = j

    # 2. Walk reading left-to-right, anchoring at kana runs.
    result: list[tuple[str, str | None, int, int]] = []
    r_idx = 0
    for k, (seg, is_k, s_start, s_end) in enumerate(runs):
        if not is_k:
            # Kana run: should appear verbatim at reading[r_idx:].
            if reading[r_idx : r_idx + len(seg)] != seg:
                # Mismatch (kata vs hira, etc.). Bail out → whole token ruby.
                return [(surface, reading, 0, len(surface))]
            result.append((seg, None, s_start, s_end))
            r_idx += len(seg)
        else:
            # Kanji run: its reading ends right before the next kana run, or
            # at end-of-string if this is the last run.
            if k + 1 < len(runs):
                next_kana = runs[k + 1][0]
                pos = reading.find(next_kana, r_idx)
                if pos == -1 or pos < r_idx:
                    return [(surface, reading, 0, len(surface))]
                kanji_reading = reading[r_idx:pos]
                r_idx = pos
            else:
                kanji_reading = reading[r_idx:]
                r_idx = len(reading)
            if not kanji_reading:
                # No reading chars left for this kanji run — fall back.
                return [(surface, reading, 0, len(surface))]
            result.append((seg, kanji_reading, s_start, s_end))

    return result


def _fmt_time(t: float) -> str:
    """Format seconds as ``[mm:ss:cs]`` (centiseconds, two digits)."""
    if t < 0:
        t = 0.0
    total_cs = int(round(t * 100))
    cs = total_cs % 100
    total_s = total_cs // 100
    s = total_s % 60
    m = total_s // 60
    return f"[{m:02d}:{s:02d}:{cs:02d}]"


def export_lrc(aligned_path: Path, out_path: Path, *, block_size: int = 2) -> None:
    """Convert aligned.json into MID2BAR's LRC dialect.

    ``block_size`` (1-4): how many lines per "page block". MID2BAR's
    ``calc_display_time`` only supports block_length 1, 2, 3, or 4 — every
    pile of lines between blank LRC lines becomes one block. We default to
    2, which matches JOYSOUND's typical "current line + next line preview"
    layout.
    """
    if block_size not in (1, 2, 3, 4):
        raise ValueError(f"block_size must be 1..4, got {block_size}")

    aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))

    ruby_defs: list[str] = []
    body_lines: list[str] = []
    ruby_id = 0
    in_block = 0

    for line in aligned:
        if not line["tokens"]:
            continue

        # Flatten chars line-wide so each char knows its global index. The
        # @Ruby end-time check inside MID2BAR (lyrics/lrc_tools.py:apply_
        # rubies_to_result) compares lyric_time_end ≤ ruby.end *strictly*,
        # where lyric_time_end is the body time-tag IMMEDIATELY after the
        # matched run. So we must set ruby.end ≥ that next body-tag time,
        # not just the segment's own char.end (which midi_timing now caps
        # at note_off, leaving a breath gap before the next char's body tag).
        flat_chars: list[dict] = [
            ch for tok in line["tokens"] for ch in (tok.get("chars") or [])
        ]
        if not flat_chars:
            continue

        body_parts: list[str] = []
        line_end = line["start"]
        flat_idx = 0  # advances as we step through tokens below
        for tok in line["tokens"]:
            if not tok.get("chars"):
                continue
            tok_chars = tok["chars"]
            for ch in tok_chars:
                body_parts.append(_fmt_time(ch["start"]) + ch["char"])
                line_end = max(line_end, ch["end"])

            if tok["reading"] and not tok["kana_only"] and tok["surface"]:
                segments = split_furigana(tok["surface"], tok["reading"])
                for seg_text, seg_reading, s_start, s_end in segments:
                    if seg_reading is None:
                        continue
                    if s_end > len(tok_chars):
                        seg_t_start = tok["start"]
                        seg_t_end_body = tok["end"]
                    else:
                        seg_t_start = tok_chars[s_start]["start"]
                        # Body tag right after the segment's last char.
                        line_idx_after = flat_idx + s_end
                        if line_idx_after < len(flat_chars):
                            seg_t_end_body = flat_chars[line_idx_after]["start"]
                        else:
                            # Last segment in the line — fall back to line_end.
                            seg_t_end_body = max(
                                tok_chars[s_end - 1]["end"], line_end
                            )
                    ruby_id += 1
                    t_start = _fmt_time(seg_t_start)
                    # Pad upward so float rounding never trips the strict
                    # ≤ check inside MID2BAR. 0.05 s is plenty (centisecond
                    # quantization in our LRC + breath windows are tens of cs).
                    t_end = _fmt_time(seg_t_end_body + 0.05)
                    ruby_defs.append(
                        f"@Ruby{ruby_id}={seg_text},{seg_reading},{t_start},{t_end}"
                    )
            flat_idx += len(tok_chars)
        body_parts.append(_fmt_time(line_end))
        body_lines.append("".join(body_parts))
        in_block += 1
        if in_block >= block_size:
            body_lines.append("")  # blank line ends the block
            in_block = 0

    out_chunks: list[str] = []
    if ruby_defs:
        out_chunks.append("\n".join(ruby_defs))
        out_chunks.append("")
    out_chunks.extend(body_lines)
    out_chunks.append("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(out_chunks), encoding="utf-8")
