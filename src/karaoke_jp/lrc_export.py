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
"""
from __future__ import annotations

import json
from pathlib import Path


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

        body_parts: list[str] = []
        line_end = line["start"]
        for tok in line["tokens"]:
            if not tok["chars"]:
                continue
            for ch in tok["chars"]:
                body_parts.append(_fmt_time(ch["start"]) + ch["char"])
                line_end = max(line_end, ch["end"])

            if tok["reading"] and not tok["kana_only"] and tok["surface"]:
                ruby_id += 1
                t_start = _fmt_time(tok["start"])
                # Pad the ruby's effective range slightly so the parser's
                # body-scan finds the surface even when float rounding shifts
                # the leading tag by a centisecond.
                t_end = _fmt_time(tok["end"] + 0.05)
                ruby_defs.append(
                    f"@Ruby{ruby_id}={tok['surface']},{tok['reading']},{t_start},{t_end}"
                )
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
