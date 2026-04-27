"""Align ASR output to lyrics.txt and emit aligned.json + ruby.lrc.

Usage:
    python scripts/align_lyrics.py \\
        --asr outputs/<song>/asr.json \\
        --tokens outputs/<song>/tokens.json \\
        --aligned-out outputs/<song>/aligned.json \\
        --lrc-out outputs/<song>/ruby.lrc
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import click

from karaoke_jp.align import (
    assign_timestamps,
    build_aligned_lines,
    emit_enhanced_lrc,
    load_asr_chars,
    load_lyrics_chars,
    needleman_wunsch,
)


@click.command()
@click.option("--asr", "asr_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--tokens", "tokens_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--aligned-out", type=click.Path(dir_okay=False), required=True)
@click.option("--lrc-out", type=click.Path(dir_okay=False), required=True)
def main(asr_path: str, tokens_path: str, aligned_out: str, lrc_out: str) -> None:
    asr_chars = load_asr_chars(Path(asr_path))
    lyrics_chars, lyrics_lines = load_lyrics_chars(Path(tokens_path))

    print(f"[align] {len(asr_chars)} ASR chars vs {len(lyrics_chars)} lyrics chars")

    pairs = needleman_wunsch(asr_chars, lyrics_chars)
    matches = sum(
        1
        for ai, li in pairs
        if ai is not None and li is not None and asr_chars[ai].char == lyrics_chars[li].char
    )
    subs = sum(
        1
        for ai, li in pairs
        if ai is not None and li is not None and asr_chars[ai].char != lyrics_chars[li].char
    )
    asr_dels = sum(1 for ai, li in pairs if li is None)
    lyr_inserts = sum(1 for ai, li in pairs if ai is None)
    print(
        f"[align] alignment: {matches} matches, {subs} substitutions, "
        f"{asr_dels} ASR-only chars dropped, {lyr_inserts} lyrics chars without ASR support"
    )

    timestamps = assign_timestamps(asr_chars, lyrics_chars, pairs)
    aligned = build_aligned_lines(lyrics_lines, lyrics_chars, timestamps)

    Path(aligned_out).parent.mkdir(parents=True, exist_ok=True)
    Path(aligned_out).write_text(
        json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    emit_enhanced_lrc(aligned, Path(lrc_out))
    print(f"[align] wrote {aligned_out} + {lrc_out}")


if __name__ == "__main__":
    main()
