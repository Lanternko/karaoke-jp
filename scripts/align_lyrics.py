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

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from karaoke_jp.align import (
    asr_chars_to_kana_units,
    assign_kana_aware_timestamps,
    build_aligned_lines,
    emit_enhanced_lrc,
    load_asr_chars,
    load_lyrics_chars,
    lyrics_tokens_to_kana_units,
    needleman_wunsch_kana,
)


@click.command()
@click.option("--asr", "asr_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--tokens", "tokens_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--aligned-out", type=click.Path(dir_okay=False), required=True)
@click.option("--lrc-out", type=click.Path(dir_okay=False), required=True)
def main(asr_path: str, tokens_path: str, aligned_out: str, lrc_out: str) -> None:
    import fugashi

    asr_chars = load_asr_chars(Path(asr_path))
    lyrics_chars, lyrics_lines = load_lyrics_chars(Path(tokens_path))

    tagger = fugashi.Tagger()
    asr_kanas = asr_chars_to_kana_units(asr_chars, tagger)
    lyr_kanas = lyrics_tokens_to_kana_units(lyrics_lines, tagger)

    print(
        f"[align] streams: {len(asr_chars)} ASR chars -> {len(asr_kanas)} kana, "
        f"{len(lyrics_chars)} lyrics chars -> {len(lyr_kanas)} kana"
    )

    pairs = needleman_wunsch_kana(asr_kanas, lyr_kanas)
    matches = sum(
        1
        for ai, li in pairs
        if ai is not None and li is not None and asr_kanas[ai].kana == lyr_kanas[li].kana
    )
    subs = sum(
        1
        for ai, li in pairs
        if ai is not None and li is not None and asr_kanas[ai].kana != lyr_kanas[li].kana
    )
    asr_dels = sum(1 for ai, li in pairs if li is None)
    lyr_inserts = sum(1 for ai, li in pairs if ai is None)
    print(
        f"[align] kana alignment: {matches} matches ({matches / max(len(lyr_kanas), 1):.0%}), "
        f"{subs} substitutions, {asr_dels} ASR-only kana, {lyr_inserts} lyrics-only kana"
    )

    timestamps = assign_kana_aware_timestamps(lyrics_chars, asr_kanas, lyr_kanas, pairs)
    aligned = build_aligned_lines(lyrics_lines, lyrics_chars, timestamps)

    Path(aligned_out).parent.mkdir(parents=True, exist_ok=True)
    Path(aligned_out).write_text(
        json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    emit_enhanced_lrc(aligned, Path(lrc_out))
    print(f"[align] wrote {aligned_out} + {lrc_out}")


if __name__ == "__main__":
    main()
