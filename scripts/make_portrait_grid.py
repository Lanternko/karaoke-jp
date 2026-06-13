#!/usr/bin/env python3
"""Portrait (9:16) display grid for karaoke pitch bars.

Two INDEPENDENT line systems, each alternating between display rows A/B:

  bar lines   — pitch bars packed greedily ("snake"): phrases flow into a
                row until the width budget runs out, then jump to the next
                row. NOT tied to lyric line boundaries.
  lyric lines — one line of text per aligned_midi.json line, alternating
                A/B by line index (JOYSOUND subtitle style).

Each line carries its own real-time wipe data; the renderer wipes bars and
lyrics independently of each other, driven purely by real_start/real_end.

Outputs a JSON sidecar consumed by render_portrait.py.
"""
from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import make_display_grid as mdg  # noqa: E402
from karaoke_jp.lrc_export import split_furigana  # noqa: E402
from karaoke_jp.midi_markers import (  # noqa: E402
    _intersect_windows,
    _lyric_windows,
    _voiced_windows,
)
from karaoke_jp.score_melody import read_midi_notes  # noqa: E402

Note = tuple[float, float, int]


def _char_windows(aligned: list[dict], *, pad: float = 0.35) -> list[tuple[float, float]]:
    """Merged windows around every MMS char — direct sung-voice evidence.

    The RMS VAD misses soft/breathy passages (night-dancer's Tu-tu-lu hook),
    but the CTC aligner heard every char there. Union-ing these windows with
    the voiced windows keeps the interlude ghost-note protection (no chars in
    an interlude) without killing softly sung notes.
    """
    spans = sorted(
        (ch["start"] - pad, ch["end"] + pad)
        for line in aligned for tok in (line.get("tokens") or [])
        for ch in tok.get("chars", []) if ch["end"] >= ch["start"])
    merged: list[tuple[float, float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _union_windows(a: list[tuple[float, float]],
                   b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for s, e in sorted(a + b):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def unwarp_notes(notes: list[Note], warp_path: str | Path) -> list[Note]:
    """Map display-timeline notes back to real time via the warp sidecar.

    The canonical chain (run_game_chain.py) only keeps the DISPLAY MIDI from
    make_display_grid — its note times include page leads, count-ins and
    breath spaces. The .warp.json anchors (real<->display, piecewise linear)
    invert that, recovering the real-time union melody the chain discarded.
    """
    import numpy as np

    w = json.loads(Path(warp_path).read_text(encoding="utf-8"))
    disp = np.asarray(w["display"], dtype=float)
    real = np.asarray(w["real"], dtype=float)
    out = []
    for s, e, p in notes:
        rs = float(np.interp(s, disp, real))
        re = float(np.interp(e, disp, real))
        if re - rs > 0.01:
            out.append((rs, re, p))
    return sorted(out)


# ---- per-mora note splitting ----

def split_notes_at_chars(
    notes: list[Note], aligned: list[dict], *, min_piece: float = 0.12,
) -> tuple[list[Note], int]:
    """Split each note at MMS char onsets strictly inside its span.

    GAME's segmentation is acoustic: legato same-pitch morae fuse into one
    note ("飛ん" -> one bar). Karaoke wants one bar per mora, so we cut every
    note at the char onsets that land inside it (keeping pitch). Display-only,
    and only meaningful AFTER drop_fragments/absorb_wiggles have run — placed
    before them the slivers would be folded straight back in.

    Every emitted piece is >= min_piece, so cuts closer than that to either
    edge (or to the previous accepted cut) are skipped. Returns
    (sorted_notes, pieces_added).
    """
    onsets = sorted(
        ch["start"]
        for line in aligned for tok in (line.get("tokens") or [])
        for ch in tok.get("chars", []) if ch["end"] > ch["start"])

    out: list[Note] = []
    added = 0
    for s, e, p in notes:
        lo = bisect.bisect_left(onsets, s + min_piece)
        hi = bisect.bisect_right(onsets, e - min_piece)
        prev = s
        for t in onsets[lo:hi]:
            if t - prev >= min_piece and e - t >= min_piece:
                out.append((prev, t, p))
                prev = t
                added += 1
        out.append((prev, e, p))
    return sorted(out), added


# ---- bar lines: one row-group per sentence, greedy to the right edge ----

def _close_bar_row(bars: list[dict], cursor: float, quarter: float,
                   row_idx: int, sent_id: int, sent_start: float) -> dict:
    return {
        "row": "A" if row_idx % 2 == 0 else "B",
        "bars": bars,
        "width_q": cursor / quarter,
        "time_start": bars[0]["real_start"],
        "time_end": bars[-1]["real_end"],
        # the lyric line this row belongs to, and when that whole line
        # starts being sung — so wrapped continuation rows stay "upcoming"
        # (not previewed-grey) until the sentence actually begins.
        "sent": sent_id,
        "sent_start": sent_start,
    }


def build_bar_lines(
    notes: list[Note],
    qnotes: list[Note],
    line_of: list[int],
    *,
    row_budget: float,
    gap: float,
    breath_space: float,
    breath_gap: float,
    pause_gap: float,
    pause_space: float,
    quarter: float,
) -> list[dict]:
    """One row-group per lyric sentence; pack notes left->right and wrap to a
    new row only when the next note would overflow the row.

    This matches reading order and Kojek's two break rules: (1) the row is
    full (the next bar would exceed row_budget -> wrap, like a line of text
    hitting the page edge), (2) a new sentence always starts a fresh row.
    A long in-sentence pause (> pause_gap) opens a WIDE gap (pause_space)
    but never forces a break; an ordinary breath (> breath_gap) opens the
    usual breath_space. Rows alternate A/B in production order.
    """
    if not notes:
        return []

    # group note indices by sentence (line_of is monotonic non-decreasing)
    sentences: list[list[int]] = [[0]]
    for i in range(1, len(notes)):
        if line_of[i] != line_of[i - 1]:
            sentences.append([i])
        else:
            sentences[-1].append(i)

    lines: list[dict] = []
    row_idx = 0
    for sent in sentences:
        sent_id = line_of[sent[0]]
        sent_start = notes[sent[0]][0]
        bars: list[dict] = []
        cursor = 0.0
        prev: int | None = None
        for i in sent:
            slot = qnotes[i][1] - qnotes[i][0]
            if prev is None:
                space = 0.0
            else:
                gap_t = notes[i][0] - notes[prev][1]
                if gap_t > pause_gap:
                    space = pause_space
                elif gap_t > breath_gap:
                    space = breath_space
                else:
                    space = 0.0
            if bars and cursor + space + slot > row_budget:
                lines.append(_close_bar_row(bars, cursor, quarter, row_idx,
                                            sent_id, sent_start))
                row_idx += 1
                bars = []
                cursor = 0.0
                space = 0.0
            cursor += space
            bars.append({
                "x_q": cursor / quarter,
                "w_q": max(slot - gap, gap) / quarter,
                "pitch": notes[i][2],
                "real_start": notes[i][0],
                "real_end": notes[i][1],
            })
            cursor += slot
            prev = i
        if bars:
            lines.append(_close_bar_row(bars, cursor, quarter, row_idx,
                                        sent_id, sent_start))
            row_idx += 1
    return lines


# ---- lyric lines ----

def _extract_line_chars(line: dict) -> list[dict]:
    """Flat char list from one aligned_midi.json line, with ruby info.

    Ruby is attached per kanji-RUN via split_furigana (never the whole
    token: 足踏み -> あしぶ over 足踏 only, not over み). `ruby_span` is how
    many chars the reading covers so the renderer can center it over the
    full run (余所 -> よそ spans both glyphs, not just 余).
    """
    chars: list[dict] = []
    for tok in line.get("tokens", []):
        tok_chars = tok.get("chars", [])
        base = len(chars)
        for ch in tok_chars:
            chars.append({
                "char": ch["char"],
                "real_start": ch["start"],
                "real_end": ch["end"],
            })
        reading = tok.get("reading")
        if tok.get("kana_only", True) or not reading:
            continue
        surface = "".join(ch["char"] for ch in tok_chars)
        for _seg, ruby, cs, ce in split_furigana(surface, reading):
            if ruby:
                chars[base + cs]["ruby"] = ruby
                chars[base + cs]["ruby_span"] = ce - cs
    return chars


def build_lyric_lines(aligned: list[dict]) -> list[dict]:
    lines: list[dict] = []
    for line in aligned:
        if not line.get("tokens"):
            continue
        chars = _extract_line_chars(line)
        if not chars:
            continue
        lines.append({
            "row": "A" if len(lines) % 2 == 0 else "B",
            "text": line["text"],
            "chars": chars,
            "time_start": chars[0]["real_start"],
            "time_end": chars[-1]["real_end"],
        })
    return lines


# ---- visibility timing (shared by both line systems) ----

def compute_line_timing(
    lines: list[dict], *, lead_max: float = 8.0, linger: float = 4.0,
) -> None:
    """Set preview_start / replace_time per display row.

    JOYSOUND flip rule: a line is replaced on its row only when the NEXT
    line overall (on the other row) starts being sung — never while its own
    content is still going (long sustains stay put). In an interlude the
    row clears `linger` after the line ends; the upcoming line previews at
    most lead_max before its own start.
    """
    order = sorted(range(len(lines)), key=lambda i: lines[i]["time_start"])
    succ_start = {}
    for k, i in enumerate(order):
        succ_start[i] = (lines[order[k + 1]]["time_start"]
                         if k + 1 < len(order) else None)

    by_row: dict[str, list[int]] = {}
    for i in order:
        by_row.setdefault(lines[i]["row"], []).append(i)
    for row_idx in by_row.values():
        for k, i in enumerate(row_idx):
            ln = lines[i]
            if k == 0:
                ln["preview_start"] = max(0.0, ln["time_start"] - lead_max)
            flip = max(ln["time_end"], succ_start[i] or ln["time_end"])
            flip = min(flip, ln["time_end"] + linger)
            if k + 1 < len(row_idx):
                nxt = lines[row_idx[k + 1]]
                nxt["preview_start"] = max(flip, nxt["time_start"] - lead_max)
                ln["replace_time"] = min(flip, nxt["preview_start"])
            else:
                ln["replace_time"] = ln["time_end"] + linger


# ---- CLI ----

@click.command()
@click.option("--midi", "midi_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--warp", "warp_path", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Display->real warp sidecar; REQUIRED when --midi is a "
                   "display MIDI from run_game_chain (the canonical output).")
@click.option("--bpm-file", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--quarters-per-row", type=float, default=8.0, show_default=True)
@click.option("--gap-units", type=float, default=0.0625, show_default=True)
@click.option("--phrase-gap-units", type=float, default=0.75, show_default=True,
              help="DEPRECATED (sentence-grouped packing ignores it).")
@click.option("--phrase-split-gap", type=float, default=0.8, show_default=True,
              help="DEPRECATED (sentence-grouped packing ignores it).")
@click.option("--min-note", type=float, default=0.09, show_default=True)
@click.option("--breath-gap", type=float, default=0.25, show_default=True)
@click.option("--breath-units", type=float, default=0.5, show_default=True)
@click.option("--pause-gap", type=float, default=1.0, show_default=True,
              help="In-sentence silence over this (s) opens a wide gap, no break.")
@click.option("--pause-units", type=float, default=1.5, show_default=True,
              help="Width (quarters) of that wide in-sentence pause gap.")
@click.option("--note-window-margin", type=float, default=0.25, show_default=True)
@click.option("--note-tail-allowance", type=float, default=1.0, show_default=True)
@click.option("--lead-max", type=float, default=8.0, show_default=True,
              help="Max seconds a line previews before its first note/char.")
@click.option("--linger", type=float, default=2.0, show_default=True,
              help="Seconds a finished line stays before the row clears.")
@click.option("--pitch-patch", "pitch_patch_path",
              type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--rms-segments", "rms_segments_path",
              type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
def main(midi_path, warp_path, bpm_file, aligned_path, quarters_per_row,
         gap_units, phrase_gap_units, phrase_split_gap, min_note, breath_gap,
         breath_units, pause_gap, pause_units, note_window_margin,
         note_tail_allowance, lead_max, linger, pitch_patch_path,
         rms_segments_path, out_path):
    bpm_2dp = round(float(Path(bpm_file).read_text().strip()), 2)
    quarter = 60.0 / bpm_2dp
    notes: list[Note] = sorted(
        (n.start, n.end, n.pitch) for n in read_midi_notes(Path(midi_path)))
    if warp_path:
        notes = unwarp_notes(notes, warp_path)

    aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))

    # gate to lyric windows ∩ (RMS voiced ∪ MMS char evidence)
    windows = _lyric_windows(aligned_path, margin=note_window_margin,
                             tail_allowance=note_tail_allowance)
    if rms_segments_path:
        voiced = _union_windows(_voiced_windows(rms_segments_path),
                                _char_windows(aligned))
        windows = _intersect_windows(windows, voiced)
    notes = [n for n in notes
             if any(ws <= n[0] < we for ws, we in windows)
             or any(n[0] < we and n[1] > ws for ws, we in windows)]

    line_starts = mdg.line_starts_from_aligned(aligned_path)
    notes, dropped = mdg.drop_fragments(notes, min_note=min_note)
    notes, wiggles = mdg.absorb_wiggles(notes)

    patched = 0
    if pitch_patch_path:
        patches = json.loads(Path(pitch_patch_path).read_text(encoding="utf-8"))
        notes, patched, missed = mdg.apply_pitch_patch(notes, patches)
        for t in missed:
            click.echo(f"[portrait-grid] WARNING: pitch patch at {t:.2f}s matched no note")

    # cut fused legato notes at MMS char onsets — one bar per mora (after
    # drop/absorb so the slivers are not folded straight back in).
    notes, mora_split = split_notes_at_chars(notes, aligned)

    qnotes = mdg.quantize(notes, quarter=quarter)
    line_of = mdg.assign_lines(notes, line_starts)

    gap = gap_units * quarter
    breath_space = breath_units * quarter
    row_budget = quarters_per_row * quarter

    bar_lines = build_bar_lines(
        notes, qnotes, line_of,
        row_budget=row_budget, gap=gap, breath_space=breath_space,
        breath_gap=breath_gap, pause_gap=pause_gap,
        pause_space=pause_units * quarter, quarter=quarter)

    lyric_lines = build_lyric_lines(aligned)

    compute_line_timing(bar_lines, lead_max=lead_max, linger=linger)
    compute_line_timing(lyric_lines, lead_max=lead_max, linger=linger)

    pitches = [n[2] for n in notes]
    pitch_min, pitch_max = min(pitches), max(pitches)

    output = {
        "bpm": bpm_2dp,
        "quarter_sec": quarter,
        "quarters_per_row": quarters_per_row,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "bar_lines": bar_lines,
        "lyric_lines": lyric_lines,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))

    click.echo(f"[portrait-grid] bar_lines={len(bar_lines)} "
               f"lyric_lines={len(lyric_lines)} "
               f"dropped={dropped} wiggles={wiggles} patched={patched} "
               f"mora_split={mora_split} "
               f"pitch=[{pitch_min},{pitch_max}] quarter={quarter:.3f}s")


if __name__ == "__main__":
    main()
