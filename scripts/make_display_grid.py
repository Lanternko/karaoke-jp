#!/usr/bin/env python3
"""Standardized display grid for the pitch bars (Kojek spec, 2026-06-11).

Fixed visual vocabulary:
  * a quarter note has a FIXED width; durations snap to 2^-2..2^2 quarters
  * a note's SLOT is exactly its quantized duration; the bar is drawn
    slot - gap, so the inter-mora gap lives inside the note's own beat
    (like engraved sheet music) and on-grid singing gives a CONSTANT
    cursor speed across note values (Kojek v11: 4th+gap vs 2nd+gap were
    not 2:1, so the cursor wobbled even on tempo)
  * a real inter-note silence above the breath threshold inserts a wider
    EXTRA space: the reader must see where the singer breathes
  * the gap between phrases is FIXED and larger
  * every page spans exactly the same display width (1-3 phrases, usually 2)

What varies instead: the cursor speed. The bars live on a *display timeline*;
a piecewise-linear warp maps real song time -> display time so the wipe and
the cursor stay perfectly synced to the audio (each bar wipes across its
quantized width during exactly its real sung duration). The renderer applies
the warp via --time-warp (see render_mp4.py).

Oversized phrase clusters (choruses sung with no >split-gap silence) are cut
preferentially at LYRIC LINE boundaries (from the aligned JSON), so a line is
never broken mid-word just because its largest acoustic gap fell there.

Long rests get TV-style treatment: the page flips shortly after the previous
line ends, the cursor parks at the left edge, then sweeps the lead-in during
the final count-in quarters before the first note.

Display-only: never feed grid output to the eval harness.
"""
from __future__ import annotations

import bisect
import json
import math
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mido  # noqa: E402

from karaoke_jp.midi_markers import (  # noqa: E402
    _intersect_windows,
    _lyric_windows,
    _voiced_windows,
)
from karaoke_jp.score_melody import read_first_tempo_bpm, read_midi_notes  # noqa: E402

Note = tuple[float, float, int]
QUANT_MULTIPLES = (0.25, 0.5, 1.0, 2.0, 4.0)  # 2^-2 .. 2^2 quarters

# a note starting up to this long before its lyric line still belongs to it
LINE_ONSET_TOLERANCE = 0.1


def drop_fragments(notes: list[Note], *, min_note: float) -> tuple[list[Note], int]:
    """Absorb-or-drop sliver fragments (< min_note).

    A fragment that touches the previous kept note (gap <= 50ms) within ±2
    semitones is folded into that note's tail instead of vanishing — it is
    almost always a transcription split of the same sung note. Benchmark-
    proven on MIR-ST500 + Kiritan (all three of COn/COnP/COnPOff up vs pure
    dropping; see benchmarks/*/RESULTS.md note-cleanup ablation).
    """
    out: list[list[float | int]] = []
    removed = 0
    for s, e, p in sorted(notes):
        if e - s >= min_note:
            out.append([s, e, p])
            continue
        removed += 1
        if out and s - out[-1][1] <= 0.05 and abs(p - out[-1][2]) <= 2:
            out[-1][1] = max(out[-1][1], e)  # absorb into previous tail
    return [tuple(n) for n in out], removed


def absorb_wiggles(notes: list[Note], *, max_dur: float = 0.22) -> tuple[list[Note], int]:
    """A short note sandwiched between two SAME-pitch neighbours within ±2
    semitones is vibrato/ornament, not score: relabel it to the neighbours'
    pitch (bars stay separate; mora identity is preserved)."""
    out = [list(n) for n in sorted(notes)]
    fixed = 0
    for i in range(1, len(out) - 1):
        a, b, c = out[i - 1], out[i], out[i + 1]
        if (
            b[1] - b[0] <= max_dur
            and a[2] == c[2]
            and b[2] != a[2]
            and abs(b[2] - a[2]) <= 2
            and b[0] - a[1] < 0.15
            and c[0] - b[1] < 0.15
        ):
            b[2] = a[2]
            fixed += 1
    return [tuple(n) for n in out], fixed


def quantize(notes: list[Note], *, quarter: float) -> list[Note]:
    out = []
    for s, e, p in notes:
        dur = max(e - s, 1e-3)
        target = min((m * quarter for m in QUANT_MULTIPLES),
                     key=lambda d: abs(math.log(dur / d)))
        out.append((s, s + target, p))
    return out


def apply_pitch_patch(
    notes: list[Note],
    patches: list[dict],
) -> tuple[list[Note], int, list[float]]:
    """Ear-verified per-song pitch fixes, display layer ONLY (the eval
    candidates stay untouched — patching them would game the benchmark).

    Each patch: {"at": <real seconds>, "pitch": <midi>, optional "from":
    <midi>, "note": <why>}. Applies to every note whose span contains `at`
    (and matches "from" when given).

    With "start"/"end" the patch is an ENSURE: if no note contains `at`,
    a note [start, end, pitch] is inserted (skipped if it would overlap an
    existing note). Robust to transcription churn — the same sidecar fixes
    the note whether the extractor mislabels it or drops it entirely.
    Returns (notes, applied, missed_ats).
    """
    out = [list(n) for n in notes]
    applied = 0
    missed: list[float] = []
    for patch in patches:
        t = float(patch["at"])
        pitch = int(patch["pitch"])
        hit = False
        for n in out:
            if n[0] <= t < n[1] and ("from" not in patch or n[2] == int(patch["from"])):
                n[2] = pitch
                applied += 1
                hit = True
        if hit:
            continue
        if "start" in patch and "end" in patch:
            s, e = float(patch["start"]), float(patch["end"])
            if not any(n[0] < e and n[1] > s for n in out):
                out.append([s, e, pitch])
                applied += 1
                continue
        missed.append(t)
    return sorted(tuple(n) for n in out), applied, missed


def line_starts_from_aligned(aligned_path: str | Path) -> list[float]:
    aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    starts = []
    for line in aligned:
        if not line.get("tokens"):
            continue
        s = float(line.get("start", 0.0))
        e = float(line.get("end", s))
        if e > s:
            starts.append(s)
    return sorted(starts)


def assign_lines(notes: list[Note], line_starts: list[float]) -> list[int]:
    """Map each note to the lyric line it belongs to (index into line_starts).

    With no line info every note gets line 0, which disables the
    line-boundary preference in split_oversized."""
    if not line_starts:
        return [0] * len(notes)
    return [bisect.bisect_right(line_starts, s + LINE_ONSET_TOLERANCE) - 1
            for s, _e, _p in notes]


def split_oversized(
    ph: list[int],
    notes: list[Note],
    width,
    budget: float,
    line_of: list[int],
) -> list[list[int]]:
    """Recursively split an over-budget phrase cluster.

    Cut candidates at lyric-line boundaries take absolute precedence (a
    chorus cluster then falls apart into its sung lines); only a single
    oversized LINE falls back to its largest internal acoustic gap.
    Among candidates: largest real gap first, ties broken toward the middle.
    """
    if width(ph) <= budget or len(ph) < 2:
        return [ph]
    cuts = range(len(ph) - 1)
    line_cuts = [k for k in cuts if line_of[ph[k + 1]] != line_of[ph[k]]]
    pool = line_cuts if line_cuts else list(cuts)
    mid_idx = (len(ph) - 1) / 2
    gaps = [(notes[ph[k + 1]][0] - notes[ph[k]][1], k) for k in pool]
    _g, cut = max(gaps, key=lambda g: (round(g[0], 3), -abs(g[1] - mid_idx)))
    return (split_oversized(ph[: cut + 1], notes, width, budget, line_of)
            + split_oversized(ph[cut + 1:], notes, width, budget, line_of))


def phrase_width(
    ph: list[int],
    notes: list[Note],
    qnotes: list[Note],
    *,
    breath_space: float,
    breath_gap: float,
) -> float:
    """Display width of a phrase: the sum of its note slots (slot = quantized
    duration; the mora gap lives INSIDE the slot) plus one extra breath space
    after every note whose real silence to the next note exceeds breath_gap.

    Must mirror layout_pages' cursor arithmetic exactly or packing overflows.
    """
    w = sum(qnotes[i][1] - qnotes[i][0] for i in ph)
    w += breath_space * sum(
        1 for a, b in zip(ph, ph[1:]) if notes[b][0] - notes[a][1] > breath_gap)
    return w


def pack_pages(
    phrases: list[list[int]],
    width,
    *,
    span: float,
    lead: float,
    pgap: float,
    quarter: float,
) -> list[list[list[int]]]:
    """Greedy fixed-span packing: a phrase that does not fit moves whole to
    the next page; oversized phrases get a page of their own."""
    pages: list[list[list[int]]] = [[]]
    cursor = lead
    for ph in phrases:
        w = width(ph)
        extra = (pgap if pages[-1] else 0.0)
        if pages[-1] and cursor + extra + w > span - 0.25 * quarter:
            pages.append([])
            cursor = lead
            extra = 0.0
        pages[-1].append(ph)
        cursor += extra + w
    return pages


def layout_pages(
    pages: list[list[list[int]]],
    notes: list[Note],
    qnotes: list[Note],
    *,
    span: float,
    lead: float,
    gap: float,
    pgap: float,
    quarter: float,
    count_in_quarters: float = 4.0,
    flip_delay: float = 0.5,
    breath_space: float = 0.0,
    breath_gap: float = 0.25,
) -> tuple[list[Note], list[float], list[float]]:
    """Place notes on the display timeline and build the real<->display warp.

    Slot semantics (v11): each note occupies exactly its quantized duration;
    the bar is drawn slot - gap so widths stay perfectly proportional and the
    on-grid cursor speed is constant. A real silence > breath_gap inserts an
    extra breath_space between slots (visible breath marker; the cursor
    sweeps it during the actual breath).

    Long rests before a page get the TV treatment: an early page-flip anchor
    (flip_delay after the previous note ends), a park anchor (cursor waits at
    the page's left edge), then the first note's own anchor sweeps the lead
    during the final count_in_quarters — the moving cursor IS the count-in.

    Short inter-page gaps flip as soon as the previous note ends (quick-flip
    anchor at 25% of the gap): the next page's pitches must be readable for
    as much of the gap as possible, or the singer cannot sight-read the line.
    """
    disp_notes: list[Note] = []
    real_anchors: list[float] = [0.0]
    disp_anchors: list[float] = [0.0]

    def add_anchor(r: float, d: float) -> None:
        if r <= real_anchors[-1] + 1e-6:
            r = real_anchors[-1] + 1e-4
        if d <= disp_anchors[-1] + 1e-6:
            d = disp_anchors[-1] + 1e-4
        real_anchors.append(r)
        disp_anchors.append(d)

    for p, page in enumerate(pages):
        cursor = p * span + lead
        first_real = notes[page[0][0]][0]
        prev_real = real_anchors[-1]
        avail = first_real - prev_real
        count_real = first_real - count_in_quarters * quarter
        flip_real = prev_real + flip_delay
        if count_real - flip_real > 0.25:
            add_anchor(flip_real, p * span)        # flip page early
            add_anchor(count_real, p * span + 1e-3)  # park until the count-in
        elif avail > 0.12:
            # quick flip right after the previous note ends; the remaining
            # ~75% of the gap previews the new page before its first note
            add_anchor(prev_real + max(0.08, 0.25 * avail), p * span)
        for k, ph in enumerate(page):
            if k:
                cursor += pgap
            for j, i in enumerate(ph):
                qdur = qnotes[i][1] - qnotes[i][0]
                rs, re, pitch = notes[i]
                bar_end = cursor + qdur - gap
                disp_notes.append((cursor, bar_end, pitch))
                add_anchor(rs, cursor)
                add_anchor(re, bar_end)
                cursor += qdur
                if (j < len(ph) - 1
                        and notes[ph[j + 1]][0] - re > breath_gap):
                    cursor += breath_space

    # closing anchor so the warp stays defined to the end of the song
    add_anchor(real_anchors[-1] + 30.0, disp_anchors[-1] + 30.0)
    return disp_notes, real_anchors, disp_anchors


@click.command()
@click.option("--midi", "midi_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--bpm-file", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out-midi", type=click.Path(dir_okay=False), required=True)
@click.option("--out-warp", type=click.Path(dir_okay=False), required=True)
@click.option("--quarters-per-page", type=float, default=16.0, show_default=True,
              help="Fixed page span in quarters => fixed quarter width on screen.")
@click.option("--gap-units", type=float, default=0.0625, show_default=True,
              help="Fixed gap between morae, in quarters (Kojek: just visibly "
              "separate, no wider — 0.25 and 0.125 both read as too sparse).")
@click.option("--phrase-gap-units", type=float, default=1.25, show_default=True,
              help="Fixed (larger) gap between phrases, in quarters.")
@click.option("--lead-units", type=float, default=0.5, show_default=True)
@click.option("--phrase-split-gap", type=float, default=0.8, show_default=True,
              help="REAL-time silence that separates phrases.")
@click.option("--min-note", type=float, default=0.09, show_default=True,
              help="Drop sliver fragments shorter than this (real seconds).")
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Gate notes to lyric windows (+ tail allowance) so stray "
              "detections in instrumental gaps never reach the display; also "
              "provides line boundaries for oversized-cluster splitting.")
@click.option("--note-window-margin", type=float, default=0.25, show_default=True)
@click.option("--note-tail-allowance", type=float, default=1.0, show_default=True)
@click.option("--count-in-quarters", type=float, default=4.0, show_default=True,
              help="After a long rest the cursor parks at the page edge and "
              "starts moving this many quarters before the first note.")
@click.option("--flip-delay", type=float, default=0.5, show_default=True,
              help="Seconds after the last note of a page before the early "
              "page flip (long rests only).")
@click.option("--pitch-patch", "pitch_patch_path",
              type=click.Path(exists=True, dir_okay=False), default=None,
              help="JSON list of ear-verified pitch fixes, display layer only "
              "(never touches eval candidates). See apply_pitch_patch().")
@click.option("--breath-gap", type=float, default=0.25, show_default=True,
              help="REAL inter-note silence above this marks a breath.")
@click.option("--breath-units", type=float, default=0.5, show_default=True,
              help="Extra display space inserted at a breath, in quarters "
              "(visibly wider than the in-slot mora gap).")
@click.option("--rms-segments", "rms_segments_path",
              type=click.Path(exists=True, dir_okay=False), default=None,
              help="rms_segments.json; lyric note windows are additionally "
              "intersected with RMS voiced segments so separation-bleed notes "
              "inside instrumental breaks never render, even when a misaligned "
              "lyric window (e.g. a no-kana ad-lib line) spans the break.")
def main(midi_path, bpm_file, out_midi, out_warp, quarters_per_page,
         gap_units, phrase_gap_units, lead_units, phrase_split_gap, min_note,
         aligned_path, note_window_margin, note_tail_allowance,
         count_in_quarters, flip_delay, pitch_patch_path,
         breath_gap, breath_units, rms_segments_path):
    # MID2BAR's mid2csv converts the tempo meta back to BPM with round(_, 2)
    # before deriving tick->seconds. Round HERE too, or the warp's display
    # seconds drift ~3e-5 relative vs the renderer's — ~5ms by song middle,
    # enough to park the count-in cursor on the wrong side of a page marker
    # (the v9 "cursor parked at the RIGHT edge during interludes" bug).
    bpm_2dp = round(float(Path(bpm_file).read_text().strip()), 2)
    quarter = 60.0 / bpm_2dp
    notes = sorted((n.start, n.end, n.pitch) for n in read_midi_notes(Path(midi_path)))

    line_starts: list[float] = []
    if aligned_path:
        windows = _lyric_windows(aligned_path, margin=note_window_margin,
                                 tail_allowance=note_tail_allowance)
        if rms_segments_path:
            windows = _intersect_windows(windows, _voiced_windows(rms_segments_path))
        notes = [n for n in notes
                 if any(ws <= n[0] < we for ws, we in windows)
                 or any(n[0] < we and n[1] > ws for ws, we in windows)]
        line_starts = line_starts_from_aligned(aligned_path)

    notes, dropped = drop_fragments(notes, min_note=min_note)
    notes, wiggles = absorb_wiggles(notes)

    patched = 0
    if pitch_patch_path:
        patches = json.loads(Path(pitch_patch_path).read_text(encoding="utf-8"))
        notes, patched, missed = apply_pitch_patch(notes, patches)
        for t in missed:
            click.echo(f"[display-grid] WARNING: pitch patch at {t:.2f}s matched no note")

    qnotes = quantize(notes, quarter=quarter)  # display widths; real times kept in `notes`
    line_of = assign_lines(notes, line_starts)

    # phrases on REAL time
    phrases: list[list[int]] = [[0]]
    for i in range(1, len(notes)):
        if notes[i][0] - notes[i - 1][1] > phrase_split_gap:
            phrases.append([i])
        else:
            phrases[-1].append(i)

    # phrase display width = sum of note slots (+ breath spaces); the mora
    # gap lives inside each slot, so it does not appear here
    gap = gap_units * quarter
    pgap = phrase_gap_units * quarter
    lead = lead_units * quarter
    span = quarters_per_page * quarter
    breath_space = breath_units * quarter

    def width(ph: list[int]) -> float:
        return phrase_width(ph, notes, qnotes,
                            breath_space=breath_space, breath_gap=breath_gap)

    # chorus sections flow with sub-split gaps and become one huge cluster;
    # split oversized phrases (preferring lyric-line boundaries) until each
    # fits the page budget
    budget = span - lead - 0.25 * quarter
    phrases = [part for ph in phrases
               for part in split_oversized(ph, notes, width, budget, line_of)]

    pages = pack_pages(phrases, width, span=span, lead=lead, pgap=pgap,
                       quarter=quarter)

    disp_notes, real_anchors, disp_anchors = layout_pages(
        pages, notes, qnotes, span=span, lead=lead, gap=gap, pgap=pgap,
        quarter=quarter, count_in_quarters=count_in_quarters,
        flip_delay=flip_delay, breath_space=breath_space,
        breath_gap=breath_gap)

    # write display MIDI: notes + tempo + page markers every `span`
    bpm = 60.0 / quarter
    mid = mido.MidiFile(ticks_per_beat=480)
    tempo_us = mido.bpm2tempo(bpm)

    def sec2tick(t: float) -> int:
        # ticks straight from the 2dp-BPM quarter (NOT via the int-µs tempo):
        # the renderer recomputes seconds as tick/480 * 60/round(bpm, 2), so
        # this keeps grid seconds and renderer seconds bit-for-bit aligned
        return int(round(t / quarter * 480))

    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4,
                                  clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    events = []
    for s, e, pitch in disp_notes:
        events.append((sec2tick(s), 1, mido.Message("note_on", note=pitch, velocity=100, time=0)))
        events.append((sec2tick(e), 0, mido.Message("note_off", note=pitch, velocity=0, time=0)))
    events.sort(key=lambda x: (x[0], x[1]))
    prev = 0
    for tick, _prio, msg in events:
        track.append(msg.copy(time=max(tick - prev, 0)))
        prev = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(track)

    marker_track = mido.MidiTrack()
    prev = 0
    for p in range(len(pages) + 1):
        tick = sec2tick(p * span)
        marker_track.append(mido.MetaMessage(
            "marker", text=f"P{p + 1:02d}" if p < len(pages) else "END",
            time=max(tick - prev, 0)))
        prev = tick
    marker_track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(marker_track)

    Path(out_midi).parent.mkdir(parents=True, exist_ok=True)
    mid.save(out_midi)
    Path(out_warp).write_text(json.dumps(
        {"real": real_anchors, "display": disp_anchors}, indent=0))

    overlaps = sum(1 for a, b in zip(disp_notes, disp_notes[1:]) if b[0] < a[1] - 1e-6)
    click.echo(f"[display-grid] notes={len(disp_notes)} pages={len(pages)} "
               f"dropped_fragments={dropped} wiggles_absorbed={wiggles} "
               f"pitch_patched={patched} overlaps={overlaps} "
               f"page_span={span:.2f}s quarter={quarter:.3f}s")


if __name__ == "__main__":
    main()
