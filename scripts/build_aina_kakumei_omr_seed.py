#!/usr/bin/env python3
"""Build score.mid (and legacy omr_seed outputs) for aina-kakumei.

Sources per segment, in priority order:
  1. Checked-in MusicXML (OMR output)
  2. Hand-transcribed TSV in tmp/aina-kakumei/tsv/<name>.tsv

Each segment carries an absolute start_measure so notes are placed at the
correct global beat position rather than concatenated with artificial gaps.
Formula:
    start_s = ((start_measure - 1) * 4 + beat_within_segment) * (60 / TEMPO)
"""
from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from karaoke_jp.melody import _write_midi

TEMPO = 93.0
SEC_PER_BEAT = 60.0 / TEMPO
BEATS_PER_MEASURE = 4  # 4/4 throughout

TYPE_TO_BEATS: dict[str, float] = {
    "whole": 4.0,
    "half": 2.0,
    "quarter": 1.0,
    "eighth": 0.5,
    "16th": 0.25,
    "32nd": 0.125,
}


@dataclass(frozen=True)
class ParsedNote:
    measure: str
    pitch: str | None  # None = rest
    beats: float


@dataclass
class SegmentDef:
    name: str
    source: Path
    start_measure: int  # global (1-based) measure where this segment begins
    drop_first: int = 0  # skip first N pitched notes (for OMR octave-error correction)


# ---------------------------------------------------------------------------
# Segment registry – add/remove rows here as OMR/manual TSVs become available.
# Missing files are skipped with a warning (partial score is still useful).
# ---------------------------------------------------------------------------
SEGMENTS: list[SegmentDef] = [
    # ── Page 1 ──────────────────────────────────────────────────────────────
    SegmentDef("p1_top",  Path("p1_top.musicxml"),                                   1,  drop_first=1),
    SegmentDef("p1_sys3", Path("tmp/aina-kakumei/tsv/p1_sys3.tsv"),                  7),
    SegmentDef("p1_sys4", Path("tmp/aina-kakumei/tsv/p1_sys4.tsv"),                 10),
    SegmentDef("p1_sys5", Path("tmp/aina-kakumei/tsv/p1_sys5.tsv"),                 13),
    SegmentDef("p1_sys6", Path("tmp/aina-kakumei/tsv/p1_sys6.tsv"),                 16),
    SegmentDef("p1_sys7", Path("tmp/aina-kakumei/tsv/p1_sys7.tsv"),                 19),
    SegmentDef("p1_sys8", Path("tmp/aina-kakumei/tsv/p1_sys8.tsv"),                 22),
    SegmentDef("p1_sys9", Path("tmp/aina-kakumei/tsv/p1_sys9.tsv"),                 25),
    # ── Page 2 ──────────────────────────────────────────────────────────────
    SegmentDef("p2_sys1", Path("tmp/aina-kakumei/tsv/p2_sys1.tsv"),                 28),
    SegmentDef("p2_top",  Path("tmp/aina-kakumei/omr/p2_top.musicxml"),             31),
    # p2_sys3 image covers m34-36; m34 is already in p2_top → start at m35
    SegmentDef("p2_sys3", Path("tmp/aina-kakumei/tsv/p2_sys3.tsv"),                 35),
    SegmentDef("p2_sys4", Path("tmp/aina-kakumei/tsv/p2_sys4.tsv"),                 37),
    SegmentDef("p2_sys5", Path("tmp/aina-kakumei/tsv/p2_sys5.tsv"),                 40),
    SegmentDef("p2_sys6", Path("tmp/aina-kakumei/tsv/p2_sys6.tsv"),                 43),
    SegmentDef("p2_sys7", Path("tmp/aina-kakumei/tsv/p2_sys7.tsv"),                 46),
    SegmentDef("p2_sys8", Path("tmp/aina-kakumei/tsv/p2_sys8.tsv"),                 49),
    SegmentDef("p2_sys9", Path("tmp/aina-kakumei/tsv/p2_sys9.tsv"),                 52),
]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _qualified(tag: str, ns: dict[str, str]) -> str:
    return f"m:{tag}" if ns else tag


def parse_musicxml_notes(path: Path) -> list[ParsedNote]:
    root = ET.parse(path).getroot()
    ns = {"m": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    q = lambda tag: _qualified(tag, ns)  # noqa: E731

    notes: list[ParsedNote] = []
    for part in root.findall(q("part"), ns):
        for measure in part.findall(q("measure"), ns):
            measure_no = measure.attrib.get("number", "?")
            for note in measure.findall(q("note"), ns):
                rest = note.find(q("rest"), ns) is not None
                note_type = note.findtext(q("type"), default="quarter")
                beats = TYPE_TO_BEATS.get(note_type, 1.0)
                if note.find(q("dot"), ns) is not None:
                    beats *= 1.5
                if rest:
                    notes.append(ParsedNote(measure=measure_no, pitch=None, beats=beats))
                    continue
                pitch_el = note.find(q("pitch"), ns)
                if pitch_el is None:
                    continue
                step   = pitch_el.findtext(q("step"),   default="")
                alter  = pitch_el.findtext(q("alter"),  default="")
                octave = pitch_el.findtext(q("octave"), default="")
                accidental = {"1": "#", "-1": "b"}.get(alter, "")
                notes.append(ParsedNote(
                    measure=measure_no,
                    pitch=f"{step}{accidental}{octave}",
                    beats=beats,
                ))
    return notes


def parse_tsv_notes(path: Path) -> list[ParsedNote]:
    """Read a hand-transcribed TSV with columns: measure  pitch  beats.

    'pitch' may be a note name (e.g. A5, F#4, Bb3) or the literal 'rest'.
    The 'measure' column is the local measure number within the segment.
    """
    notes: list[ParsedNote] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            pitch_val = row["pitch"].strip()
            notes.append(ParsedNote(
                measure=row["measure"].strip(),
                pitch=None if pitch_val.lower() == "rest" else pitch_val,
                beats=float(row["beats"].strip()),
            ))
    return notes


def pitch_to_midi(pitch_name: str) -> int:
    step = pitch_name[0]
    octave = int(pitch_name[-1])
    accidental = pitch_name[1:-1]
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step]
    accidental_offset = {"": 0, "#": 1, "b": -1}[accidental]
    return (octave + 1) * 12 + base + accidental_offset


# ---------------------------------------------------------------------------
# Timing: segment-local beats → absolute seconds
# ---------------------------------------------------------------------------

def to_absolute_timed_notes(
    parsed: list[ParsedNote],
    start_measure: int,
    *,
    drop_first: int = 0,
    sec_per_beat: float = SEC_PER_BEAT,
) -> list[tuple[float, float, int]]:
    """Convert a segment's note list to (abs_start_s, abs_end_s, midi_pitch).

    start_measure: global measure number (1-based) where this segment begins.
    """
    base_beat = (start_measure - 1) * BEATS_PER_MEASURE
    current_beat = 0.0
    timed: list[tuple[float, float, int]] = []
    pitched_count = 0

    for item in parsed:
        abs_start = (base_beat + current_beat) * sec_per_beat
        abs_end   = (base_beat + current_beat + item.beats) * sec_per_beat
        current_beat += item.beats

        if item.pitch is None:
            continue
        if pitched_count < drop_first:
            pitched_count += 1
            continue
        pitched_count += 1
        timed.append((abs_start, abs_end, pitch_to_midi(item.pitch)))

    return timed


# ---------------------------------------------------------------------------
# Legacy per-segment seed outputs (TSV + individual MIDI)
# ---------------------------------------------------------------------------

def write_seed_tsv(path: Path, parsed: list[ParsedNote]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("measure\tpitch\tbeats\n")
        for item in parsed:
            fh.write(f"{item.measure}\t{item.pitch or 'rest'}\t{item.beats}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build aina-kakumei score MIDI from OMR/hand-transcribed segments."
    )
    parser.add_argument("--tempo", type=float, default=TEMPO,
                        help="BPM (default: 93)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("songs/aina-kakumei/omr_seed"),
                        help="Directory for per-segment seed outputs (TSV + MIDI)")
    parser.add_argument("--output-path", type=Path,
                        default=Path("songs/aina-kakumei/score.mid"),
                        help="Destination path for the final merged score MIDI")
    parser.add_argument("--seed-only", action="store_true",
                        help="Write per-segment seed files but skip final score.mid")
    args = parser.parse_args()

    tempo: float = args.tempo
    sec_per_beat: float = 60.0 / tempo

    all_notes: list[tuple[float, float, int]] = []
    loaded: list[str] = []
    skipped: list[str] = []

    for seg in SEGMENTS:
        if not seg.source.exists():
            skipped.append(seg.name)
            continue

        if seg.source.suffix.lower() == ".musicxml":
            parsed = parse_musicxml_notes(seg.source)
        elif seg.source.suffix.lower() == ".tsv":
            parsed = parse_tsv_notes(seg.source)
        else:
            print(f"[WARN] unknown source type for {seg.name}: {seg.source}")
            skipped.append(seg.name)
            continue

        # Legacy seed outputs
        if not args.seed_only:
            write_seed_tsv(args.output_dir / f"{seg.name}.tsv", parsed)
        timed = to_absolute_timed_notes(
            parsed, seg.start_measure,
            drop_first=seg.drop_first,
            sec_per_beat=sec_per_beat,
        )
        if not args.seed_only:
            _write_midi(timed, args.output_dir / f"{seg.name}.mid", tempo=tempo)

        all_notes.extend(timed)
        loaded.append(seg.name)

    if skipped:
        print(f"[INFO] Skipped (missing): {', '.join(skipped)}")
    print(f"[INFO] Loaded:  {', '.join(loaded)}")

    if not all_notes:
        print("[ERROR] No notes collected – score.mid not written.")
        return

    # Sort by start time and remove exact-duplicate events
    all_notes.sort(key=lambda n: (n[0], n[2]))
    deduped: list[tuple[float, float, int]] = []
    seen: set[tuple[float, int]] = set()
    for start_s, end_s, pitch in all_notes:
        key = (round(start_s, 6), pitch)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((start_s, end_s, pitch))

    if not args.seed_only:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_midi(deduped, args.output_path, tempo=tempo)
        pitches = [p for _, _, p in deduped]
        print(f"[OK]  score.mid  notes={len(deduped)}  range=({min(pitches)}, {max(pitches)})")
        print(f"      → {args.output_path.resolve()}")


if __name__ == "__main__":
    main()
