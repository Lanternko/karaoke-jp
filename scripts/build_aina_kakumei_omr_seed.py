#!/usr/bin/env python3
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from karaoke_jp.melody import _write_midi


TYPE_TO_BEATS = {
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
    pitch: str | None
    beats: float


def _qualified(tag: str, ns: dict[str, str]) -> str:
    return f"m:{tag}" if ns else tag


def parse_musicxml_notes(path: Path) -> list[ParsedNote]:
    root = ET.parse(path).getroot()
    ns = {"m": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    q = lambda tag: _qualified(tag, ns)

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
                pitch_name: str | None
                if rest:
                    pitch_name = None
                else:
                    pitch = note.find(q("pitch"), ns)
                    if pitch is None:
                        continue
                    step = pitch.findtext(q("step"), default="")
                    alter = pitch.findtext(q("alter"), default="")
                    octave = pitch.findtext(q("octave"), default="")
                    accidental = {"1": "#", "-1": "b"}.get(alter, "")
                    pitch_name = f"{step}{accidental}{octave}"
                notes.append(ParsedNote(measure=measure_no, pitch=pitch_name, beats=beats))
    return notes


def pitch_to_midi(pitch_name: str) -> int:
    step = pitch_name[0]
    octave = int(pitch_name[-1])
    accidental = pitch_name[1:-1]
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[step]
    accidental_offset = {"": 0, "#": 1, "b": -1}[accidental]
    return (octave + 1) * 12 + base + accidental_offset


def to_timed_notes(parsed: list[ParsedNote], tempo: float, *, drop_first: int = 0) -> list[tuple[float, float, int]]:
    sec_per_beat = 60.0 / tempo
    current_beats = 0.0
    timed: list[tuple[float, float, int]] = []
    pitched_count = 0
    for item in parsed:
        start_s = current_beats * sec_per_beat
        end_s = (current_beats + item.beats) * sec_per_beat
        current_beats += item.beats
        if item.pitch is None:
            continue
        if pitched_count < drop_first:
            pitched_count += 1
            continue
        pitched_count += 1
        timed.append((start_s, end_s, pitch_to_midi(item.pitch)))
    return timed


def write_tsv(path: Path, parsed: list[ParsedNote]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("measure\tpitch\tbeats\n")
        for item in parsed:
            fh.write(f"{item.measure}\t{item.pitch or 'rest'}\t{item.beats}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export current AiNA The End OMR seed fragments to MIDI.")
    parser.add_argument("--tempo", type=float, default=93.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("songs/aina-kakumei/omr_seed"),
    )
    args = parser.parse_args()

    segments = [
        ("p1_top", Path("p1_top.musicxml"), 1),
        ("p2_top", Path("tmp/aina-kakumei/omr/p2_top.musicxml"), 0),
    ]

    combined: list[tuple[float, float, int]] = []
    segment_gap_s = 4 * (60.0 / args.tempo)
    cursor_s = 0.0

    for name, musicxml_path, drop_first in segments:
        parsed = parse_musicxml_notes(musicxml_path)
        write_tsv(args.output_dir / f"{name}.tsv", parsed)
        timed = to_timed_notes(parsed, args.tempo, drop_first=drop_first)
        _write_midi(timed, args.output_dir / f"{name}.mid", tempo=args.tempo)
        for start_s, end_s, pitch in timed:
            combined.append((start_s + cursor_s, end_s + cursor_s, pitch))
        if timed:
            cursor_s = combined[-1][1] + segment_gap_s

    _write_midi(combined, args.output_dir / "combined.mid", tempo=args.tempo)
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
