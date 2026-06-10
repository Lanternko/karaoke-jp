#!/usr/bin/env python3
"""Patch selected MIDI time ranges with reference-guide notes.

This is a gold-driven sidecar helper: keep a full-song candidate melody outside
the named ranges, but replace the notes inside those ranges with a trusted
reference MIDI extracted from a score / official karaoke guide.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from karaoke_jp.melody import _write_midi  # noqa: E402
from karaoke_jp.score_melody import MidiNote, read_first_tempo_bpm, read_midi_notes  # noqa: E402


@dataclass(frozen=True)
class Segment:
    label: str
    start: float
    end: float


def _parse_segment(spec: str) -> Segment:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise click.BadParameter("segment must be LABEL:START:END, seconds")
    label, start, end = parts
    start_s = float(start)
    end_s = float(end)
    if end_s <= start_s:
        raise click.BadParameter(f"segment end must be greater than start: {spec}")
    return Segment(label=label, start=start_s, end=end_s)


def _trim_base_note(note: MidiNote, segments: list[Segment], min_duration: float) -> list[MidiNote]:
    fragments = [note]
    for segment in segments:
        next_fragments: list[MidiNote] = []
        for fragment in fragments:
            if fragment.end <= segment.start or fragment.start >= segment.end:
                next_fragments.append(fragment)
                continue
            if fragment.start < segment.start:
                next_fragments.append(
                    MidiNote(fragment.start, min(fragment.end, segment.start), fragment.pitch)
                )
            if fragment.end > segment.end:
                next_fragments.append(
                    MidiNote(max(fragment.start, segment.end), fragment.end, fragment.pitch)
                )
        fragments = next_fragments
    return [fragment for fragment in fragments if fragment.end - fragment.start >= min_duration]


def _reference_notes_in_segments(
    notes: list[MidiNote],
    segments: list[Segment],
    min_duration: float,
) -> list[MidiNote]:
    patched: list[MidiNote] = []
    for note in notes:
        for segment in segments:
            start = max(note.start, segment.start)
            end = min(note.end, segment.end)
            if end - start >= min_duration:
                patched.append(MidiNote(start, end, note.pitch))
    return patched


def _as_tuples(notes: list[MidiNote]) -> list[tuple[float, float, int]]:
    return [(note.start, note.end, note.pitch) for note in notes]


@click.command()
@click.option("--base-midi", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--reference-midi", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--segment", "segments", multiple=True, callback=lambda _ctx, _param, vals: [_parse_segment(v) for v in vals])
@click.option("--out-midi", type=click.Path(dir_okay=False), required=True)
@click.option("--report", type=click.Path(dir_okay=False), required=True)
@click.option("--tempo", type=float, default=None)
@click.option("--min-duration", type=float, default=0.03, show_default=True)
def main(
    base_midi: str,
    reference_midi: str,
    segments: list[Segment],
    out_midi: str,
    report: str,
    tempo: float | None,
    min_duration: float,
) -> None:
    if not segments:
        raise click.UsageError("At least one --segment LABEL:START:END is required.")
    if min_duration <= 0:
        raise click.BadParameter("--min-duration must be positive")

    base_notes = read_midi_notes(base_midi)
    reference_notes = read_midi_notes(reference_midi)

    kept_base: list[MidiNote] = []
    removed_base = 0
    for note in base_notes:
        fragments = _trim_base_note(note, segments, min_duration)
        if fragments != [note]:
            removed_base += 1
        kept_base.extend(fragments)

    patched_reference = _reference_notes_in_segments(reference_notes, segments, min_duration)
    output_notes = sorted(kept_base + patched_reference, key=lambda n: (n.start, n.end, n.pitch))

    tempo_bpm = tempo if tempo is not None else read_first_tempo_bpm(base_midi)
    out_path = Path(out_midi)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_midi(_as_tuples(output_notes), out_path, tempo=tempo_bpm)

    report_data = {
        "base_midi": base_midi,
        "reference_midi": reference_midi,
        "out_midi": out_midi,
        "segments": [asdict(segment) for segment in segments],
        "base_notes": len(base_notes),
        "base_notes_removed_or_trimmed": removed_base,
        "base_fragments_kept": len(kept_base),
        "reference_notes_inserted": len(patched_reference),
        "output_notes": len(output_notes),
        "tempo_bpm": tempo_bpm,
        "min_duration": min_duration,
    }
    report_path = Path(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    click.echo(
        f"[ref-patch] wrote {out_path} "
        f"(base={len(base_notes)}, inserted={len(patched_reference)}, output={len(output_notes)})"
    )
    click.echo(f"[ref-patch] wrote {report_path}")


if __name__ == "__main__":
    main()
