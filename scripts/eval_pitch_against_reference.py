#!/usr/bin/env python3
"""Evaluate candidate melody MIDI files against a sparse reference MIDI.

The reference MIDI is intended to come from a karaoke-guide / score-like source
for selected key segments.  Metrics are computed on frames where the reference
has an active note, so a candidate can be penalized for missing notes, wrong
semitones, and octave-like errors separately.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from karaoke_jp.score_melody import MidiNote, read_midi_notes  # noqa: E402


@dataclass(frozen=True)
class Segment:
    label: str
    start: float
    end: float


@dataclass(frozen=True)
class Candidate:
    label: str
    path: Path


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


def _parse_candidate(spec: str) -> Candidate:
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise click.BadParameter("candidate must be LABEL:PATH")
    label, path = parts
    candidate_path = Path(path)
    if not candidate_path.exists():
        raise click.BadParameter(f"candidate path does not exist: {path}")
    return Candidate(label=label, path=candidate_path)


def _default_segments(reference: list[MidiNote]) -> list[Segment]:
    if not reference:
        raise click.UsageError("Reference MIDI has no notes; pass a non-empty reference.")
    return [
        Segment(
            label="reference_active_span",
            start=min(note.start for note in reference),
            end=max(note.end for note in reference),
        )
    ]


def _pitch_at(notes: list[MidiNote], time_s: float, *, offset: float = 0.0) -> int | None:
    shifted_time = time_s - offset
    for note in notes:
        if note.start <= shifted_time < note.end:
            return note.pitch
    return None


def _empty_counts() -> dict[str, int]:
    return {
        "total": 0,
        "miss": 0,
        "exact": 0,
        "within1": 0,
        "within2": 0,
        "octave": 0,
        "bad_non_octave_gt2": 0,
    }


def _rates(counts: dict[str, int]) -> dict[str, float | int]:
    total = counts["total"]
    result: dict[str, float | int] = dict(counts)
    for key in ("miss", "exact", "within1", "within2", "octave", "bad_non_octave_gt2"):
        result[f"{key}_rate"] = counts[key] / total if total else 0.0
    return result


def _evaluate_counts(
    *,
    reference: list[MidiNote],
    candidate: list[MidiNote],
    segments: list[Segment],
    fps: float,
    candidate_offset: float,
) -> tuple[dict[str, int], dict[str, dict[str, float | int]]]:
    totals = _empty_counts()
    per_segment: dict[str, dict[str, float | int]] = {}
    step = 1.0 / fps

    for segment in segments:
        counts = _empty_counts()
        times = np.arange(segment.start, segment.end, step, dtype=np.float64)
        for time_s in times:
            ref_pitch = _pitch_at(reference, float(time_s))
            if ref_pitch is None:
                continue
            counts["total"] += 1
            cand_pitch = _pitch_at(candidate, float(time_s), offset=candidate_offset)
            if cand_pitch is None:
                counts["miss"] += 1
                continue

            diff = cand_pitch - ref_pitch
            abs_diff = abs(diff)
            if abs_diff == 0:
                counts["exact"] += 1
            if abs_diff <= 1:
                counts["within1"] += 1
            if abs_diff <= 2:
                counts["within2"] += 1
            elif abs(abs_diff - 12) <= 1:
                counts["octave"] += 1
            else:
                counts["bad_non_octave_gt2"] += 1

        for key, value in counts.items():
            totals[key] += value
        per_segment[segment.label] = _rates(counts)

    return totals, per_segment


def _score(counts: dict[str, int], offset: float) -> tuple[float, float, float, float, float]:
    total = counts["total"] or 1
    return (
        counts["within2"] / total,
        -counts["miss"] / total,
        -counts["bad_non_octave_gt2"] / total,
        counts["exact"] / total,
        -abs(offset),
    )


def _offset_grid(min_offset: float, max_offset: float, step: float) -> list[float]:
    if step <= 0:
        raise click.BadParameter("--offset-step must be positive")
    values: list[float] = []
    current = min_offset
    while current <= max_offset + step * 0.5:
        values.append(round(current, 6))
        current += step
    return values


def _best_offset(
    *,
    reference: list[MidiNote],
    candidate: list[MidiNote],
    segments: list[Segment],
    fps: float,
    min_offset: float,
    max_offset: float,
    offset_step: float,
) -> tuple[float, dict[str, int], dict[str, dict[str, float | int]]]:
    best_offset = 0.0
    best_counts: dict[str, int] | None = None
    best_per_segment: dict[str, dict[str, float | int]] | None = None
    best_score: tuple[float, float, float, float, float] | None = None
    for offset in _offset_grid(min_offset, max_offset, offset_step):
        counts, per_segment = _evaluate_counts(
            reference=reference,
            candidate=candidate,
            segments=segments,
            fps=fps,
            candidate_offset=offset,
        )
        score = _score(counts, offset)
        if best_score is None or score > best_score:
            best_score = score
            best_offset = offset
            best_counts = counts
            best_per_segment = per_segment
    assert best_counts is not None
    assert best_per_segment is not None
    return best_offset, best_counts, best_per_segment


def _summary_row(candidate: str, mode: str, offset: float, metrics: dict[str, float | int]) -> dict[str, object]:
    return {
        "candidate": candidate,
        "mode": mode,
        "candidate_offset": f"{offset:.3f}",
        "total": metrics["total"],
        "miss_rate": f"{metrics['miss_rate']:.4f}",
        "exact_rate": f"{metrics['exact_rate']:.4f}",
        "within1_rate": f"{metrics['within1_rate']:.4f}",
        "within2_rate": f"{metrics['within2_rate']:.4f}",
        "octave_rate": f"{metrics['octave_rate']:.4f}",
        "bad_non_octave_gt2_rate": f"{metrics['bad_non_octave_gt2_rate']:.4f}",
    }


@click.command()
@click.option("--reference-midi", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--candidate", "candidates", multiple=True, callback=lambda _ctx, _param, vals: [_parse_candidate(v) for v in vals])
@click.option("--segment", "segments", multiple=True, callback=lambda _ctx, _param, vals: [_parse_segment(v) for v in vals])
@click.option("--out-json", type=click.Path(dir_okay=False), required=True)
@click.option("--out-tsv", type=click.Path(dir_okay=False), required=True)
@click.option("--fps", type=float, default=60.0, show_default=True)
@click.option("--min-offset", type=float, default=-0.8, show_default=True)
@click.option("--max-offset", type=float, default=0.8, show_default=True)
@click.option("--offset-step", type=float, default=0.02, show_default=True)
def main(
    reference_midi: str,
    candidates: list[Candidate],
    segments: list[Segment],
    out_json: str,
    out_tsv: str,
    fps: float,
    min_offset: float,
    max_offset: float,
    offset_step: float,
) -> None:
    if not candidates:
        raise click.UsageError("At least one --candidate LABEL:PATH is required.")
    if fps <= 0:
        raise click.BadParameter("--fps must be positive")

    reference_notes = read_midi_notes(reference_midi)
    if not segments:
        segments = _default_segments(reference_notes)

    results: dict[str, object] = {
        "_meta": {
            "reference_midi": reference_midi,
            "fps": fps,
            "min_offset": min_offset,
            "max_offset": max_offset,
            "offset_step": offset_step,
            "segments": [segment.__dict__ for segment in segments],
        },
        "candidates": {},
    }
    rows: list[dict[str, object]] = []

    for candidate in candidates:
        candidate_notes = read_midi_notes(candidate.path)
        zero_counts, zero_per_segment = _evaluate_counts(
            reference=reference_notes,
            candidate=candidate_notes,
            segments=segments,
            fps=fps,
            candidate_offset=0.0,
        )
        best_offset, best_counts, best_per_segment = _best_offset(
            reference=reference_notes,
            candidate=candidate_notes,
            segments=segments,
            fps=fps,
            min_offset=min_offset,
            max_offset=max_offset,
            offset_step=offset_step,
        )
        zero_metrics = _rates(zero_counts)
        best_metrics = _rates(best_counts)
        results["candidates"][candidate.label] = {
            "path": str(candidate.path),
            "zero_offset": {
                "candidate_offset": 0.0,
                "overall": zero_metrics,
                "segments": zero_per_segment,
            },
            "best_offset": {
                "candidate_offset": best_offset,
                "overall": best_metrics,
                "segments": best_per_segment,
            },
        }
        rows.append(_summary_row(candidate.label, "zero", 0.0, zero_metrics))
        rows.append(_summary_row(candidate.label, "best", best_offset, best_metrics))

    out_json_path = Path(out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_tsv_path = Path(out_tsv)
    out_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    click.echo(f"[ref-pitch-eval] wrote {out_json_path}")
    click.echo(f"[ref-pitch-eval] wrote {out_tsv_path}")


if __name__ == "__main__":
    main()
