#!/usr/bin/env python3
"""Run pitch-contour ablations for one karaoke song.

This runner separates:
- rendered/current MIDI quality,
- RMVPE segmentation and octave fixers,
- pYIN as a second-estimator failure mode,
- late-fusion consensus-veto octave repair.
"""
from __future__ import annotations

import csv
import json
import sys
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from karaoke_jp.melody import (  # noqa: E402
    _write_midi,
    fill_held_note_gaps,
    fix_octave_errors,
    fix_phrase_octave,
    segment_f0_to_notes,
)
from karaoke_jp.pitch_eval import (  # noqa: E402
    F0Track,
    compare_notes_to_f0,
    fragmentation_metrics,
    lyric_char_windows,
    lyric_line_windows,
    merge_adjacent_same_pitch_notes,
    metrics_to_dict,
    shift_octave_notes_by_f0_consensus,
    stable_char_windows,
    transition_char_windows,
)
from karaoke_jp.score_melody import MidiNote, read_first_tempo_bpm, read_midi_notes  # noqa: E402


def _load_aligned(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _track_to_segmentable_f0(track: F0Track) -> tuple[np.ndarray, float]:
    f0 = np.nan_to_num(track.f0_hz, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    hop = track.hop_seconds
    if hop <= 0:
        raise ValueError("F0 track needs at least two frames to infer hop_seconds")
    return f0, hop


def _tuple_notes_to_midi_notes(notes: list[tuple[float, float, int]]) -> list[MidiNote]:
    return [MidiNote(float(s), float(e), int(p)) for s, e, p in notes]


def _midi_notes_to_tuples(notes: list[MidiNote]) -> list[tuple[float, float, int]]:
    return [(note.start, note.end, note.pitch) for note in notes]


def _rmvpe_notes(track: F0Track, postprocess: Callable[[list[tuple[float, float, int]]], list[tuple[float, float, int]]]) -> list[MidiNote]:
    f0, hop = _track_to_segmentable_f0(track)
    notes = segment_f0_to_notes(f0, hop_seconds=hop)
    return _tuple_notes_to_midi_notes(postprocess(notes))


def _pyin_raw_notes(track: F0Track) -> list[MidiNote]:
    f0, hop = _track_to_segmentable_f0(track)
    return _tuple_notes_to_midi_notes(segment_f0_to_notes(f0, hop_seconds=hop))


def _evaluate_variant(
    notes: list[MidiNote],
    *,
    rmvpe: F0Track,
    pyin: F0Track | None,
    line_windows,
    char_windows,
    stable_windows,
    transition_windows,
) -> dict:
    f0_metrics = {
        "rmvpe": {
            "all": metrics_to_dict(compare_notes_to_f0(notes, rmvpe)),
            "stable": metrics_to_dict(compare_notes_to_f0(notes, rmvpe, windows=stable_windows)),
            "transition": metrics_to_dict(compare_notes_to_f0(notes, rmvpe, windows=transition_windows)),
        }
    }
    if pyin is not None:
        f0_metrics["pyin"] = {
            "all": metrics_to_dict(compare_notes_to_f0(notes, pyin)),
            "stable": metrics_to_dict(compare_notes_to_f0(notes, pyin, windows=stable_windows)),
            "transition": metrics_to_dict(compare_notes_to_f0(notes, pyin, windows=transition_windows)),
        }
    return {
        "fragmentation": metrics_to_dict(
            fragmentation_metrics(notes, line_windows=line_windows, char_windows=char_windows)
        ),
        "f0": f0_metrics,
    }


def _summary_row(song: str, variant: str, result: dict) -> dict[str, object]:
    frag = result["fragmentation"]
    rmvpe = result["f0"]["rmvpe"]["stable"]
    pyin = result["f0"].get("pyin", {}).get("stable", {})
    return {
        "song": song,
        "variant": variant,
        "notes": frag["notes"],
        "outside_lyric_window": frag["outside_lyric_window"],
        "same_pitch_tiny_gap": frag["same_pitch_tiny_gap"],
        "rapid_aba_jitter": frag["rapid_aba_jitter"],
        "long_char_multi_note": frag["long_char_multi_note"],
        "long_char_pitch_span_ge_2": frag["long_char_pitch_span_ge_2"],
        "rmvpe_stable_rpa": round(rmvpe["rpa"], 4),
        "rmvpe_stable_rca": round(rmvpe["rca"], 4),
        "rmvpe_stable_octave_proxy": round(rmvpe["octave_proxy"], 4),
        "rmvpe_stable_note_octave": rmvpe["note_octave"],
        "rmvpe_stable_note_1semi": rmvpe["note_1_semitone"],
        "rmvpe_stable_note_2semi": rmvpe["note_2_semitone"],
        "rmvpe_stable_note_gt250": rmvpe["note_gt_250c"],
        "pyin_stable_octave_proxy": round(pyin.get("octave_proxy", 0.0), 4),
        "pyin_stable_note_octave": pyin.get("note_octave", ""),
    }


@click.command()
@click.option("--song", required=True, help="Song id used only in reports.")
@click.option("--current-midi", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--rmvpe-f0", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--pyin-f0", type=click.Path(exists=True, dir_okay=False))
@click.option("--out-dir", type=click.Path(file_okay=False), required=True)
@click.option("--tempo", type=float, default=None, help="Override MIDI tempo for generated variants.")
def main(
    song: str,
    current_midi: str,
    aligned_path: str,
    rmvpe_f0: str,
    pyin_f0: str | None,
    out_dir: str,
    tempo: float | None,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    aligned = _load_aligned(Path(aligned_path))
    line_windows = lyric_line_windows(aligned)
    char_windows = lyric_char_windows(aligned)
    stable_windows = stable_char_windows(char_windows)
    transition_windows = transition_char_windows(char_windows)

    rmvpe = F0Track.from_npz(rmvpe_f0)
    pyin = F0Track.from_npz(pyin_f0) if pyin_f0 else None
    tempo_bpm = tempo if tempo is not None else read_first_tempo_bpm(current_midi)

    current_notes = read_midi_notes(current_midi)
    consensus_notes, consensus_changes = shift_octave_notes_by_f0_consensus(
        current_notes,
        primary=rmvpe,
        veto=pyin,
    )
    consensus_merge_notes = merge_adjacent_same_pitch_notes(consensus_notes)
    guarded_notes, guarded_changes = shift_octave_notes_by_f0_consensus(
        current_notes,
        primary=rmvpe,
        veto=pyin,
        span_guard_windows=char_windows,
    )
    guarded_merge_notes = merge_adjacent_same_pitch_notes(guarded_notes)

    variants: dict[str, list[MidiNote]] = {
        "current_midi": current_notes,
        "current_consensus_octave": consensus_notes,
        "current_consensus_octave_merge": consensus_merge_notes,
        "current_consensus_octave_guarded": guarded_notes,
        "current_consensus_octave_guarded_merge": guarded_merge_notes,
        "rmvpe_raw": _rmvpe_notes(rmvpe, lambda notes: notes),
        "rmvpe_no_phrase_octave": _rmvpe_notes(
            rmvpe,
            lambda notes: fill_held_note_gaps(fix_octave_errors(notes)),
        ),
        "rmvpe_full_algo": _rmvpe_notes(
            rmvpe,
            lambda notes: fill_held_note_gaps(fix_octave_errors(fix_phrase_octave(notes))),
        ),
    }
    if pyin is not None:
        variants["pyin_raw"] = _pyin_raw_notes(pyin)

    results: dict[str, object] = {
        "_meta": {
            "song": song,
            "current_midi": current_midi,
            "aligned": aligned_path,
            "rmvpe_f0": rmvpe_f0,
            "pyin_f0": pyin_f0,
            "line_windows": len(line_windows),
            "char_windows": len(char_windows),
            "stable_windows": len(stable_windows),
            "transition_windows": len(transition_windows),
            "consensus_octave_changes": consensus_changes,
            "consensus_octave_guarded_changes": guarded_changes,
        }
    }

    rows = []
    for name, notes in variants.items():
        if name != "current_midi":
            _write_midi(_midi_notes_to_tuples(notes), out / f"{name}.mid", tempo=tempo_bpm)
        result = _evaluate_variant(
            notes,
            rmvpe=rmvpe,
            pyin=pyin,
            line_windows=line_windows,
            char_windows=char_windows,
            stable_windows=stable_windows,
            transition_windows=transition_windows,
        )
        results[name] = result
        rows.append(_summary_row(song, name, result))

    (out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(rows[0].keys()) if rows else []
    with (out / "summary.tsv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[pitch-ablation] wrote {out / 'results.json'}")
    print(f"[pitch-ablation] wrote {out / 'summary.tsv'}")


if __name__ == "__main__":
    main()
