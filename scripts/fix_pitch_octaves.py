#!/usr/bin/env python3
"""Canonical octave-error fix for an extracted melody MIDI.

Late-fusion consensus repair: shift only the notes RMVPE flags as a full octave
off, optionally vetoed by a second estimator (pYIN) on high-risk corrections,
then merge adjacent same-pitch fragments.

The DEFAULT is *unguarded* (no long-char span guard).  Ablation on
haru-hikage / tuki-zero / chidori (2026-06-06) showed the span guard buys no
``long_char_pitch_span>=2`` benefit (identical with or without it) yet leaves
octave-error clusters half-corrected — the uncorrected residuals float a full
octave above the F0 contour, i.e. visible octave zigzags (worst on chidori:
13 residual stable-octave notes guarded vs 1 unguarded).  Pass ``--span-guard``
(with ``--aligned``) only if you specifically need to protect a sustained char
from being split.

Usage:
    python scripts/fix_pitch_octaves.py outputs/<song>/melody_markers.mid \
        -o outputs/<song>/melody_markers.octavefix.mid \
        --rmvpe-f0 <rmvpe.npz> --pyin-f0 <pyin.npz>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import mido

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from karaoke_jp.melody import _seconds_to_ticks, _write_midi  # noqa: E402
from karaoke_jp.pitch_eval import (  # noqa: E402
    F0Track,
    lyric_char_windows,
    merge_adjacent_same_pitch_notes,
    shift_octave_notes_by_f0_consensus,
)
from karaoke_jp.score_melody import MidiNote, read_first_tempo_bpm, read_midi_notes  # noqa: E402


def _rebuild_note_track(track: mido.MidiTrack, notes: list[MidiNote], tpb: int, tempo_us: int) -> mido.MidiTrack:
    """Rewrite a track's note events from ``notes`` while keeping its other
    messages (track name, time_signature, …) at their absolute positions."""
    events: list[tuple[int, int, mido.Message]] = []  # (abs_tick, prio, msg)
    abs_tick = 0
    for msg in track:
        abs_tick += msg.time
        if msg.type in ("note_on", "note_off") or msg.type == "end_of_track":
            continue
        events.append((abs_tick, 0, msg))  # meta sorts before notes at the same tick
    for note in notes:
        on = _seconds_to_ticks(note.start, tpb, tempo_us)
        off = max(on + 1, _seconds_to_ticks(note.end, tpb, tempo_us))
        events.append((on, 2, mido.Message("note_on", note=int(note.pitch), velocity=100, time=0)))
        events.append((off, 1, mido.Message("note_off", note=int(note.pitch), velocity=0, time=0)))
    events.sort(key=lambda ev: (ev[0], ev[1]))

    rebuilt = mido.MidiTrack()
    prev = 0
    for abs_t, _prio, msg in events:
        rebuilt.append(msg.copy(time=max(abs_t - prev, 0)))
        prev = abs_t
    rebuilt.append(mido.MetaMessage("end_of_track", time=0))
    return rebuilt


def _write_notes_preserving_structure(src_path: str, out: Path, notes: list[MidiNote], *, tempo_bpm: float) -> None:
    """Write ``notes`` into a copy of ``src_path``, rewriting only the note
    track so page-marker / tempo tracks (a separate track in melody_markers.mid)
    survive intact.  Falls back to a fresh 2-track file when the source has no
    notes (e.g. a plain melody MIDI)."""
    src = mido.MidiFile(src_path)
    note_idx = next(
        (i for i, tr in enumerate(src.tracks)
         if any(m.type == "note_on" and m.velocity > 0 for m in tr)),
        None,
    )
    if note_idx is None:
        _write_midi([(n.start, n.end, n.pitch) for n in notes], out, tempo=tempo_bpm,
                    ticks_per_beat=src.ticks_per_beat)
        return
    tempo_us = mido.bpm2tempo(tempo_bpm)
    fixed = mido.MidiFile(ticks_per_beat=src.ticks_per_beat)
    for i, track in enumerate(src.tracks):
        if i == note_idx:
            fixed.tracks.append(_rebuild_note_track(track, notes, src.ticks_per_beat, tempo_us))
        else:
            fixed.tracks.append(track)  # verbatim: tempo/meta + page-marker tracks
    out.parent.mkdir(parents=True, exist_ok=True)
    fixed.save(out)


@click.command()
@click.argument("midi_in", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--rmvpe-f0", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Primary estimator F0 npz (decides which notes are an octave off).")
@click.option("--pyin-f0", type=click.Path(exists=True, dir_okay=False),
              help="Second estimator F0 npz; vetoes a shift when it disagrees (late fusion).")
@click.option("--span-guard", is_flag=True, default=False,
              help="Re-enable the long-char span guard (off by default; needs --aligned).")
@click.option("--aligned", type=click.Path(exists=True, dir_okay=False),
              help="aligned sidecar json; only used with --span-guard.")
@click.option("--no-merge", is_flag=True, default=False,
              help="Skip merging adjacent same-pitch fragments.")
@click.option("--tempo", type=float, default=None, help="Override output MIDI tempo (bpm).")
def main(
    midi_in: str,
    out_path: str,
    rmvpe_f0: str,
    pyin_f0: str | None,
    span_guard: bool,
    aligned: str | None,
    no_merge: bool,
    tempo: float | None,
) -> None:
    notes = read_midi_notes(midi_in)
    rmvpe = F0Track.from_npz(rmvpe_f0)
    pyin = F0Track.from_npz(pyin_f0) if pyin_f0 else None

    guard_windows = None
    if span_guard:
        if not aligned:
            raise click.UsageError("--span-guard requires --aligned (lyric char windows).")
        guard_windows = lyric_char_windows(json.loads(Path(aligned).read_text(encoding="utf-8")))

    shifted, n_changes = shift_octave_notes_by_f0_consensus(
        notes, primary=rmvpe, veto=pyin, span_guard_windows=guard_windows
    )
    if not no_merge:
        shifted = merge_adjacent_same_pitch_notes(shifted)

    tempo_bpm = tempo if tempo is not None else read_first_tempo_bpm(midi_in)
    out = Path(out_path)
    _write_notes_preserving_structure(midi_in, out, shifted, tempo_bpm=tempo_bpm)

    mode = "guarded" if span_guard else "unguarded"
    if no_merge:
        mode += ", no-merge"
    click.echo(
        f"[fix-octaves] {len(notes)} -> {len(shifted)} notes, "
        f"{n_changes} octave shift(s) ({mode}) -> {out}"
    )


if __name__ == "__main__":
    main()
