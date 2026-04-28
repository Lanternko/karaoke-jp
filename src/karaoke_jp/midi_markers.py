"""Inject MID2BAR-style page markers into a melody MIDI.

MID2BAR-Player's renderer divides the song into "pages" using
``MetaMessage('marker', text=...)`` events. Without markers the bar
display has no page break and the lyric layer cannot scroll. The shipped
``midi_marker_editor.py`` is a Tkinter GUI; we want a headless
``aligned.json -> markers`` step.

Strategy: one marker per lyrics line at the line's ``start`` second.
"""
from __future__ import annotations

import json
from pathlib import Path

import mido


def _seconds_to_ticks(seconds: float, ticks_per_beat: int, tempo_us: int) -> int:
    """Convert wall-clock seconds to MIDI ticks under a fixed tempo.

    ``tempo_us`` is microseconds per beat (the ``set_tempo`` value).
    """
    beats = seconds * 1_000_000 / tempo_us
    return int(round(beats * ticks_per_beat))


def inject_line_markers(
    midi_path: str | Path,
    aligned_path: str | Path,
    out_path: str | Path,
) -> int:
    """Add one ``marker`` meta-event per lyrics line at its start time.

    Returns the number of markers written.
    """
    midi_path = Path(midi_path)
    aligned_path = Path(aligned_path)
    out_path = Path(out_path)

    mid = mido.MidiFile(midi_path)
    aligned = json.loads(aligned_path.read_text(encoding="utf-8"))

    # Pull the song-wide tempo. SOME emits a single set_tempo near track 0
    # head; if we ever switch to a model that ramps tempo we'd need a
    # piecewise conversion here.
    tempo_us = 500_000  # default 120 bpm
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo_us = msg.tempo
            break

    # MID2BAR's mid2csv needs a time signature to build bars_data. SOME
    # doesn't emit one. Inject a default 4/4 at tick 0 if missing — this
    # only affects bar-counting display, not note timing.
    has_ts = any(
        msg.type == "time_signature"
        for tr in mid.tracks
        for msg in tr
    )
    if not has_ts:
        # Prepend to the first track's head. Existing first event's delta
        # stays correct because our injected event has time=0.
        ts_msg = mido.MetaMessage(
            "time_signature",
            numerator=4, denominator=4,
            clocks_per_click=24, notated_32nd_notes_per_beat=8,
            time=0,
        )
        mid.tracks[0].insert(0, ts_msg)

    line_starts: list[tuple[float, str]] = []
    for i, line in enumerate(aligned, start=1):
        if not line["tokens"]:
            continue
        line_starts.append((float(line["start"]), f"L{i:02d}"))
    line_starts.sort(key=lambda p: p[0])

    if not line_starts:
        # No lines to mark; just copy the file through.
        mid.save(out_path)
        return 0

    # Add a tail marker so the renderer has an end-of-page boundary
    # after the last lyrics line.
    last_t = line_starts[-1][0]
    final_line = aligned[-1]
    tail_t = max(float(final_line.get("end", last_t)), last_t) + 1.0
    line_starts.append((tail_t, "END"))

    # Build a marker-only track. Mido tracks use *delta* times.
    marker_track = mido.MidiTrack()
    prev_tick = 0
    for t_s, label in line_starts:
        abs_tick = _seconds_to_ticks(t_s, mid.ticks_per_beat, tempo_us)
        delta = max(abs_tick - prev_tick, 0)
        marker_track.append(mido.MetaMessage("marker", text=label, time=delta))
        prev_tick = abs_tick
    marker_track.append(mido.MetaMessage("end_of_track", time=0))

    mid.tracks.append(marker_track)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(out_path)
    return len(line_starts) - 1  # don't count the synthetic END
