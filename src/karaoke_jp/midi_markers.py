"""Inject MID2BAR-style page markers into a melody MIDI.

MID2BAR-Player's renderer divides the bar display into "pages" using
``MetaMessage('marker', text=...)`` events. Our LRC export groups lyrics into
multi-line blocks, so page markers need to follow the same grouping.

We also clamp each next-page marker so it never lands before the previous page
has actually finished singing. That avoids PREVIEW_TIME cutting off a page's
final sustained note.
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


def _midi_duration_seconds(mid: mido.MidiFile, tempo_us: int) -> float:
    """Return the wall-clock end time of the last note in *mid* (seconds)."""
    last = 0.0
    for track in mid.tracks:
        abs_ticks = 0
        for msg in track:
            abs_ticks += msg.time
            if msg.type in ("note_on", "note_off"):
                t = abs_ticks * tempo_us / (mid.ticks_per_beat * 1_000_000)
                if t > last:
                    last = t
    return last


def inject_line_markers(
    midi_path: str | Path,
    aligned_path: str | Path,
    out_path: str | Path,
    *,
    block_size: int = 2,
    dummy_interval: float = 4.0,
) -> int:
    """Add one ``marker`` meta-event per lyric block at its start time.

    Returns the number of markers written.
    """
    if block_size not in (1, 2, 3, 4):
        raise ValueError(f"block_size must be 1..4, got {block_size}")

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

    lines = [line for line in aligned if line["tokens"]]
    if not lines:
        # No lyrics — inject dummy markers every dummy_interval seconds so
        # MID2BAR's page-advance logic still fires and pitch bars are visible.
        duration = _midi_duration_seconds(mid, tempo_us)
        page_starts: list[tuple[float, str]] = []
        t = 0.0
        i = 1
        while t < duration:
            page_starts.append((t, f"P{i:02d}"))
            t += dummy_interval
            i += 1
        page_starts.append((duration + 1.0, "END"))

        marker_track = mido.MidiTrack()
        prev_tick = 0
        for t_s, label in page_starts:
            abs_tick = _seconds_to_ticks(t_s, mid.ticks_per_beat, tempo_us)
            delta = max(abs_tick - prev_tick, 0)
            marker_track.append(mido.MetaMessage("marker", text=label, time=delta))
            prev_tick = abs_tick
        marker_track.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(marker_track)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mid.save(out_path)
        return len(page_starts) - 1

    blocks = [lines[i:i + block_size] for i in range(0, len(lines), block_size)]
    page_starts: list[tuple[float, str]] = []
    prev_page_end = 0.0
    for i, block in enumerate(blocks, start=1):
        raw_start = float(block[0]["start"])
        page_end = max(float(line.get("end", raw_start)) for line in block)
        start_time = max(raw_start, prev_page_end)
        page_starts.append((start_time, f"P{i:02d}"))
        prev_page_end = max(prev_page_end, page_end, start_time)

    tail_t = prev_page_end + 1.0
    page_starts.append((tail_t, "END"))

    # Build a marker-only track. Mido tracks use *delta* times.
    marker_track = mido.MidiTrack()
    prev_tick = 0
    for t_s, label in page_starts:
        abs_tick = _seconds_to_ticks(t_s, mid.ticks_per_beat, tempo_us)
        delta = max(abs_tick - prev_tick, 0)
        marker_track.append(mido.MetaMessage("marker", text=label, time=delta))
        prev_tick = abs_tick
    marker_track.append(mido.MetaMessage("end_of_track", time=0))

    mid.tracks.append(marker_track)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(out_path)
    return len(page_starts) - 1  # don't count the synthetic END
