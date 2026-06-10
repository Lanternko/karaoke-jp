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


def _is_note_on(msg: mido.Message) -> bool:
    return msg.type == "note_on" and msg.velocity > 0


def _is_note_off(msg: mido.Message) -> bool:
    return msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)


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


def _merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not windows:
        return []
    windows = sorted(windows)
    merged = [windows[0]]
    for start, end in windows[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _lyric_windows(
    aligned_path: str | Path,
    *,
    margin: float,
) -> list[tuple[float, float]]:
    aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    windows: list[tuple[float, float]] = []
    for line in aligned:
        if not line.get("tokens"):
            continue
        start = float(line.get("start", 0.0))
        end = float(line.get("end", start))
        if end <= start:
            continue
        windows.append((max(0.0, start - margin), end + margin))
    return _merge_windows(windows)


def _overlaps_any(
    start_tick: int,
    end_tick: int,
    windows_ticks: list[tuple[int, int]],
) -> bool:
    return any(start_tick < win_end and end_tick > win_start for win_start, win_end in windows_ticks)


def filter_notes_to_windows(
    mid: mido.MidiFile,
    windows_s: list[tuple[float, float]],
    *,
    tempo_us: int,
) -> int:
    """Drop note events that do not overlap any lyric window.

    Returns the number of complete notes kept. Non-note messages, including
    tempo, time signature, and markers, are preserved.
    """
    windows_ticks = [
        (
            _seconds_to_ticks(start, mid.ticks_per_beat, tempo_us),
            _seconds_to_ticks(end, mid.ticks_per_beat, tempo_us),
        )
        for start, end in windows_s
        if end > start
    ]
    if not windows_ticks:
        return 0

    kept_notes = 0
    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        abs_ticks: list[int] = []
        keep = [False] * len(track)
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}

        for idx, msg in enumerate(track):
            abs_tick += msg.time
            abs_ticks.append(abs_tick)

            if _is_note_on(msg):
                key = (getattr(msg, "channel", 0), msg.note)
                active.setdefault(key, []).append((idx, abs_tick))
            elif _is_note_off(msg):
                key = (getattr(msg, "channel", 0), msg.note)
                starts = active.get(key)
                if starts:
                    start_idx, start_tick = starts.pop(0)
                    if _overlaps_any(start_tick, abs_tick, windows_ticks):
                        keep[start_idx] = True
                        keep[idx] = True
                        kept_notes += 1
            else:
                keep[idx] = True

        rebuilt = mido.MidiTrack()
        prev_tick = 0
        for msg, msg_abs_tick, should_keep in zip(track, abs_ticks, keep, strict=False):
            if not should_keep:
                continue
            rebuilt.append(msg.copy(time=max(msg_abs_tick - prev_tick, 0)))
            prev_tick = msg_abs_tick
        mid.tracks[track_idx] = rebuilt
    return kept_notes


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


def inject_beat_markers(
    midi_path: str | Path,
    out_path: str | Path,
    *,
    bpm: float,
    quarters_per_page: int = 10,
    aligned_path: str | Path | None = None,
    note_window_margin: float = 0.25,
) -> int:
    """Inject ``marker`` meta-events at fixed quarter-note intervals.

    Use this when you want every page on the bar display to render at the
    same pixels-per-quarter scale (instead of phrase-stretching). Each page
    spans exactly ``quarters_per_page`` quarter notes. A larger value shows
    more time in the same bar area, so note bars render narrower.

    Lyrics layout is unaffected: it reads from the .lrc file directly, not
    from MIDI markers. So the page boundaries here only control the bar
    area's visual scale.
    """
    midi_path = Path(midi_path)
    out_path = Path(out_path)
    if bpm <= 0:
        raise ValueError(f"bpm must be positive, got {bpm}")

    mid = mido.MidiFile(midi_path)

    tempo_us = 500_000
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo_us = msg.tempo
            break

    original_song_end_s = _midi_duration_seconds(mid, tempo_us)

    if aligned_path is not None:
        windows = _lyric_windows(aligned_path, margin=note_window_margin)
        filter_notes_to_windows(mid, windows, tempo_us=tempo_us)

    has_ts = any(
        msg.type == "time_signature" for tr in mid.tracks for msg in tr
    )
    if not has_ts:
        ts_msg = mido.MetaMessage(
            "time_signature",
            numerator=4, denominator=4,
            clocks_per_click=24, notated_32nd_notes_per_beat=8,
            time=0,
        )
        mid.tracks[0].insert(0, ts_msg)

    # Use the original note span for page markers even when notes are filtered
    # to lyric windows, so the bar display still advances across the full song.
    song_end_s = original_song_end_s
    if song_end_s <= 0:
        raise ValueError(f"Could not determine song duration from {midi_path}")

    seconds_per_quarter = 60.0 / bpm
    seconds_per_page = quarters_per_page * seconds_per_quarter

    n_pages = int(song_end_s // seconds_per_page) + 1
    page_starts: list[tuple[float, str]] = []
    for i in range(n_pages):
        page_starts.append((i * seconds_per_page, f"P{i + 1:02d}"))
    # Synthetic END marker so MID2BAR's last-page logic has somewhere to land.
    page_starts.append((song_end_s + 1.0, "END"))

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
