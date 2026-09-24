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


def _ticks_to_seconds(ticks: int, ticks_per_beat: int, tempo_us: int) -> float:
    return ticks * tempo_us / (ticks_per_beat * 1_000_000)


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
    aligned_path: str | Path | list,
    *,
    margin: float,
    tail_allowance: float = 0.0,
) -> list[tuple[float, float]]:
    # accepts the parsed aligned data directly so callers that patch it
    # in memory (lyric_recut in make_portrait_grid) gate on the fixed times
    if isinstance(aligned_path, list):
        aligned = aligned_path
    else:
        aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    raw: list[tuple[float, float]] = []
    for line in aligned:
        if not line.get("tokens"):
            continue
        start = float(line.get("start", 0.0))
        end = float(line.get("end", start))
        if end <= start:
            continue
        raw.append((start, end))
    raw.sort()
    windows: list[tuple[float, float]] = []
    for i, (start, end) in enumerate(raw):
        if tail_allowance > 0.0:
            # phrase-tail sustains/falls legitimately start right after the
            # aligned lyric ends; extend toward (not into) the next line so
            # they survive the gate that kills instrumental-gap detections
            limit = raw[i + 1][0] - 0.05 if i + 1 < len(raw) else end + tail_allowance
            end = max(end, min(end + tail_allowance, limit))
        windows.append((max(0.0, start - margin), end + margin))
    return _merge_windows(windows)


def _voiced_windows(
    segments_path: str | Path,
    *,
    pad: float = 0.3,
) -> list[tuple[float, float]]:
    """Voiced windows from rms_vad_segments.py output.

    A misaligned lyric line can claim a window spanning an instrumental
    break (e.g. an ad-lib line with no kana evidence stretched across a
    16 s interlude); lyric windows alone then let separation-bleed ghost
    notes through. RMS voiced segments are independent of alignment, so
    intersecting with them kills interlude notes generically.
    """
    data = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    segments = data["segments"] if isinstance(data, dict) else data
    raw: list[tuple[float, float]] = []
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        if end > start:
            raw.append((max(0.0, start - pad), end + pad))
    return _merge_windows(raw)


def _intersect_windows(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Intersect two sorted, merged window lists."""
    out: list[tuple[float, float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            out.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


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
    note_tail_allowance: float = 1.0,
    rms_segments_path: str | Path | None = None,
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
        windows = _lyric_windows(
            aligned_path, margin=note_window_margin,
            tail_allowance=note_tail_allowance,
        )
        if rms_segments_path is not None:
            windows = _intersect_windows(windows, _voiced_windows(rms_segments_path))
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


def inject_pack_markers(
    midi_path: str | Path,
    out_path: str | Path,
    *,
    page_seconds: float = 9.0,
    phrase_gap: float = 0.8,
    lead: float = 0.30,
    trail: float = 0.20,
    aligned_path: str | Path | None = None,
    note_window_margin: float = 0.25,
    note_tail_allowance: float = 1.0,
    rms_segments_path: str | Path | None = None,
) -> int:
    """Content-aligned pages at a (near-)fixed visual scale.

    Every content page targets exactly ``page_seconds`` of span, so bar
    width per second stays constant across pages — no fat bars on short
    pages or sliver bars on long ones. Phrases (note clusters split at gaps
    > ``phrase_gap``; oversized chorus runs split at their largest internal
    gap) pack greedily: a phrase that cannot finish inside the page moves
    WHOLE to the next page (truncate-to-next-line). Unused right side stays
    blank, like the TV reference. Long gaps become their own empty filler
    page (scale is invisible without bars). A page only shrinks below
    ``page_seconds`` when the next phrase starts before the page budget
    ends — the bounded price of never splitting a phrase across pages.
    """
    midi_path = Path(midi_path)
    out_path = Path(out_path)

    mid = mido.MidiFile(midi_path)
    tempo_us = 500_000
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo_us = msg.tempo
            break

    song_end_s = _midi_duration_seconds(mid, tempo_us)

    if aligned_path is not None:
        windows = _lyric_windows(
            aligned_path, margin=note_window_margin,
            tail_allowance=note_tail_allowance,
        )
        if rms_segments_path is not None:
            windows = _intersect_windows(windows, _voiced_windows(rms_segments_path))
        filter_notes_to_windows(mid, windows, tempo_us=tempo_us)

    notes: list[tuple[float, float]] = []
    for track in mid.tracks:
        abs_tick = 0
        active: dict[int, int] = {}
        for msg in track:
            abs_tick += msg.time
            if _is_note_on(msg):
                active[msg.note] = abs_tick
            elif _is_note_off(msg) and msg.note in active:
                start_tick = active.pop(msg.note)
                notes.append((
                    _ticks_to_seconds(start_tick, mid.ticks_per_beat, tempo_us),
                    _ticks_to_seconds(abs_tick, mid.ticks_per_beat, tempo_us),
                ))
    notes.sort()
    if not notes:
        raise ValueError(f"No notes to paginate in {midi_path}")

    clusters: list[list[tuple[float, float]]] = [[notes[0]]]
    for n in notes[1:]:
        if n[0] - clusters[-1][-1][1] > phrase_gap:
            clusters.append([n])
        else:
            clusters[-1].append(n)

    def split_oversized(cluster: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
        budget = page_seconds - lead - trail
        if cluster[-1][1] - cluster[0][0] <= budget or len(cluster) < 2:
            return [cluster]
        gaps = [(cluster[i + 1][0] - cluster[i][1], i) for i in range(len(cluster) - 1)]
        mid_idx = (len(cluster) - 1) / 2
        _gap, cut = max(gaps, key=lambda g: (round(g[0], 3), -abs(g[1] - mid_idx)))
        return split_oversized(cluster[: cut + 1]) + split_oversized(cluster[cut + 1:])

    phrases: list[tuple[float, float]] = []
    for cluster in clusters:
        for part in split_oversized(cluster):
            phrases.append((part[0][0], part[-1][1]))

    budget = page_seconds - lead - trail

    # plan pages as phrase-index groups (greedy fill)
    pages: list[list[int]] = [[0]]
    for i in range(1, len(phrases)):
        first = phrases[pages[-1][0]][0]
        if phrases[i][1] - first <= budget:
            pages[-1].append(i)
        else:
            pages.append([i])

    # balance pass: a page left with a single short phrase (because its
    # neighbour could not fit) borrows trailing phrases from the previous
    # page so adjacent scales stay even instead of one page rendering fat
    def span(page: list[int]) -> float:
        return phrases[page[-1]][1] - phrases[page[0]][0]

    for _ in range(2):
        for i in range(len(pages) - 1, 0, -1):
            prev, cur = pages[i - 1], pages[i]
            while len(prev) >= 2 and span(cur) < 0.65 * budget:
                cand = prev[-1]
                if phrases[cur[-1]][1] - phrases[cand][0] > budget:
                    break
                cur.insert(0, prev.pop())

    boundaries: list[float] = []
    for i in range(len(pages) - 1):
        P = phrases[pages[i][0]][0] - lead
        content_end = phrases[pages[i][-1]][1]
        next_start = phrases[pages[i + 1][0]][0]
        cut = min(P + page_seconds, next_start - lead)
        cut = max(cut, content_end + 0.05)
        cut = min(cut, next_start - 0.05)
        boundaries.append(cut)
        if next_start - lead - cut > 0.5:
            boundaries.append(next_start - lead)  # empty filler page
    content_end = phrases[pages[-1][-1]][1]

    page_starts: list[tuple[float, str]] = [(max(0.0, phrases[0][0] - lead), "P01")]
    page_starts += [(t, f"P{i + 2:02d}") for i, t in enumerate(sorted(set(boundaries)))]
    page_starts.append((max(song_end_s, content_end or 0.0) + 1.0, "END"))

    has_ts = any(msg.type == "time_signature" for tr in mid.tracks for msg in tr)
    if not has_ts:
        mid.tracks[0].insert(0, mido.MetaMessage(
            "time_signature", numerator=4, denominator=4,
            clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0,
        ))

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
