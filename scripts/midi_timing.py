"""Replace Whisper timing in aligned.json with MIDI note onsets.

Whisper word timestamps are unreliable for singing (speech-trained model,
vowels stretched over multiple beats).  SOME / RMVPE / CTC+CE backends
emit per-note onsets that capture the ACTUAL note onsets from the audio,
giving us mora-precise timing anchors.

Algorithm (mora mode, default)
------------------------------
The unit of timing is the **mora** (one kana of a token's reading), not
the surface character. A kanji compound like ``再会`` (2 chars) carries
4 morae ``さ・い・か・い`` and consumes 4 notes; the per-char start/end
is derived by splitting the run's morae across its kanji.

For each lyrics line we use Whisper's coarser line.start/end (± margin)
as a window into the global note list, monotone-constrained against
previous lines so within-line drift cannot leak into the next line. Each
line's morae are greedy-monotone matched against its windowed notes,
using the existing Whisper char start as a proximity hint.

Char timings are then derived: char.start = first mora's onset,
char.end = last mora's offset. Punctuation gets pinned to neighbouring
sung-char boundaries.

The legacy char-level matcher is reachable via ``--mode char``.

Usage
-----
    python scripts/midi_timing.py \\
        --midi   outputs/<song>/melody.mid \\
        --aligned outputs/<song>/aligned.json \\
        --out    outputs/<song>/aligned.json  # overwrites in-place
"""
from __future__ import annotations

import copy
import json
import sys
import unicodedata
from pathlib import Path

import click
import mido

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from karaoke_jp.lrc_export import split_furigana  # noqa: E402
from karaoke_jp.ruby import kata_to_hira  # noqa: E402


# ---------------------------------------------------------------------------
# MIDI helpers
# ---------------------------------------------------------------------------

def extract_notes(midi_path: Path) -> list[tuple[float, float, int]]:
    """Return list of (onset_sec, offset_sec, pitch) sorted by onset."""
    mid = mido.MidiFile(midi_path)
    tempo = 500000  # 120 BPM default
    active: dict[int, int] = {}
    events: list[tuple[float, float, int]] = []

    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = abs_tick
            elif (msg.type == "note_off" or msg.type == "note_on" and msg.velocity == 0) and (
                msg.note in active
            ):
                on_tick = active.pop(msg.note)
                on_s = mido.tick2second(on_tick, mid.ticks_per_beat, tempo)
                off_s = mido.tick2second(abs_tick, mid.ticks_per_beat, tempo)
                events.append((on_s, off_s, msg.note))

    events.sort(key=lambda e: e[0])
    return events


# ---------------------------------------------------------------------------
# Char classification
# ---------------------------------------------------------------------------

def _is_sung_char(ch: str) -> bool:
    """Return True for lyric chars that should consume a MIDI note onset."""
    if ch.isspace() or ch == "　":
        return False
    return unicodedata.category(ch)[0] not in {"P", "S"}


def _retime_unsung_chars(chars: list[dict]) -> None:
    """Pin punctuation / symbols to neighbouring sung-char boundaries.

    These chars stay in the LRC body but must not consume MIDI note onsets.
    Snap them to the closest surrounding sung-char boundary so timing stays
    monotone after the sung chars are retimed.
    """
    sung_indices = [i for i, ch in enumerate(chars) if _is_sung_char(ch["char"])]
    if not sung_indices:
        return

    prev_sung: list[int | None] = [None] * len(chars)
    next_sung: list[int | None] = [None] * len(chars)

    last_seen: int | None = None
    for i, ch in enumerate(chars):
        if _is_sung_char(ch["char"]):
            last_seen = i
        prev_sung[i] = last_seen

    last_seen = None
    for i in range(len(chars) - 1, -1, -1):
        if _is_sung_char(chars[i]["char"]):
            last_seen = i
        next_sung[i] = last_seen

    for i, ch in enumerate(chars):
        if _is_sung_char(ch["char"]):
            continue
        prev_idx = prev_sung[i]
        next_idx = next_sung[i]
        if prev_idx is not None:
            t = float(chars[prev_idx]["end"])
        elif next_idx is not None:
            t = float(chars[next_idx]["start"])
        else:
            continue
        ch["start"] = round(t, 3)
        ch["end"] = round(t, 3)


# ---------------------------------------------------------------------------
# Greedy monotone allocator (shared by char + mora modes)
# ---------------------------------------------------------------------------

def _allocate_notes(
    notes: list[tuple[float, float, int]],
    hints: list[float],
) -> tuple[list[tuple[float, float]], int]:
    """Return (spans, last_consumed_idx).

    spans: one (start, end) per hint via greedy monotone matching.
    last_consumed_idx: index into ``notes`` of the last note actually used,
    or -1 if no notes were consumed. Callers use this to advance a cursor
    past every note this allocation touched (the n_n >= n_h path may skip
    notes via ``max_skip``, so chosen[-1] can exceed n_h - 1).

    When notes >= hints: each hint picks the closest available note while
    keeping enough notes for remaining hints. End time is capped at the
    next chosen note's onset so a breath gap before the next hint doesn't
    drag the wipe past the sung sustain.

    When notes < hints: chars share notes; subdivide each note's usable
    time evenly across its char group.
    """
    n_h = len(hints)
    n_n = len(notes)
    if n_h == 0:
        return [], -1
    if n_n == 0:
        return [(h, h) for h in hints], -1

    if n_n >= n_h:
        chosen: list[int] = []
        ptr = 0
        for i, ht in enumerate(hints):
            remaining_h = n_h - i
            remaining_n = n_n - ptr
            max_skip = remaining_n - remaining_h
            best_j = ptr
            best_dist = abs(notes[ptr][0] - ht)
            for j in range(ptr + 1, ptr + max_skip + 1):
                d = abs(notes[j][0] - ht)
                if d < best_dist:
                    best_dist = d
                    best_j = j
            chosen.append(best_j)
            ptr = best_j + 1
        spans: list[tuple[float, float]] = []
        for i, note_idx in enumerate(chosen):
            on, off, _ = notes[note_idx]
            if i + 1 < n_h:
                next_on = notes[chosen[i + 1]][0]
                end = min(off, next_on)
            else:
                end = off
            spans.append((on, end))
        return spans, chosen[-1]

    per = n_h / n_n
    note_idxs = [min(int(i / per), n_n - 1) for i in range(n_h)]
    spans: list[tuple[float, float]] = []
    i = 0
    while i < n_h:
        ni = note_idxs[i]
        j = i + 1
        while j < n_h and note_idxs[j] == ni:
            j += 1
        on, off, _ = notes[ni]
        usable = notes[ni + 1][0] if ni + 1 < n_n else off
        usable = max(usable, on)
        group = j - i
        sp = usable - on
        for offset in range(group):
            s = on + sp * offset / group
            e = on + sp * (offset + 1) / group
            spans.append((s, e))
        i = j
    return spans, n_n - 1


def _allocate_notes_dp(
    notes: list[tuple[float, float, int]],
    hints: list[float],
    *,
    skip_penalty: float = 0.20,
    extra_note_penalty: float = 0.06,
    max_notes_per_mora: int = 4,
) -> tuple[list[tuple[float, float]], int]:
    """Dynamic-programming note→mora allocator.

    The greedy allocator picks one note onset per mora, with a special fallback
    when there are fewer notes than morae. This DP handles the opposite case
    explicitly: one mora may own multiple consecutive notes (melisma / sustained
    vowel across notes), while other notes may be skipped with a penalty.

    Cost terms are deliberately simple and in seconds-equivalent units:

    * onset distance from the mora hint to the first note in the group;
    * ``extra_note_penalty`` for each additional note owned by that mora;
    * ``skip_penalty`` for notes left unassigned.

    Returns spans plus the last note index actually assigned to a mora. Skipped
    trailing notes stay available to the following line via the caller cursor.
    """
    n_h = len(hints)
    n_n = len(notes)
    if n_h == 0:
        return [], -1
    if n_n == 0:
        return [(h, h) for h in hints], -1
    if n_n < n_h:
        return _allocate_notes(notes, hints)

    max_group = max(1, max_notes_per_mora)
    inf = float("inf")
    # dp[i][j] = best cost after assigning i morae, considering notes before j.
    dp = [[inf] * (n_n + 1) for _ in range(n_h + 1)]
    back: list[list[tuple[int, int, int] | None]] = [
        [None] * (n_n + 1) for _ in range(n_h + 1)
    ]
    dp[0][0] = 0.0

    for i in range(n_h + 1):
        for j in range(n_n):
            cur = dp[i][j]
            if cur == inf:
                continue
            # Skip note j. This keeps it out of any mora group but pays a cost.
            if cur + skip_penalty < dp[i][j + 1]:
                dp[i][j + 1] = cur + skip_penalty
                back[i][j + 1] = (i, j, -1)
            if i >= n_h:
                continue
            # Assign one consecutive note group to mora i.
            for k in range(j, min(n_n, j + max_group)):
                group_len = k - j + 1
                onset = notes[j][0]
                cost = (
                    cur
                    + abs(onset - hints[i])
                    + extra_note_penalty * (group_len - 1)
                )
                if cost < dp[i + 1][k + 1]:
                    dp[i + 1][k + 1] = cost
                    back[i + 1][k + 1] = (i, j, k)

    end_j = min(range(n_n + 1), key=lambda j: dp[n_h][j] + skip_penalty * (n_n - j))
    if dp[n_h][end_j] == inf:
        return _allocate_notes(notes, hints)

    groups: list[tuple[int, int]] = []
    i, j = n_h, end_j
    while i > 0:
        prev = back[i][j]
        if prev is None:
            return _allocate_notes(notes, hints)
        pi, pj, pk = prev
        if pk == -1:
            i, j = pi, pj
            continue
        groups.append((pj, pk))
        i, j = pi, pj
    groups.reverse()
    if len(groups) != n_h:
        return _allocate_notes(notes, hints)

    spans = [(notes[start][0], notes[end][1]) for start, end in groups]
    return spans, groups[-1][1]


# ---------------------------------------------------------------------------
# Mora mode (default)
# ---------------------------------------------------------------------------

def _split_morae_across_chars(n_morae: int, n_chars: int) -> list[int]:
    """Even split: how many morae per char in a kanji run.

    Correct for ``再会 (さい+かい)``; wrong for ``名前 (な+まえ)`` where
    UniDic per-kanji lookup would do better. M3 v2 punt.
    """
    if n_chars <= 0:
        return []
    base = n_morae // n_chars
    rem = n_morae % n_chars
    return [base + (1 if i < rem else 0) for i in range(n_chars)]


def expand_line_to_morae(line: dict) -> list[dict]:
    """Yield mora records for one line, each carrying a back-pointer to
    its target char dict so we can write timings back later.

    A mora record looks like::

        {"kana": "さ", "char": <ref to char dict in line>, "hint": <whisper start>}

    Punctuation / symbols get *zero* morae (they don't consume notes).
    """
    morae: list[dict] = []
    for tok in line.get("tokens", []):
        chars = tok.get("chars") or []
        if not chars:
            continue
        reading = tok.get("reading")
        if reading and not tok.get("kana_only"):
            segments = split_furigana(tok["surface"], kata_to_hira(reading))
        else:
            segments = [(tok["surface"], None, 0, len(tok["surface"]))]

        for _seg_text, seg_reading, c_start, c_end in segments:
            seg_chars = chars[c_start:c_end]
            if not seg_chars:
                continue
            sung_seg_chars = [c for c in seg_chars if _is_sung_char(c["char"])]
            if not sung_seg_chars:
                continue

            if seg_reading is None:
                # Kana / okurigana: 1 mora per sung char.
                for ch in sung_seg_chars:
                    morae.append({"kana": ch["char"], "char": ch, "hint": float(ch["start"])})
            else:
                # Kanji run: evenly distribute reading kana across the kanji.
                morae_seq = list(seg_reading)
                splits = _split_morae_across_chars(len(morae_seq), len(sung_seg_chars))
                pos = 0
                for ch, n in zip(sung_seg_chars, splits, strict=True):
                    if n == 0:
                        continue
                    for i in range(n):
                        morae.append(
                            {
                                "kana": morae_seq[pos + i],
                                "char": ch,
                                "hint": float(ch["start"]),
                            }
                        )
                    pos += n
    return morae


def _writeback_char_timings(
    morae: list[dict],
    spans: list[tuple[float, float]],
) -> None:
    """char.start = first mora's onset, char.end = last mora's offset."""
    by_char: dict[int, list[tuple[float, float]]] = {}
    for m, sp in zip(morae, spans, strict=True):
        cid = id(m["char"])
        by_char.setdefault(cid, []).append(sp)

    seen = set()
    for m in morae:
        cid = id(m["char"])
        if cid in seen:
            continue
        seen.add(cid)
        sp_list = by_char[cid]
        m["char"]["start"] = round(sp_list[0][0], 3)
        m["char"]["end"] = round(sp_list[-1][1], 3)


def apply_mora_timing(
    lines: list[dict],
    notes: list[tuple[float, float, int]],
    margin: float = 0.4,
    allocator: str = "greedy",
    first_mora_min_delay: float | None = None,
    first_mora_gate_prev_gap: float | None = None,
    first_mora_gate_lead_tolerance: float | None = None,
    absorb_trailing_notes: bool = False,
    next_line_hint_guard: float | None = None,
    next_line_hint_min_start_delay: float | None = None,
    dp_skip_penalty: float = 0.20,
    dp_extra_note_penalty: float = 0.06,
    dp_max_notes_per_mora: int = 4,
) -> tuple[int, int, int]:
    """Per-line bounded mora→note allocation.

    For each line we use Whisper's line.start/end (± margin) as a window
    into the note list, monotone-constrained against previous lines.
    Within-line drift cannot leak into the next line.

    ``first_mora_min_delay`` is an experimental boundary prior. When enabled,
    the first mora of each line cannot take a note whose onset is earlier than
    ``line.start + first_mora_min_delay``. This is useful for sidecar tests of
    butted-boundary failures where a new line steals the previous line's sustain
    note. The default is disabled so canonical timing remains unchanged.
    ``first_mora_gate_prev_gap`` and ``first_mora_gate_lead_tolerance`` restrict
    that prior to likely stress boundaries: adjacent lyrics lines, or a first
    note that starts implausibly before the ASR line start. If neither guard is
    set, the prior applies to every line (the original pilot behavior).

    ``absorb_trailing_notes`` is another experimental score-informed prior. When
    the line window has more notes than morae, any notes left after the greedy
    assignment are treated as a melisma/sustain extension of the final mora and
    consumed by the current line. This models the literature's many-notes-per-
    vowel case, but it is intentionally opt-in because it needs cross-song guard
    validation before becoming canonical.

    ``next_line_hint_guard`` is an experimental line-boundary guard. When set,
    the current line may not consume note onsets at or after
    ``next_line.start + guard``. This prevents a line-final mora from stealing a
    note that the original ASR/RMS hint already places inside the next line.
    ``next_line_hint_min_start_delay`` makes that guard safer by first running a
    dry no-guard allocation and only enabling the guard for boundaries whose
    next line still starts at least this many seconds after its ASR/RMS hint.

    ``allocator="dp"`` switches the within-line note→mora assignment from greedy
    nearest-note matching to a local DP that can group multiple notes under one
    mora and skip implausible notes with penalties. It is opt-in for ablation.

    Returns (n_lines_updated, n_morae, n_notes_consumed_by_cursor).
    """
    if not notes:
        return 0, sum(len(expand_line_to_morae(ln)) for ln in lines), 0

    note_onsets = [n[0] for n in notes]
    n_total = len(notes)
    cursor = 0
    total_morae = 0
    total_notes = 0
    updated = 0
    prev_raw_line_end: float | None = None
    guarded_boundaries: set[int] | None = None

    if next_line_hint_guard is not None and next_line_hint_min_start_delay is not None:
        dry_lines = copy.deepcopy(lines)
        apply_mora_timing(
            dry_lines,
            notes,
            margin=margin,
            allocator=allocator,
            first_mora_min_delay=first_mora_min_delay,
            first_mora_gate_prev_gap=first_mora_gate_prev_gap,
            first_mora_gate_lead_tolerance=first_mora_gate_lead_tolerance,
            absorb_trailing_notes=absorb_trailing_notes,
            next_line_hint_guard=None,
            dp_skip_penalty=dp_skip_penalty,
            dp_extra_note_penalty=dp_extra_note_penalty,
            dp_max_notes_per_mora=dp_max_notes_per_mora,
        )
        guarded_boundaries = set()
        for boundary_idx in range(len(lines) - 1):
            next_line = lines[boundary_idx + 1]
            next_morae = expand_line_to_morae(next_line)
            if not next_morae:
                continue
            hint_start = float(next_line.get("start", next_morae[0]["hint"]))
            dry_morae = expand_line_to_morae(dry_lines[boundary_idx + 1])
            if not dry_morae:
                continue
            dry_start = float(dry_lines[boundary_idx + 1].get("start", dry_morae[0]["hint"]))
            if dry_start - hint_start >= next_line_hint_min_start_delay:
                guarded_boundaries.add(boundary_idx)

    for line_idx, line in enumerate(lines):
        morae = expand_line_to_morae(line)
        if not morae:
            continue
        total_morae += len(morae)

        line_start = float(line.get("start", morae[0]["hint"]))
        line_end = float(line.get("end", morae[-1]["hint"]))

        lo = cursor
        while lo < n_total and note_onsets[lo] < line_start - margin:
            lo += 1
        hi = lo
        while hi < n_total and note_onsets[hi] <= line_end + margin:
            hi += 1
        if (
            next_line_hint_guard is not None
            and line_idx + 1 < len(lines)
            and (guarded_boundaries is None or line_idx in guarded_boundaries)
        ):
            next_line = lines[line_idx + 1]
            next_morae = expand_line_to_morae(next_line)
            if next_morae:
                next_start = float(next_line.get("start", next_morae[0]["hint"]))
                cap = next_start + next_line_hint_guard
                while hi > lo and note_onsets[hi - 1] >= cap:
                    hi -= 1

        line_notes = notes[lo:hi]
        if not line_notes:
            continue

        gate_rel = 0
        if first_mora_min_delay is not None:
            has_guard = (
                first_mora_gate_prev_gap is not None
                or first_mora_gate_lead_tolerance is not None
            )
            gate_this_line = not has_guard
            if first_mora_gate_prev_gap is not None and prev_raw_line_end is not None:
                gate_this_line = gate_this_line or (
                    line_start - prev_raw_line_end <= first_mora_gate_prev_gap
                )
            if first_mora_gate_lead_tolerance is not None:
                gate_this_line = gate_this_line or (
                    line_notes[0][0] < line_start - first_mora_gate_lead_tolerance
                )

            if gate_this_line:
                gate_time = line_start + first_mora_min_delay
                while gate_rel < len(line_notes) and line_notes[gate_rel][0] < gate_time:
                    gate_rel += 1
                # If the gate would discard the whole window, leave this line on
                # the normal allocator instead of collapsing it back to Whisper
                # timing.
                if gate_rel >= len(line_notes):
                    gate_rel = 0

        hints = [m["hint"] for m in morae]
        gated_notes = line_notes[gate_rel:]
        if allocator == "dp":
            spans, last_rel = _allocate_notes_dp(
                gated_notes,
                hints,
                skip_penalty=dp_skip_penalty,
                extra_note_penalty=dp_extra_note_penalty,
                max_notes_per_mora=dp_max_notes_per_mora,
            )
        else:
            spans, last_rel = _allocate_notes(gated_notes, hints)
        if (
            absorb_trailing_notes
            and spans
            and len(gated_notes) > len(morae)
            and last_rel + 1 < len(gated_notes)
        ):
            trailing = gated_notes[last_rel + 1 :]
            spans[-1] = (spans[-1][0], trailing[-1][1])
            last_rel = len(gated_notes) - 1
        _writeback_char_timings(morae, spans)
        total_notes += gate_rel + max(0, last_rel + 1)
        updated += 1
        # Advance past the last note actually consumed. The n_n >= n_h
        # branch can skip notes via max_skip, so chosen[-1] may exceed
        # len(morae) - 1; using min(...) here would let the next line's
        # window re-include notes already used.
        cursor = lo + gate_rel + last_rel + 1
        prev_raw_line_end = line_end

    _retime_lines_after_char_update(lines)
    return updated, total_morae, total_notes


# ---------------------------------------------------------------------------
# Char mode (legacy, --mode char)
# ---------------------------------------------------------------------------

def apply_char_timing(
    lines: list[dict],
    notes: list[tuple[float, float, int]],
) -> tuple[int, int]:
    """Global char→note allocation. One note per surface character.

    Kept as fallback; mora mode is preferred because kanji compounds carry
    multiple morae per char and char-mode under-uses the available notes.
    """
    flat: list[tuple[dict, dict]] = []
    for line in lines:
        for tok in line.get("tokens", []):
            for ch in tok.get("chars", []):
                if _is_sung_char(ch["char"]):
                    flat.append((line, ch))

    if not flat:
        return 0, 0
    if not notes:
        return 0, len(lines)

    hints = [ch["start"] for _, ch in flat]
    spans, _ = _allocate_notes(notes, hints)
    for (_line, ch_dict), (s, e) in zip(flat, spans, strict=True):
        ch_dict["start"] = round(s, 3)
        ch_dict["end"] = round(e, 3)

    updated = _retime_lines_after_char_update(lines)
    return updated, 0


# ---------------------------------------------------------------------------
# Shared post-processing
# ---------------------------------------------------------------------------

def _retime_lines_after_char_update(lines: list[dict]) -> int:
    updated = 0
    for line in lines:
        all_chars = [
            ch for tok in line.get("tokens", []) for ch in (tok.get("chars") or [])
        ]
        sung = [c for c in all_chars if _is_sung_char(c["char"])]
        if not sung:
            continue
        _retime_unsung_chars(all_chars)
        for tok in line.get("tokens", []):
            tok_chars = tok.get("chars") or []
            sung_tok = [c for c in tok_chars if _is_sung_char(c["char"])]
            timing = sung_tok or tok_chars
            if timing:
                tok["start"] = timing[0]["start"]
                tok["end"] = timing[-1]["end"]
        line["start"] = round(min(c["start"] for c in sung), 3)
        line["end"] = round(max(c["end"] for c in sung), 3)
        updated += 1
    return updated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--midi", "midi_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="melody.mid (output of M2 backend; rmvpe gives best mora fit).",
)
@click.option(
    "--aligned", "aligned_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="aligned.json produced by align_lyrics.py.",
)
@click.option(
    "--out", "out_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output path. Defaults to overwriting --aligned in-place.",
)
@click.option(
    "--mode", "mode",
    type=click.Choice(["mora", "char"]),
    default="mora", show_default=True,
    help="mora = mora→note (kanji compounds get multiple notes); "
         "char = legacy char→note.",
)
@click.option(
    "--margin", default=0.4, show_default=True,
    help="Window margin (s) around Whisper line boundaries (mora mode only).",
)
@click.option(
    "--allocator",
    type=click.Choice(["greedy", "dp"]),
    default="greedy",
    show_default=True,
    help="Mora-mode note allocator. dp is experimental sidecar mode.",
)
@click.option(
    "--first-mora-min-delay",
    type=click.FloatRange(min=0.0),
    default=None,
    help="Experimental sidecar prior (mora mode only): require each line's "
         "first mora note onset to be at least line.start + this many seconds. "
         "Default disabled.",
)
@click.option(
    "--first-mora-gate-prev-gap",
    type=click.FloatRange(min=0.0),
    default=None,
    help="Guard for --first-mora-min-delay: only gate lines whose original "
         "ASR gap from the previous line is at most this many seconds.",
)
@click.option(
    "--first-mora-gate-lead-tolerance",
    type=click.FloatRange(min=0.0),
    default=None,
    help="Guard for --first-mora-min-delay: only gate lines whose first MIDI "
         "candidate starts more than this many seconds before line.start.",
)
@click.option(
    "--absorb-trailing-notes",
    is_flag=True,
    help="Experimental sidecar prior (mora mode only): consume notes left after "
         "the final assigned mora as a final-vowel melisma. Default disabled.",
)
@click.option(
    "--next-line-hint-guard",
    type=click.FloatRange(min=0.0),
    default=None,
    help="Experimental sidecar prior (mora mode only): prevent the current line "
         "from consuming notes at/after next_line.start + this many seconds.",
)
@click.option(
    "--next-line-hint-min-start-delay",
    type=click.FloatRange(min=0.0),
    default=None,
    help="Guard for --next-line-hint-guard: first run a dry no-guard allocation "
         "and only cap boundaries whose next line would start at least this "
         "many seconds after its ASR/RMS hint.",
)
@click.option("--dp-skip-penalty", default=0.20, show_default=True,
              help="DP allocator cost for skipping a note.")
@click.option("--dp-extra-note-penalty", default=0.06, show_default=True,
              help="DP allocator cost per extra note owned by one mora.")
@click.option("--dp-max-notes-per-mora", default=4, show_default=True,
              help="DP allocator maximum notes one mora may own.")
def main(
    midi_path: str,
    aligned_path: str,
    out_path: str | None,
    mode: str,
    margin: float,
    allocator: str,
    first_mora_min_delay: float | None,
    first_mora_gate_prev_gap: float | None,
    first_mora_gate_lead_tolerance: float | None,
    absorb_trailing_notes: bool,
    next_line_hint_guard: float | None,
    next_line_hint_min_start_delay: float | None,
    dp_skip_penalty: float,
    dp_extra_note_penalty: float,
    dp_max_notes_per_mora: int,
) -> None:
    notes = extract_notes(Path(midi_path))
    print(f"[midi_timing] loaded {len(notes)} notes from {midi_path}")

    aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    total_chars = sum(sum(1 for ch in line["text"] if _is_sung_char(ch)) for line in aligned)
    print(f"[midi_timing] {len(aligned)} lines, {total_chars} sung chars (mode={mode})")

    if mode == "mora":
        upd, n_morae, n_used = apply_mora_timing(
            aligned,
            notes,
            margin=margin,
            allocator=allocator,
            first_mora_min_delay=first_mora_min_delay,
            first_mora_gate_prev_gap=first_mora_gate_prev_gap,
            first_mora_gate_lead_tolerance=first_mora_gate_lead_tolerance,
            absorb_trailing_notes=absorb_trailing_notes,
            next_line_hint_guard=next_line_hint_guard,
            next_line_hint_min_start_delay=next_line_hint_min_start_delay,
            dp_skip_penalty=dp_skip_penalty,
            dp_extra_note_penalty=dp_extra_note_penalty,
            dp_max_notes_per_mora=dp_max_notes_per_mora,
        )
        print(f"[midi_timing] {n_morae} morae across {upd} lines, {n_used} notes consumed")
    else:
        upd, kept = apply_char_timing(aligned, notes)
        print(f"[midi_timing] updated {upd} lines, kept Whisper for {kept}")

    dest = Path(out_path) if out_path else Path(aligned_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[midi_timing] wrote {dest}")


if __name__ == "__main__":
    main()
