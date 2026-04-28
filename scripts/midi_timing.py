"""Replace Whisper char-level timing in aligned.json with MIDI note onsets.

Whisper word timestamps are unreliable for singing (speech-trained model,
vowels stretched over multiple beats).  SOME's melody.mid captures the ACTUAL
note onsets from the audio, giving us mora-precise timing anchors.

Algorithm
---------
For each lyrics line (using Whisper's coarser *line* boundary as a window):
1. Collect MIDI notes that fall inside that window (±tolerance).
2. Greedily match one note per character using the character's existing
   Whisper timestamp as a proximity hint (monotone left-to-right assignment).
3. Assign: char_start = note_onset, char_end = next_char_onset | note_offset.
4. Update token-level and line-level start/end to match.

Lines with no notes in the window keep their original Whisper timing.

Usage
-----
    python scripts/midi_timing.py \\
        --midi   outputs/<song>/melody.mid \\
        --aligned outputs/<song>/aligned.json \\
        --out    outputs/<song>/aligned.json  # overwrites in-place by default
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import click
import mido

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# MIDI helpers
# ---------------------------------------------------------------------------

def extract_notes(midi_path: Path) -> list[tuple[float, float, int]]:
    """Return list of (onset_sec, offset_sec, pitch) sorted by onset.

    Handles type-0 and type-1 files with a single tempo.
    """
    mid = mido.MidiFile(midi_path)
    tempo = 500000  # 120 BPM default
    active: dict[int, int] = {}  # pitch -> abs_tick of note_on
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
# Matching
# ---------------------------------------------------------------------------

def _match_notes_to_chars(
    notes: list[tuple[float, float, int]],
    whisper_times: list[float],
) -> list[tuple[float, float]] | None:
    """Greedy monotone assignment: one note per character.

    *whisper_times*: the char's existing Whisper onset estimate, used as a
    proximity hint for which note to pick when multiple candidates exist.

    Returns None if there are no notes (caller keeps Whisper timing).
    The returned tuples are per-char ``(start, end)`` spans. When there are
    fewer notes than chars, multiple chars can share a note; in that case we
    subdivide the available note duration so every char still gets a positive
    span instead of collapsing to zero length.
    """
    if not notes:
        return None

    n_c = len(whisper_times)
    if n_c == 0:
        return []

    n_n = len(notes)

    if n_n >= n_c:
        # Greedy: for each char pick the closest available note while keeping
        # enough notes for the remaining chars.
        chosen: list[int] = []
        ptr = 0
        for i, wt in enumerate(whisper_times):
            remaining_chars = n_c - i
            remaining_notes = n_n - ptr
            # Maximum notes we can skip and still have enough for the rest
            max_skip = remaining_notes - remaining_chars
            best_j = ptr
            best_dist = abs(notes[ptr][0] - wt)
            for j in range(ptr + 1, ptr + max_skip + 1):
                dist = abs(notes[j][0] - wt)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            chosen.append(best_j)
            ptr = best_j + 1
        result: list[tuple[float, float]] = []
        for i, note_idx in enumerate(chosen):
            note_on, note_off, _ = notes[note_idx]
            note_end = notes[chosen[i + 1]][0] if i + 1 < len(chosen) else note_off
            result.append((note_on, note_end))
        return result

    else:
        # Fewer notes than chars: assign chars to notes monotonically, then
        # subdivide each note's usable time span across its char group.
        result: list[tuple[float, float]] = []
        chars_per_note = n_c / n_n
        note_idxs = [min(int(i / chars_per_note), n_n - 1) for i in range(n_c)]
        i = 0
        while i < n_c:
            note_idx = note_idxs[i]
            j = i + 1
            while j < n_c and note_idxs[j] == note_idx:
                j += 1

            note_on, note_off, _ = notes[note_idx]
            usable_end = notes[note_idx + 1][0] if note_idx + 1 < n_n else note_off
            usable_end = max(usable_end, note_on)

            group_size = j - i
            span = usable_end - note_on
            for offset in range(group_size):
                ch_start = note_on + span * offset / group_size
                ch_end = note_on + span * (offset + 1) / group_size
                result.append((ch_start, ch_end))
            i = j
        return result


def _is_sung_char(ch: str) -> bool:
    """Return True for lyric chars that should consume a MIDI note onset."""
    if ch.isspace() or ch == "　":
        return False
    return unicodedata.category(ch)[0] not in {"P", "S"}


def _retime_unsung_chars(chars: list[dict]) -> None:
    """Pin punctuation / symbols to neighbouring sung-char boundaries.

    These chars should stay in the output LRC body, but must not consume MIDI
    note onsets. We therefore snap them to the closest surrounding sung-char
    boundary so timing stays monotone after the sung chars are retimed.
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
# Main logic
# ---------------------------------------------------------------------------

def apply_midi_timing(
    lines: list[dict],
    notes: list[tuple[float, float, int]],
    window_margin: float = 0.4,
) -> tuple[int, int]:
    """Mutate *lines* in-place, replacing char-level timing with MIDI onsets.

    Returns (n_lines_updated, n_lines_kept_whisper).
    """
    updated = 0
    kept = 0

    for line in lines:
        all_chars = [
            ch
            for tok in line.get("tokens", [])
            for ch in tok.get("chars", [])
        ]
        char_refs = [ch for ch in all_chars if _is_sung_char(ch["char"])]

        if not char_refs:
            continue

        t0 = line["start"] - window_margin
        t1 = line["end"] + window_margin
        window_notes = [(on, off, p) for on, off, p in notes if t0 <= on <= t1]

        whisper_times = [c["start"] for c in char_refs]
        matched = _match_notes_to_chars(window_notes, whisper_times)

        if matched is None:
            kept += 1
            continue

        # Write sung-char timing directly from the matched MIDI spans.
        for ch_dict, (char_start, char_end) in zip(char_refs, matched, strict=False):
            ch_dict["start"] = round(char_start, 3)
            ch_dict["end"] = round(char_end, 3)

        _retime_unsung_chars(all_chars)

        # Re-derive token timing from updated chars.
        for tok in line.get("tokens", []):
            tok_chars = tok.get("chars", [])
            sung_tok_chars = [c for c in tok_chars if _is_sung_char(c["char"])]
            timing_chars = sung_tok_chars or tok_chars
            if timing_chars:
                tok["start"] = timing_chars[0]["start"]
                tok["end"] = timing_chars[-1]["end"]

        # Re-derive line timing from sung chars so decorative punctuation does
        # not move marker placement or ruby windows by itself.
        all_starts = [c["start"] for c in char_refs]
        all_ends = [c["end"] for c in char_refs]
        line["start"] = round(min(all_starts), 3)
        line["end"] = round(max(all_ends), 3)

        updated += 1

    return updated, kept


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--midi", "midi_path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="melody.mid (output of M2 SOME inference).",
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
    help="Output path.  Defaults to overwriting --aligned in-place.",
)
@click.option(
    "--margin", default=0.4, show_default=True,
    help="Window margin (s) around Whisper line boundaries when collecting notes.",
)
def main(midi_path: str, aligned_path: str, out_path: str | None, margin: float) -> None:
    notes = extract_notes(Path(midi_path))
    print(f"[midi_timing] loaded {len(notes)} notes from {midi_path}")

    aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    total_chars = sum(sum(1 for ch in line["text"] if _is_sung_char(ch)) for line in aligned)
    print(f"[midi_timing] {len(aligned)} lines, {total_chars} sung chars")

    upd, kept = apply_midi_timing(aligned, notes, window_margin=margin)
    print(f"[midi_timing] updated {upd} lines with MIDI timing, kept Whisper for {kept} lines")

    dest = Path(out_path) if out_path else Path(aligned_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[midi_timing] wrote {dest}")


if __name__ == "__main__":
    main()
