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
) -> tuple[int, int, int]:
    """Per-line bounded mora→note allocation.

    For each line we use Whisper's line.start/end (± margin) as a window
    into the note list, monotone-constrained against previous lines.
    Within-line drift cannot leak into the next line.

    Returns (n_lines_updated, n_morae, n_notes_used).
    """
    if not notes:
        return 0, sum(len(expand_line_to_morae(ln)) for ln in lines), 0

    note_onsets = [n[0] for n in notes]
    n_total = len(notes)
    cursor = 0
    total_morae = 0
    total_notes = 0
    updated = 0

    for line in lines:
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

        line_notes = notes[lo:hi]
        if not line_notes:
            continue

        hints = [m["hint"] for m in morae]
        spans, last_rel = _allocate_notes(line_notes, hints)
        _writeback_char_timings(morae, spans)
        total_notes += min(len(line_notes), len(morae))
        updated += 1
        # Advance past the last note actually consumed. The n_n >= n_h
        # branch can skip notes via max_skip, so chosen[-1] may exceed
        # len(morae) - 1; using min(...) here would let the next line's
        # window re-include notes already used.
        cursor = lo + last_rel + 1

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
def main(midi_path: str, aligned_path: str, out_path: str | None, mode: str, margin: float) -> None:
    notes = extract_notes(Path(midi_path))
    print(f"[midi_timing] loaded {len(notes)} notes from {midi_path}")

    aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    total_chars = sum(sum(1 for ch in line["text"] if _is_sung_char(ch)) for line in aligned)
    print(f"[midi_timing] {len(aligned)} lines, {total_chars} sung chars (mode={mode})")

    if mode == "mora":
        upd, n_morae, n_used = apply_mora_timing(aligned, notes, margin=margin)
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
