"""Targeted line-start boundary override (sidecar, explicit lines only).

The gold-label eval on haru-hikage showed three lines (24/25/27) whose *ends*
stay ~0.6 s early because they are butted against the *next* line whose start is
itself ~0.6 s too early — ``midi_timing``'s mora→note assignment buried the line
boundary inside the previous phrase's sustain.  ``line_end_repair`` cannot fix
this: its next-line guard (correctly) refuses to extend a line into the next
line's onset.

A *fully automatic* boundary repair is unsafe: in legato passages the RMS
envelope cannot distinguish a misplaced boundary (line 25 start 0.6 s early) from
a correctly-placed one whose next syllable simply lands ~0.6 s later (line 27,
line 31) — energy heuristics fire on both and would clip the onset of already
correct lines.  So this tool does **not** auto-detect.  It moves *only* the line
indices you name (``--set LINE:TIME``), where TIME is the true vocal onset — in
practice the MIDI note onset that gold confirms (lines validated to <35 ms).

For each named line it pins the start to TIME and greedily re-snaps the leading
mis-placed chars (those starting before TIME) onto consecutive MIDI note onsets,
cascading through the off-by-one-note collisions (e.g. line 26's た/い/せ) until a
char already has slack.  Char ends stay contiguous (``end == next start``), which
matches how ``midi_timing`` already lays out within-word chars.  The previous
line's *end* is left for a subsequent ``line_end_repair`` pass to extend into the
freed gap.

Sidecar only: canonical ``aligned*_midi.json`` is untouched, and ``midi_timing``
is not modified.

Usage:
    python scripts/boundary_override.py \
        --aligned outputs/<song>/aligned.vad_midi.json \
        --midi    outputs/<song>/melody_quantized.mid \
        --set 25:143.530 --set 26:147.550 --set 28:154.680 \
        --out     outputs/<song>/aligned.vad_midi.bound.json
"""
from __future__ import annotations

import json
from pathlib import Path

import click


def midi_note_onsets(midi_path: str) -> list[float]:
    """Note-on onsets (seconds) from a tempo-mapped MIDI, sorted ascending."""
    import mido

    mid = mido.MidiFile(midi_path)
    tempo = 500000
    secs = 0.0
    onsets: list[float] = []
    for msg in mido.merge_tracks(mid.tracks):
        secs += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
        if msg.type == "note_on" and msg.velocity > 0:
            onsets.append(secs)
    return sorted(onsets)


def sung_chars(line: dict) -> list[dict]:
    """Flat, in-order list of the line's non-punct chars that carry timing."""
    out: list[dict] = []
    for tok in line.get("tokens", []):
        if tok.get("is_punct"):
            continue
        for ch in tok.get("chars", []):
            if ch.get("start") is not None:
                out.append(ch)
    return out


def apply_override(
    line: dict, new_start: float, onsets: list[float], *, tol: float = 1e-3
) -> int:
    """Pin ``line`` to ``new_start`` and re-snap its leading mis-placed chars to
    consecutive MIDI onsets >= ``new_start``.  Returns the number of chars moved.

    A char is "mis-placed" if it currently starts before the onset it should
    occupy; the cascade stops at the first char that already has slack, so a
    correctly-timed line interior is never disturbed.
    """
    chars = sung_chars(line)
    if not chars:
        return 0
    onsets_ge = [o for o in onsets if o >= new_start - tol]
    if not onsets_ge:
        onsets_ge = [new_start]
    onsets_ge[0] = new_start  # honour the caller's exact onset for the first char

    moved = 0
    oi = 0
    for ch in chars:
        if oi >= len(onsets_ge):
            break
        target = onsets_ge[oi]
        if ch["start"] < target - tol:  # char sits earlier than its rightful onset
            ch["start"] = round(target, 3)
            moved += 1
            oi += 1
        else:
            break  # this char already has slack — interior is fine, stop

    if moved == 0:
        # first char was already at/after new_start; still pin the line anchor.
        chars[0]["start"] = round(new_start, 3)
        moved = 1

    # Re-lay contiguous ends across the disturbed prefix so each moved char wipes
    # up to the next char's (new) start — matches midi_timing's within-word layout
    # and guarantees monotone, non-zero spans.
    for idx in range(moved):
        nxt = chars[idx + 1]["start"] if idx + 1 < len(chars) else line.get("end")
        if nxt is not None and nxt > chars[idx]["start"]:
            chars[idx]["end"] = round(nxt, 3)

    # pull any leading chars (incl. punctuation like the open paren) that still
    # sit before the new start up to it, so the line span stays consistent.
    for tok in line.get("tokens", []):
        for ch in tok.get("chars", []):
            if ch.get("start") is not None and ch["start"] < new_start - tol:
                ch["start"] = round(new_start, 3)
                ch["end"] = round(max(ch["end"], new_start), 3)

    # propagate to the containing token start/end + line start
    line["start"] = round(new_start, 3)
    for tok in line.get("tokens", []):
        toks = [c for c in tok.get("chars", []) if c.get("start") is not None]
        if not toks:
            continue
        tok["start"] = toks[0]["start"]
        tok["end"] = max(c["end"] for c in toks)
    return moved


def parse_sets(set_specs: tuple[str, ...]) -> dict[int, float]:
    out: dict[int, float] = {}
    for spec in set_specs:
        idx_s, time_s = spec.split(":")
        out[int(idx_s)] = float(time_s)
    return out


@click.command()
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--midi", "midi_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "-o", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--set", "set_specs", multiple=True, required=True,
              help="LINE:TIME — pin line index LINE's start to TIME (s). Repeatable.")
def main(aligned_path: str, midi_path: str, out_path: str, set_specs: tuple[str, ...]) -> None:
    lines = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    onsets = midi_note_onsets(midi_path)
    overrides = parse_sets(set_specs)

    for idx, new_start in sorted(overrides.items()):
        if not (0 <= idx < len(lines)):
            raise click.ClickException(f"line index {idx} out of range (0..{len(lines)-1})")
        line = lines[idx]
        old = line["start"]
        moved = apply_override(line, new_start, onsets)
        print(f"  line {idx:2d}: start {old:7.3f} -> {new_start:7.3f}  ({moved} char(s) re-snapped)")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
