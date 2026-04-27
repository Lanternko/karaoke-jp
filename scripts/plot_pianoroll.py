"""Render a piano-roll PNG for visual sanity-checking of an extracted MIDI.

Usage:
    python scripts/plot_pianoroll.py outputs/<song>/melody.mid out.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido


def collect_notes(midi_path: Path) -> list[tuple[float, float, int]]:
    mid = mido.MidiFile(midi_path)
    tempo = 500_000  # default 120 bpm in microseconds per beat
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo = msg.tempo
            break

    sec_per_tick = tempo / 1_000_000 / mid.ticks_per_beat
    notes = []
    pending: dict[int, float] = {}
    abs_tick = 0
    for msg in mid.tracks[0]:
        abs_tick += msg.time
        t = abs_tick * sec_per_tick
        if msg.type == "note_on" and msg.velocity > 0:
            pending[msg.note] = t
        elif msg.type in {"note_off", "note_on"} and msg.note in pending:
            start = pending.pop(msg.note)
            notes.append((start, t, msg.note))
    return notes


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: plot_pianoroll.py <melody.mid> <out.png>", file=sys.stderr)
        sys.exit(2)
    midi_path = Path(sys.argv[1])
    out_png = Path(sys.argv[2])

    notes = collect_notes(midi_path)
    if not notes:
        print("no notes parsed", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(20, 6))
    for start, end, pitch in notes:
        ax.add_patch(
            plt.Rectangle(
                (start, pitch - 0.4),
                max(end - start, 0.05),
                0.8,
                facecolor="#3478f6",
                edgecolor="#1f3f7a",
                linewidth=0.4,
            )
        )

    pitches = [p for _, _, p in notes]
    ends = [e for _, e, _ in notes]
    ax.set_xlim(0, max(ends) + 1)
    ax.set_ylim(min(pitches) - 2, max(pitches) + 2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MIDI pitch")
    ax.set_title(
        f"{midi_path.name} — {len(notes)} notes, pitch {min(pitches)}–{max(pitches)}"
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
