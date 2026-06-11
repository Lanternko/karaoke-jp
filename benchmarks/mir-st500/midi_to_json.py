#!/usr/bin/env python3
"""Collect GAME-output MIDI files into MIR-ST500 prediction JSON.

Input: a directory of <song_id>.mid files (GAME `extract` output).
Output: {song_id: [[onset_sec, offset_sec, midi_pitch_float], ...]}
matching the format the official evaluate.py consumes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mido


def midi_notes(path: Path) -> list[list[float]]:
    mid = mido.MidiFile(path)
    tempo = 500000
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
    notes = []
    for tr in mid.tracks:
        t = 0.0
        ons: dict[int, float] = {}
        for msg in tr:
            t += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            if msg.type == "note_on" and msg.velocity:
                ons[msg.note] = t
            elif msg.type == "note_off" or (msg.type == "note_on" and not msg.velocity):
                if msg.note in ons:
                    notes.append([ons.pop(msg.note), t, float(msg.note)])
    notes.sort()
    return notes


def main() -> None:
    mid_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    pred = {}
    for f in sorted(mid_dir.glob("*.mid")):
        pred[f.stem] = midi_notes(f)
    out_path.write_text(json.dumps(pred))
    total = sum(len(v) for v in pred.values())
    print(f"[midi-to-json] {len(pred)} songs, {total} notes -> {out_path}")


if __name__ == "__main__":
    main()
