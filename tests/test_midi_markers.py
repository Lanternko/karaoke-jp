from __future__ import annotations

import json
import sys
from pathlib import Path

import mido
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from karaoke_jp.midi_markers import inject_line_markers


def _marker_times(mid: mido.MidiFile) -> list[tuple[str, float]]:
    tempo = 500_000
    out: list[tuple[str, float]] = []
    for msg in mid.tracks[0]:
        if msg.type == "set_tempo":
            tempo = msg.tempo
            break
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "marker":
                out.append((msg.text, mido.tick2second(abs_tick, mid.ticks_per_beat, tempo)))
    return out


def test_inject_line_markers_uses_lrc_block_size_and_clamps_overlap(tmp_path: Path) -> None:
    midi_path = tmp_path / "in.mid"
    out_path = tmp_path / "out.mid"
    aligned_path = tmp_path / "aligned.json"

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(track)
    mid.save(midi_path)

    aligned = [
        {"text": "L1", "start": 0.0, "end": 1.0, "tokens": [{"surface": "L1"}]},
        {"text": "L2", "start": 1.2, "end": 3.5, "tokens": [{"surface": "L2"}]},
        {"text": "L3", "start": 3.0, "end": 4.0, "tokens": [{"surface": "L3"}]},
        {"text": "L4", "start": 4.2, "end": 5.0, "tokens": [{"surface": "L4"}]},
    ]
    aligned_path.write_text(json.dumps(aligned), encoding="utf-8")

    count = inject_line_markers(midi_path, aligned_path, out_path, block_size=2)

    assert count == 2
    markers = _marker_times(mido.MidiFile(out_path))
    assert markers[0][0] == "P01"
    assert markers[0][1] == pytest.approx(0.0)
    assert markers[1][0] == "P02"
    assert markers[1][1] == pytest.approx(3.5)
    assert markers[2][0] == "END"
    assert markers[2][1] == pytest.approx(6.0)
