"""Snap MIDI note durations to musical units (8th / quarter / half note).

Singing transcription models (RMVPE / CECTC) emit note durations in
continuous seconds, which often look ragged on the karaoke display: a
quarter note that should last 0.50 s might come out 0.43 or 0.61 s.
Since each MIDI note in our pipeline is one sung syllable (see
``midi_timing.py``), and Japanese pop typically lands every syllable on
8th / quarter / half, snapping durations to a beat-derived grid produces
cleaner block widths without harming the note pitch (which is already
correct).

Algorithm:
1. Estimate global tempo via ``librosa.beat.beat_track`` on the
   instrumental track (clean → reliable). One global BPM is sufficient
   because the snap grid is tempo-relative.
2. For each MIDI note, compute current duration in beats and snap to the
   nearest of ``{0.5, 1.0, 2.0}`` beats.
3. If the snapped end overlaps the next note's onset, clamp to that onset.

Onsets are preserved verbatim — CECTC's onset estimation is its strong
suit, no need to disturb it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import mido
import numpy as np
import scipy.signal

# librosa 0.9.x calls scipy.signal.hann inside __trim_beats; scipy >= 1.13
# removed it in favor of scipy.signal.windows.hann. Restore the alias before
# librosa.beat.beat_track is called.
if not hasattr(scipy.signal, "hann"):
    scipy.signal.hann = scipy.signal.windows.hann  # type: ignore[attr-defined]

CANDIDATE_BEATS = (0.5, 1.0, 2.0)
DEFAULT_SR = 22050


def _read_notes(mid: mido.MidiFile) -> list[tuple[float, float, int, int]]:
    """Return ``(start_s, end_s, pitch, channel)`` per note, ignoring rests."""
    tempo_us = 500_000  # default 120 BPM if no set_tempo
    for tr in mid.tracks:
        for msg in tr:
            if msg.type == "set_tempo":
                tempo_us = msg.tempo
                break
        else:
            continue
        break

    notes: list[tuple[float, float, int, int]] = []
    for tr in mid.tracks:
        abs_tick = 0
        starts: dict[int, tuple[int, int]] = {}  # pitch -> (tick, channel)
        for msg in tr:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                starts[msg.note] = (abs_tick, getattr(msg, "channel", 0))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in starts:
                    start_tick, ch = starts.pop(msg.note)
                    s = mido.tick2second(start_tick, mid.ticks_per_beat, tempo_us)
                    e = mido.tick2second(abs_tick, mid.ticks_per_beat, tempo_us)
                    notes.append((s, e, msg.note, ch))
    notes.sort(key=lambda n: n[0])
    return notes


def _write_notes(
    notes: list[tuple[float, float, int, int]],
    midi_path: Path,
    *,
    tempo: float = 120.0,
    ticks_per_beat: int = 480,
) -> None:
    tempo_us = mido.bpm2tempo(tempo)
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    meta.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta)

    track = mido.MidiTrack()
    prev_tick = 0
    for s, e, pitch, ch in notes:
        start_tick = int(round(mido.second2tick(s, ticks_per_beat, tempo_us)))
        end_tick = int(round(mido.second2tick(e, ticks_per_beat, tempo_us)))
        end_tick = max(start_tick + 1, end_tick)
        track.append(
            mido.Message(
                "note_on", note=int(pitch), velocity=100,
                time=max(0, start_tick - prev_tick), channel=int(ch),
            )
        )
        track.append(
            mido.Message(
                "note_off", note=int(pitch), velocity=0,
                time=max(1, end_tick - start_tick), channel=int(ch),
            )
        )
        prev_tick = end_tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(track)

    midi_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(midi_path)


def _estimate_bpm(audio_path: Path, *, sr: int = DEFAULT_SR) -> float:
    y, sr_loaded = librosa.load(str(audio_path), sr=sr, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr_loaded)
    bpm = float(np.atleast_1d(tempo)[0])
    if not (40.0 <= bpm <= 240.0):
        raise ValueError(f"Implausible BPM estimate: {bpm:.2f}")
    return bpm


def quantize_durations(
    in_midi: Path,
    out_midi: Path,
    instrumental: Path,
    *,
    sr: int = DEFAULT_SR,
    bpm_override: float | None = None,
) -> dict:
    bpm = bpm_override if bpm_override is not None else _estimate_bpm(instrumental, sr=sr)
    sec_per_beat = 60.0 / bpm
    grid = [c * sec_per_beat for c in CANDIDATE_BEATS]

    mid = mido.MidiFile(str(in_midi))
    notes = _read_notes(mid)
    if not notes:
        raise ValueError(f"No notes in {in_midi}")

    quantized: list[tuple[float, float, int, int]] = []
    snap_counts = {c: 0 for c in CANDIDATE_BEATS}
    for idx, (s, e, pitch, ch) in enumerate(notes):
        cur_dur = e - s
        if cur_dur <= 0:
            continue
        # Pick beat candidate whose target seconds is closest to current dur.
        best_c, _ = min(
            zip(CANDIDATE_BEATS, grid),
            key=lambda kv: abs(kv[1] - cur_dur),
        )
        new_end = s + best_c * sec_per_beat
        # Clamp to next-onset to avoid overlap.
        if idx + 1 < len(notes):
            next_start = notes[idx + 1][0]
            if new_end > next_start:
                new_end = next_start
        quantized.append((s, max(s + 1e-3, new_end), pitch, ch))
        snap_counts[best_c] += 1

    _write_notes(quantized, out_midi)
    return {
        "bpm": bpm,
        "in_notes": len(notes),
        "out_notes": len(quantized),
        "snap_counts": snap_counts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True, type=Path)
    ap.add_argument("--instrumental", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bpm", type=float, default=None,
                    help="Override the BPM estimate (skip librosa beat track).")
    args = ap.parse_args()

    stats = quantize_durations(
        args.midi, args.out, args.instrumental, bpm_override=args.bpm,
    )
    # Sidecar: <out>.bpm.txt holds the float BPM for downstream consumers
    # (midi_markers fixed-grid mode, render_mp4 scale annotation).
    bpm_sidecar = args.out.with_suffix(args.out.suffix + ".bpm.txt")
    bpm_sidecar.write_text(f"{stats['bpm']:.4f}\n")
    print(
        f"[quantize] bpm={stats['bpm']:.2f} notes={stats['in_notes']}->{stats['out_notes']} "
        f"snap_counts={stats['snap_counts']} -> {args.out}; bpm -> {bpm_sidecar}"
    )


if __name__ == "__main__":
    main()
