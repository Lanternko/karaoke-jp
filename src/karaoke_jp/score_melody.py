"""Score-driven melody extraction for piano-only recordings.

This path is meant for the "I have the score already" case:

1. Read a score MIDI file.
2. Align the score against a piano recording with chroma DTW.
3. Emit the score's top voice with timings warped onto the audio.

Pitch therefore comes from the score, not from audio F0 estimation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np

from .melody import _write_midi


@dataclass(frozen=True)
class MidiNote:
    start: float
    end: float
    pitch: int


def _require_librosa():
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - exercised through CLI
        raise RuntimeError(
            "score-melody requires librosa. Install it with "
            "`~/venvs/karaoke-jp/bin/pip install -e '.[score]'`."
        ) from exc
    return librosa


def read_midi_notes(midi_path: str | Path) -> list[MidiNote]:
    """Read note events from a MIDI file with tempo-aware absolute seconds."""
    midi_path = Path(midi_path).resolve()
    mid = mido.MidiFile(midi_path)

    tempo = 500000  # default: 120 BPM
    current_seconds = 0.0
    active: dict[int, list[float]] = defaultdict(list)
    notes: list[MidiNote] = []

    for msg in mido.merge_tracks(mid.tracks):
        current_seconds += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note].append(current_seconds)
            continue
        if msg.type not in {"note_off", "note_on"}:
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            continue
        if msg.note not in active or not active[msg.note]:
            continue
        start = active[msg.note].pop(0)
        if not active[msg.note]:
            del active[msg.note]
        end = max(current_seconds, start + 1e-3)
        notes.append(MidiNote(start=start, end=end, pitch=int(msg.note)))

    notes.sort(key=lambda note: (note.start, note.end, note.pitch))
    return notes


def read_first_tempo_bpm(midi_path: str | Path) -> float:
    """Return the first MIDI tempo marker, or 120 BPM if none exists."""
    midi_path = Path(midi_path).resolve()
    mid = mido.MidiFile(midi_path)
    for msg in mido.merge_tracks(mid.tracks):
        if msg.type == "set_tempo":
            return float(mido.tempo2bpm(msg.tempo))
    return 120.0


def extract_top_voice_notes(
    notes: list[MidiNote],
    *,
    onset_tolerance: float = 1e-4,
) -> list[MidiNote]:
    """Select the highest note for each onset cluster.

    This works better for piano scores than a continuous "skyline" pass because
    long accompaniment tails should not suddenly become melody after the real
    top note releases.
    """
    if not notes:
        return []

    grouped: list[list[MidiNote]] = []
    current_group: list[MidiNote] = [notes[0]]

    for note in notes[1:]:
        if abs(note.start - current_group[0].start) <= onset_tolerance:
            current_group.append(note)
            continue
        grouped.append(current_group)
        current_group = [note]
    grouped.append(current_group)

    selected = [
        max(group, key=lambda note: (note.pitch, note.end - note.start))
        for group in grouped
    ]
    return merge_adjacent_same_pitch(selected, tolerance=onset_tolerance)


def merge_adjacent_same_pitch(
    notes: list[MidiNote],
    *,
    tolerance: float = 1e-4,
) -> list[MidiNote]:
    """Merge back-to-back same-pitch notes, which commonly represent ties."""
    if not notes:
        return []

    merged = [notes[0]]
    for note in notes[1:]:
        prev = merged[-1]
        if note.pitch == prev.pitch and note.start <= prev.end + tolerance:
            merged[-1] = MidiNote(prev.start, max(prev.end, note.end), prev.pitch)
            continue
        merged.append(note)
    return merged


def build_score_chroma(
    notes: list[MidiNote],
    *,
    hop_seconds: float,
    origin_seconds: float | None = None,
) -> tuple[np.ndarray, float]:
    """Convert score notes into a framewise 12-bin chroma matrix."""
    if not notes:
        raise ValueError("Cannot build chroma from an empty note list.")
    if hop_seconds <= 0:
        raise ValueError(f"hop_seconds must be positive, got {hop_seconds}")

    origin = min(note.start for note in notes) if origin_seconds is None else origin_seconds
    duration = max(note.end for note in notes) - origin
    n_frames = max(1, int(np.ceil(duration / hop_seconds)) + 1)
    chroma = np.zeros((12, n_frames), dtype=np.float32)

    for note in notes:
        start_frame = max(0, int(np.floor((note.start - origin) / hop_seconds)))
        end_frame = max(start_frame + 1, int(np.ceil((note.end - origin) / hop_seconds)))
        chroma[note.pitch % 12, start_frame:end_frame] += 1.0

    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    nonzero = norms[0] > 0
    chroma[:, nonzero] /= norms[:, nonzero]
    return chroma, origin


def compute_frame_alignment(
    score_chroma: np.ndarray,
    audio_chroma: np.ndarray,
) -> np.ndarray:
    """Return a monotone mapping: score-frame index -> audio-frame index."""
    if score_chroma.ndim != 2 or audio_chroma.ndim != 2:
        raise ValueError("score_chroma and audio_chroma must both be 2-D.")
    if score_chroma.shape[0] != 12 or audio_chroma.shape[0] != 12:
        raise ValueError("Expected 12-bin chroma features.")

    librosa = _require_librosa()
    _, path = librosa.sequence.dtw(X=score_chroma, Y=audio_chroma, metric="euclidean")
    path = path[::-1]

    grouped: dict[int, list[int]] = defaultdict(list)
    for score_frame, audio_frame in path:
        grouped[int(score_frame)].append(int(audio_frame))

    score_frames = np.array(sorted(grouped), dtype=np.float64)
    audio_frames = np.array(
        [float(np.median(grouped[idx])) for idx in score_frames.astype(int)],
        dtype=np.float64,
    )
    full_score_frames = np.arange(score_chroma.shape[1], dtype=np.float64)
    return np.interp(full_score_frames, score_frames, audio_frames)


def map_notes_to_audio(
    notes: list[MidiNote],
    *,
    score_origin: float,
    frame_map: np.ndarray,
    score_hop_seconds: float,
    audio_hop_seconds: float,
    audio_offset_seconds: float,
) -> list[MidiNote]:
    """Warp score-note times into audio-note times using the DTW frame map."""
    if frame_map.ndim != 1 or frame_map.size == 0:
        raise ValueError("frame_map must be a non-empty 1-D array.")

    frame_positions = np.arange(frame_map.size, dtype=np.float64)

    def map_time(score_time: float) -> float:
        relative = max(score_time - score_origin, 0.0)
        score_frame = relative / score_hop_seconds
        audio_frame = np.interp(score_frame, frame_positions, frame_map)
        return audio_offset_seconds + audio_frame * audio_hop_seconds

    aligned: list[MidiNote] = []
    for note in notes:
        start = map_time(note.start)
        end = max(map_time(note.end), start + 1e-3)
        aligned.append(MidiNote(start=start, end=end, pitch=note.pitch))
    return aligned


def extract_score_aligned_melody(
    audio_path: str | Path,
    score_midi_path: str | Path,
    midi_path: str | Path,
    *,
    top_voice: bool = True,
    sample_rate: int = 22050,
    hop_length: int = 1024,
    trim_top_db: float = 40.0,
    tempo: float | None = None,
) -> Path:
    """Align a score MIDI to piano audio and emit melody MIDI."""
    audio_path = Path(audio_path).resolve()
    score_midi_path = Path(score_midi_path).resolve()
    midi_path = Path(midi_path).resolve()

    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    if not score_midi_path.is_file():
        raise FileNotFoundError(score_midi_path)
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if hop_length <= 0:
        raise ValueError(f"hop_length must be positive, got {hop_length}")

    score_notes = read_midi_notes(score_midi_path)
    if not score_notes:
        raise ValueError(f"No notes found in score MIDI: {score_midi_path}")

    melody_notes = extract_top_voice_notes(score_notes) if top_voice else score_notes

    librosa = _require_librosa()
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    trimmed, trim_index = librosa.effects.trim(audio, top_db=trim_top_db)
    if trimmed.size == 0:
        raise ValueError(f"Audio became empty after trim: {audio_path}")

    harmonic = librosa.effects.harmonic(trimmed)
    audio_chroma = librosa.feature.chroma_cens(
        y=harmonic,
        sr=sr,
        hop_length=hop_length,
    )

    hop_seconds = hop_length / sr
    score_chroma, score_origin = build_score_chroma(
        score_notes,
        hop_seconds=hop_seconds,
    )
    frame_map = compute_frame_alignment(score_chroma, audio_chroma)
    aligned_notes = map_notes_to_audio(
        melody_notes,
        score_origin=score_origin,
        frame_map=frame_map,
        score_hop_seconds=hop_seconds,
        audio_hop_seconds=hop_seconds,
        audio_offset_seconds=trim_index[0] / sr,
    )

    export_tempo = tempo if tempo is not None else read_first_tempo_bpm(score_midi_path)
    midi_notes = [(note.start, note.end, note.pitch) for note in aligned_notes]
    _write_midi(midi_notes, midi_path, tempo=export_tempo)
    return midi_path
