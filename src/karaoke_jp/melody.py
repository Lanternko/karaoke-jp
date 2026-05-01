"""M2: vocals.wav -> melody.mid.

Default path: RMVPE F0 extraction from ``third_party/SOME/modules/rmvpe``
running inside the dedicated ``~/venvs/karaoke-jp-melody/`` interpreter, then
local note segmentation to MIDI. We keep the original SOME CLI inference as a
fallback backend for A/B comparison.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOME_DIR = PROJECT_ROOT / "third_party" / "SOME"
DEFAULT_SOME_CKPT = (
    DEFAULT_SOME_DIR / "pretrained" / "0119_continuous256_5spk" /
    "model_ckpt_steps_100000_simplified.ckpt"
)
DEFAULT_RMVPE_CKPT = DEFAULT_SOME_DIR / "pretrained" / "rmvpe" / "model.pt"
DEFAULT_SOME_PYTHON = Path.home() / "venvs" / "karaoke-jp-melody" / "bin" / "python"
DEFAULT_RMVPE_SCRIPT = PROJECT_ROOT / "scripts" / "extract_rmvpe_f0.py"

DEFAULT_CECTC_DIR = PROJECT_ROOT / "third_party" / "CTC_CE_for_AST"
DEFAULT_CECTC_CKPT = (
    DEFAULT_CECTC_DIR / "pretrained" / "CTC_CE_for_AST" / "ctc_ce#3_98" / "ctc_ce#3_98"
)
DEFAULT_CECTC_YAML = (
    DEFAULT_CECTC_DIR / "pretrained" / "CTC_CE_for_AST" / "ctc_ce#3_98" /
    "inference_ce_ctc#3_98.yaml"
)
DEFAULT_CECTC_SCRIPT = PROJECT_ROOT / "scripts" / "run_cectc_inference.py"


@dataclass(frozen=True)
class Run:
    value: int
    start: int
    end: int

    @property
    def is_rest(self) -> bool:
        return self.value < 0

    @property
    def duration_frames(self) -> int:
        return self.end - self.start


def _build_child_env(cuda_device: int | None) -> dict[str, str]:
    """Build a minimal child env so the parent venv does not leak into SOME."""
    parent = os.environ
    passthrough = (
        "PATH", "HOME", "USER", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE",
        "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH",
    )
    env = {k: parent[k] for k in passthrough if k in parent}
    env["CUDA_VISIBLE_DEVICES"] = "" if cuda_device is None else str(cuda_device)
    return env


def _seconds_to_ticks(seconds: float, ticks_per_beat: int, tempo_us: int) -> int:
    beats = seconds * 1_000_000 / tempo_us
    return int(round(beats * ticks_per_beat))


def _write_midi(
    notes: list[tuple[float, float, int]],
    midi_path: Path,
    *,
    tempo: float,
    ticks_per_beat: int = 480,
) -> None:
    tempo_us = mido.bpm2tempo(tempo)
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    meta_track = mido.MidiTrack()
    meta_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    meta_track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=4,
            denominator=4,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0,
        )
    )
    meta_track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(meta_track)

    note_track = mido.MidiTrack()
    prev_tick = 0
    for start_s, end_s, pitch in notes:
        abs_start = _seconds_to_ticks(start_s, ticks_per_beat, tempo_us)
        abs_end = max(abs_start + 1, _seconds_to_ticks(end_s, ticks_per_beat, tempo_us))
        note_track.append(
            mido.Message("note_on", note=int(pitch), velocity=100, time=max(abs_start - prev_tick, 0))
        )
        note_track.append(
            mido.Message("note_off", note=int(pitch), velocity=0, time=max(abs_end - abs_start, 1))
        )
        prev_tick = abs_end
    note_track.append(mido.MetaMessage("end_of_track", time=0))
    mid.tracks.append(note_track)

    midi_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(midi_path)


def _hz_to_midi(f0_hz: np.ndarray) -> np.ndarray:
    midi = np.full(f0_hz.shape, np.nan, dtype=np.float64)
    voiced = f0_hz > 0
    midi[voiced] = 69.0 + 12.0 * np.log2(f0_hz[voiced] / 440.0)
    return midi


def _bridge_short_gaps(
    midi_track: np.ndarray,
    *,
    max_gap_frames: int,
    pitch_tolerance: float,
) -> np.ndarray:
    bridged = midi_track.copy()
    i = 0
    n = len(bridged)
    while i < n:
        if np.isfinite(bridged[i]):
            i += 1
            continue
        gap_start = i
        while i < n and not np.isfinite(bridged[i]):
            i += 1
        gap_end = i
        gap_len = gap_end - gap_start
        if gap_start == 0 or gap_end >= n or gap_len > max_gap_frames:
            continue
        left = bridged[gap_start - 1]
        right = bridged[gap_end]
        if not np.isfinite(left) or not np.isfinite(right):
            continue
        if abs(left - right) > pitch_tolerance:
            continue
        bridged[gap_start:gap_end] = np.linspace(left, right, gap_len + 2)[1:-1]
    return bridged


def _median_smooth(midi_track: np.ndarray, *, window: int) -> np.ndarray:
    if window <= 1:
        return midi_track.copy()
    radius = window // 2
    smoothed = midi_track.copy()
    for idx in np.where(np.isfinite(midi_track))[0]:
        lo = max(0, idx - radius)
        hi = min(len(midi_track), idx + radius + 1)
        segment = midi_track[lo:hi]
        segment = segment[np.isfinite(segment)]
        if segment.size:
            smoothed[idx] = float(np.median(segment))
    return smoothed


def _build_runs(quantized_track: np.ndarray) -> list[Run]:
    if quantized_track.size == 0:
        return []
    runs: list[Run] = []
    run_start = 0
    current = int(quantized_track[0])
    for idx in range(1, len(quantized_track)):
        value = int(quantized_track[idx])
        if value != current:
            runs.append(Run(current, run_start, idx))
            run_start = idx
            current = value
    runs.append(Run(current, run_start, len(quantized_track)))
    return runs


def _coalesce_runs(runs: list[Run]) -> list[Run]:
    if not runs:
        return []
    merged = [runs[0]]
    for run in runs[1:]:
        prev = merged[-1]
        if prev.value == run.value:
            merged[-1] = Run(prev.value, prev.start, run.end)
        else:
            merged.append(run)
    return merged


def _merge_short_runs(
    runs: list[Run],
    *,
    min_note_frames: int,
    max_gap_frames: int,
) -> list[Run]:
    runs = _coalesce_runs(runs)
    changed = True
    while changed:
        changed = False
        merged: list[Run] = []
        i = 0
        while i < len(runs):
            run = runs[i]

            if (
                run.is_rest
                and run.duration_frames <= max_gap_frames
                and merged
                and i + 1 < len(runs)
                and not merged[-1].is_rest
                and not runs[i + 1].is_rest
                and merged[-1].value == runs[i + 1].value
            ):
                prev = merged.pop()
                merged.append(Run(prev.value, prev.start, runs[i + 1].end))
                i += 2
                changed = True
                continue

            if not run.is_rest and run.duration_frames < min_note_frames:
                prev_note = merged[-1] if merged and not merged[-1].is_rest else None
                next_note = runs[i + 1] if i + 1 < len(runs) and not runs[i + 1].is_rest else None

                if prev_note and next_note and prev_note.value == next_note.value:
                    prev = merged.pop()
                    merged.append(Run(prev.value, prev.start, next_note.end))
                    i += 2
                    changed = True
                    continue

                if prev_note and prev_note.duration_frames >= min_note_frames and abs(prev_note.value - run.value) <= 1:
                    prev = merged.pop()
                    merged.append(Run(prev.value, prev.start, run.end))
                    i += 1
                    changed = True
                    continue

                if next_note and next_note.duration_frames >= min_note_frames:
                    merged.append(Run(next_note.value, run.start, next_note.end))
                    i += 2
                    changed = True
                    continue

                merged.append(Run(-1, run.start, run.end))
                i += 1
                changed = True
                continue

            merged.append(run)
            i += 1

        runs = _coalesce_runs(merged)
    return runs


def segment_f0_to_notes(
    f0_hz: np.ndarray,
    *,
    hop_seconds: float,
    min_note_duration: float = 0.08,
    max_gap_duration: float = 0.05,
    pitch_tolerance: float = 0.75,
    smoothing_window: int = 5,
) -> list[tuple[float, float, int]]:
    """Convert an F0 track into discrete MIDI notes."""
    if hop_seconds <= 0:
        raise ValueError(f"hop_seconds must be positive, got {hop_seconds}")
    if f0_hz.ndim != 1:
        raise ValueError(f"f0_hz must be 1-D, got shape {f0_hz.shape}")

    midi_track = _hz_to_midi(f0_hz.astype(np.float64, copy=False))
    max_gap_frames = max(1, int(round(max_gap_duration / hop_seconds)))
    min_note_frames = max(1, int(round(min_note_duration / hop_seconds)))

    midi_track = _bridge_short_gaps(
        midi_track,
        max_gap_frames=max_gap_frames,
        pitch_tolerance=pitch_tolerance,
    )
    midi_track = _median_smooth(midi_track, window=smoothing_window)

    quantized = np.full(midi_track.shape, -1, dtype=np.int16)
    voiced = np.isfinite(midi_track)
    quantized[voiced] = np.rint(midi_track[voiced]).astype(np.int16)

    runs = _merge_short_runs(
        _build_runs(quantized),
        min_note_frames=min_note_frames,
        max_gap_frames=max_gap_frames,
    )

    notes: list[tuple[float, float, int]] = []
    for run in runs:
        if run.is_rest or run.duration_frames < min_note_frames:
            continue
        start_s = run.start * hop_seconds
        end_s = run.end * hop_seconds
        notes.append((start_s, end_s, int(run.value)))
    return notes


def _duration_weighted_median(pitches: list[int], durations: list[float]) -> float:
    total = sum(durations)
    if total == 0:
        return float(np.median(pitches))
    cumsum = 0.0
    for pitch, dur in sorted(zip(pitches, durations)):
        cumsum += dur
        if cumsum >= total / 2:
            return float(pitch)
    return float(pitches[-1])


def fix_phrase_octave(
    notes: list[tuple[float, float, int]],
    *,
    jump_min: int = 10,
    jump_max: int = 15,
    tolerance: int = 4,
    max_passes: int = 5,
) -> list[tuple[float, float, int]]:
    """Correct phrase-level octave tracking errors using a song-wide anchor.

    RMVPE occasionally locks onto the sub-harmonic for entire phrases (not
    just isolated notes), producing runs of F3/G3/A#3 where the correct
    pitches are F4/G4/A#4.  A local-window median cannot fix this because
    the neighbourhood itself is in the wrong octave.

    Instead we compute a duration-weighted median over the whole note list as
    the ``anchor_pitch``.  Any note whose pitch is ``jump_min``–``jump_max``+2
    semitones away from the anchor, and whose octave-shifted pitch lands within
    ``tolerance`` semitones of the anchor, is shifted by ±12.

    After each pass the anchor is recomputed from the updated notes.  This
    handles the common case where a large sub-harmonic cluster dragged the
    initial anchor down, hiding a second tier of slightly-elevated errors that
    only become detectable once the first tier is corrected.  Iteration stops
    when no note changes or ``max_passes`` is reached.
    """
    if not notes:
        return []
    current = list(notes)
    for _ in range(max_passes):
        durations = [e - s for s, e, _ in current]
        pitches = [p for _, _, p in current]
        anchor = _duration_weighted_median(pitches, durations)

        nxt = []
        changed = False
        for s, e, p in current:
            diff_below = anchor - p
            if jump_min <= diff_below <= jump_max + 2 and abs((p + 12) - anchor) <= tolerance:
                nxt.append((s, e, p + 12))
                changed = True
            else:
                nxt.append((s, e, p))
        current = nxt
        if not changed:
            break
    return current


def fix_octave_errors(
    notes: list[tuple[float, float, int]],
    *,
    jump_min: int = 10,
    jump_max: int = 15,
    window: int = 5,
    tolerance: int = 4,
) -> list[tuple[float, float, int]]:
    """Raise notes that RMVPE tracked one octave too low.

    For each note whose pitch is ``jump_min``–``jump_max`` semitones below
    the median of the ``window`` nearest neighbours, try shifting it up by
    12.  The shift is accepted only if the result lands within ``tolerance``
    semitones of that median — so legitimate low notes (e.g. a real chest-
    voice phrase surrounded by head-voice neighbours) are not touched.

    This corrects the systematic sub-harmonic tracking error that shows up as
    isolated D#3/E3 notes in an otherwise D#4 melody.
    """
    if len(notes) < 3:
        return list(notes)
    fixed = list(notes)
    n = len(fixed)
    for i in range(n):
        start_s, end_s, pitch = fixed[i]
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neighbours = [fixed[j][2] for j in range(lo, hi) if j != i]
        if not neighbours:
            continue
        median_pitch = int(np.median(neighbours))
        diff = median_pitch - pitch
        if jump_min <= diff <= jump_max + 2:
            candidate = pitch + 12
            if abs(candidate - median_pitch) <= tolerance:
                fixed[i] = (start_s, end_s, candidate)
    return fixed


def fill_held_note_gaps(
    notes: list[tuple[float, float, int]],
    *,
    gap_threshold: float = 8.0,
    pitch_tolerance: int = 0,
) -> list[tuple[float, float, int]]:
    """Fill long silences that are clearly held notes RMVPE dropped.

    RMVPE loses tracking on very long sustained notes (>8 s), leaving a gap
    even though the singer is still vocalising.  When the note on both sides
    of a gap is the same pitch (within ``pitch_tolerance`` semitones), we
    insert a bridging note to cover the silent span.

    Conservative defaults (pitch_tolerance=0) mean only exact-match pairs are
    filled, avoiding false insertions during real instrumental breaks where the
    pitches before/after the rest are usually different.
    """
    if len(notes) < 2:
        return list(notes)
    result = list(notes)
    inserts: list[tuple[float, float, int]] = []
    for i in range(len(notes) - 1):
        gap = notes[i + 1][0] - notes[i][1]
        if gap < gap_threshold:
            continue
        prev_p = notes[i][2]
        next_p = notes[i + 1][2]
        if abs(prev_p - next_p) <= pitch_tolerance:
            fill_p = round((prev_p + next_p) / 2)
            inserts.append((notes[i][1], notes[i + 1][0], fill_p))
    result.extend(inserts)
    result.sort(key=lambda n: n[0])
    return result


def _extract_midi_with_some(
    vocals_path: Path,
    midi_path: Path,
    *,
    tempo: float,
    some_dir: Path,
    some_ckpt: Path,
    some_python: Path,
    cuda_device: int | None,
) -> Path:
    if not some_ckpt.exists():
        raise FileNotFoundError(
            f"SOME checkpoint not found at {some_ckpt}. Run the M2 setup."
        )
    cmd = [
        str(some_python),
        "infer.py",
        "--model",
        str(some_ckpt),
        "--wav",
        str(vocals_path),
        "--midi",
        str(midi_path),
        "--tempo",
        str(tempo),
    ]
    subprocess.run(cmd, cwd=some_dir, env=_build_child_env(cuda_device), check=True)
    return midi_path


def _infer_rmvpe_f0(
    vocals_path: Path,
    *,
    some_python: Path,
    some_dir: Path,
    rmvpe_ckpt: Path,
    cuda_device: int | None,
) -> tuple[np.ndarray, float]:
    if not rmvpe_ckpt.exists():
        raise FileNotFoundError(
            f"RMVPE checkpoint not found at {rmvpe_ckpt}. "
            "Expected third_party/SOME/pretrained/rmvpe/model.pt."
        )
    if not DEFAULT_RMVPE_SCRIPT.exists():
        raise FileNotFoundError(DEFAULT_RMVPE_SCRIPT)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        device = "cpu" if cuda_device is None else f"cuda:{cuda_device}"
        cmd = [
            str(some_python),
            str(DEFAULT_RMVPE_SCRIPT),
            "--wav",
            str(vocals_path),
            "--model",
            str(rmvpe_ckpt),
            "--out",
            str(tmp_path),
            "--device",
            device,
        ]
        subprocess.run(cmd, cwd=some_dir, env=_build_child_env(cuda_device), check=True)
        with np.load(tmp_path) as data:
            f0 = data["f0"].astype(np.float32)
            hop_seconds = float(data["hop_seconds"][0])
        return f0, hop_seconds
    finally:
        tmp_path.unlink(missing_ok=True)


def _extract_midi_with_cectc(
    vocals_path: Path,
    instrumental_path: Path,
    midi_path: Path,
    *,
    cectc_python: Path,
    cectc_script: Path,
    cectc_ckpt: Path,
    cectc_yaml: Path,
    cuda_device: int | None,
) -> Path:
    if not cectc_ckpt.exists():
        raise FileNotFoundError(
            f"CTC+CE checkpoint not found at {cectc_ckpt}. "
            "Run: gdown --folder https://drive.google.com/drive/folders/"
            "1lxq-IF83cEXE8XsTFywNJhwtDSRXWqRx (ctc_ce#3_98 subdir)."
        )
    if not cectc_yaml.exists():
        raise FileNotFoundError(cectc_yaml)
    if not cectc_script.exists():
        raise FileNotFoundError(cectc_script)
    device = "cpu" if cuda_device is None else f"cuda:{cuda_device}"
    cmd = [
        str(cectc_python),
        str(cectc_script),
        "--vocals", str(vocals_path),
        "--instrumental", str(instrumental_path),
        "--midi", str(midi_path),
        "--model", str(cectc_ckpt),
        "--yaml", str(cectc_yaml),
        "--device", device,
    ]
    subprocess.run(cmd, env=_build_child_env(cuda_device), check=True)
    return midi_path


def extract_midi(
    vocals_path: str | Path,
    midi_path: str | Path,
    *,
    tempo: float = 120.0,
    backend: str = "rmvpe",
    instrumental_path: str | Path | None = None,
    some_dir: Path | None = None,
    some_ckpt: Path | None = None,
    rmvpe_ckpt: Path | None = None,
    some_python: Path | None = None,
    cectc_ckpt: Path | None = None,
    cectc_yaml: Path | None = None,
    cectc_script: Path | None = None,
    cuda_device: int | None = 0,
    min_note_duration: float = 0.08,
    max_gap_duration: float = 0.05,
    pitch_tolerance: float = 0.75,
) -> Path:
    """Extract a melody MIDI from ``vocals_path``.

    ``backend="cectc"`` runs Wang & Jang's CTC+CE direct note transcription
    (TASLP 2023). It additionally requires ``instrumental_path`` because the
    model's input is the 6-channel CQT of (vocal, vocal+instrumental).
    """
    vocals_path = Path(vocals_path).resolve()
    if not vocals_path.is_file():
        raise FileNotFoundError(vocals_path)

    midi_path = Path(midi_path).resolve()
    midi_path.parent.mkdir(parents=True, exist_ok=True)

    some_dir = some_dir or DEFAULT_SOME_DIR
    some_ckpt = some_ckpt or DEFAULT_SOME_CKPT
    rmvpe_ckpt = rmvpe_ckpt or DEFAULT_RMVPE_CKPT
    some_python = some_python or DEFAULT_SOME_PYTHON

    if not some_python.exists():
        raise FileNotFoundError(
            f"Melody python interpreter not found at {some_python}. "
            "Set up ~/venvs/karaoke-jp-melody (see README)."
        )

    if backend == "cectc":
        if instrumental_path is None:
            raise ValueError(
                "backend='cectc' requires instrumental_path "
                "(model needs vocal+instrumental CQT pair)."
            )
        instrumental_path = Path(instrumental_path).resolve()
        if not instrumental_path.is_file():
            raise FileNotFoundError(instrumental_path)
        return _extract_midi_with_cectc(
            vocals_path,
            instrumental_path,
            midi_path,
            cectc_python=some_python,
            cectc_script=cectc_script or DEFAULT_CECTC_SCRIPT,
            cectc_ckpt=cectc_ckpt or DEFAULT_CECTC_CKPT,
            cectc_yaml=cectc_yaml or DEFAULT_CECTC_YAML,
            cuda_device=cuda_device,
        )

    if backend == "some":
        return _extract_midi_with_some(
            vocals_path,
            midi_path,
            tempo=tempo,
            some_dir=some_dir,
            some_ckpt=some_ckpt,
            some_python=some_python,
            cuda_device=cuda_device,
        )
    if backend != "rmvpe":
        raise ValueError(f"Unsupported melody backend: {backend}")

    f0_hz, hop_seconds = _infer_rmvpe_f0(
        vocals_path,
        some_python=some_python,
        some_dir=some_dir,
        rmvpe_ckpt=rmvpe_ckpt,
        cuda_device=cuda_device,
    )
    notes = segment_f0_to_notes(
        f0_hz,
        hop_seconds=hop_seconds,
        min_note_duration=min_note_duration,
        max_gap_duration=max_gap_duration,
        pitch_tolerance=pitch_tolerance,
    )
    notes = fix_phrase_octave(notes)
    notes = fix_octave_errors(notes)
    notes = fill_held_note_gaps(notes)
    _write_midi(notes, midi_path, tempo=tempo)
    return midi_path
