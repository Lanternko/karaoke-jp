"""Sidecar line-end repair: extend early line endings along the vocal tail.

RMS-VAD pre-segmentation fixes the *gross* segmentation collapse (zero-duration
lines, mega-segments), but the gold-label eval on haru-hikage line 20-31 shows a
residual **systematic early cutoff**: line ends land ~0.5-1.9 s before the true
vocal offset (mean_bias -0.573 s, P90 1.601 s).  Cause: ``midi_timing`` pins the
last char to the last quantized MIDI note-off, but a sung vowel sustains past the
note.

This repair nudges each line's end *forward* to where the voice actually stops,
read straight off the separated vocals' RMS envelope, under three guards so it
never invents duration:

  * **strict tail threshold** — the tail must stay within ``tail_top_db`` of the
    loud reference (tighter than segmentation's 40 dB), so a decaying reverb or a
    breath (both quieter) does not get swallowed;
  * **next-line guard** — never extend past ``next_start - next_guard``, so a
    line cannot eat into the following line's onset;
  * **max extend** — never grow a line by more than ``max_extend`` seconds.

It only ever *grows* a line end (monotone), extends the line's last sung
(non-punct) char + its token + the line span together, and writes a sidecar —
canonical ``aligned*_midi.json`` is untouched.

Usage:
    python scripts/line_end_repair.py \
        --aligned outputs/<song>/aligned.vad_midi.json \
        --vocals  outputs/<song>/vocals.wav \
        --out     outputs/<song>/aligned.vad_midi.repaired.json
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import click
import numpy as np

# Reuse the exact RMS front-end used for segmentation (scripts/ is not a
# package, so load the sibling module by path — same trick as run_asr_segmented).
_RMS = Path(__file__).resolve().parent / "rms_vad_segments.py"
_spec = importlib.util.spec_from_file_location("_rms_vad", _RMS)
_rms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rms)  # type: ignore[union-attr]
frame_rms_db = _rms.frame_rms_db
DEFAULTS = _rms.DEFAULTS


def load_audio_mono(audio_path: str | Path, target_sr: int) -> np.ndarray:
    """Load audio as mono float32 at ``target_sr`` without requiring Whisper.

    The main karaoke venv already has soundfile/scipy; the lyrics venv has
    faster-whisper. Prefer the lightweight path and keep faster-whisper as a
    compatibility fallback for environments that lack a resampler.
    """
    try:
        import soundfile as sf

        y, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr == target_sr:
            return y

        try:
            from scipy.signal import resample_poly
        except ModuleNotFoundError:
            raise RuntimeError("scipy unavailable for resampling") from None

        gcd = math.gcd(int(sr), int(target_sr))
        up = target_sr // gcd
        down = sr // gcd
        return np.asarray(resample_poly(y, up, down), dtype=np.float32)
    except Exception:
        try:
            from faster_whisper.audio import decode_audio
        except ModuleNotFoundError:
            raise
        return np.asarray(decode_audio(str(audio_path), sampling_rate=target_sr), dtype=np.float32)


def voiced_tail_end(
    cur_end: float,
    next_start: float,
    rms_db: np.ndarray,
    hop_s: float,
    *,
    tail_top_db: float,
    max_extend: float,
    next_guard: float,
    tail_gap: float,
    decay_db: float = 0.0,
) -> float:
    """Walk the RMS envelope forward from ``cur_end`` and return the time the
    voice actually stops, bounded by the next-line and max-extend guards.

    Returns a value >= ``cur_end`` (never shortens a line).

    ``decay_db`` (Phase-0 relative-decay mode, survey B1): when > 0, a frame
    also counts as voiced if it stays within ``decay_db`` of the note's local
    peak (measured in the 0.4 s before ``cur_end``). This follows the *singer's
    natural decay* of a sustained vowel past the strict absolute floor, instead
    of only chasing a fixed ``-tail_top_db`` threshold — the sustained tail is a
    relative-loudness phenomenon, not an absolute one.
    """
    n = len(rms_db)

    def fidx(t: float) -> int:
        return int(np.clip(round(t / hop_s), 0, n - 1))

    ceil = min(next_start - next_guard, cur_end + max_extend)
    if ceil <= cur_end:
        return cur_end

    voiced = rms_db > -tail_top_db
    if decay_db > 0:
        lo = fidx(cur_end - 0.4)
        hi = fidx(cur_end)
        local_peak = float(np.max(rms_db[lo:hi + 1])) if hi >= lo else float(np.max(rms_db))
        voiced = voiced | (rms_db > local_peak - decay_db)
    i = fidx(cur_end)
    j = fidx(ceil)
    last_voiced: int | None = i if voiced[i] else None
    gap = 0.0
    k = i
    while k <= j:
        if voiced[k]:
            last_voiced = k
            gap = 0.0
        else:
            gap += hop_s
            if gap > tail_gap:  # a real silence/breath — stop following the tail
                break
        k += 1

    if last_voiced is None or last_voiced <= i:
        return cur_end
    new_end = times_of(last_voiced, hop_s) + hop_s
    return float(min(max(new_end, cur_end), ceil))


def times_of(frame: int, hop_s: float) -> float:
    return frame * hop_s


def last_sung_char(line: dict) -> dict | None:
    """The line's last non-punct char that carries timing — the one whose wipe
    should reach the vocal offset.  Trailing punctuation (zero-dur ``)`` etc.)
    is skipped."""
    target = None
    for tok in line.get("tokens", []):
        if tok.get("is_punct"):
            continue
        for ch in tok.get("chars", []):
            if ch.get("start") is not None:
                target = ch
    return target


def repair(
    lines: list[dict],
    rms_db: np.ndarray,
    hop_s: float,
    duration: float,
    *,
    tail_top_db: float,
    max_extend: float,
    next_guard: float,
    tail_gap: float,
    decay_db: float = 0.0,
) -> list[tuple[int, float, float]]:
    """Mutate ``lines`` in place; return (line_idx, old_end, new_end) for each
    line actually extended."""
    changes: list[tuple[int, float, float]] = []
    for i, line in enumerate(lines):
        cur_end = line.get("end")
        if cur_end is None:
            continue
        next_start = lines[i + 1].get("start", duration) if i + 1 < len(lines) else duration
        if next_start is None:
            next_start = duration
        new_end = voiced_tail_end(
            cur_end, next_start, rms_db, hop_s,
            tail_top_db=tail_top_db, max_extend=max_extend,
            next_guard=next_guard, tail_gap=tail_gap, decay_db=decay_db,
        )
        if new_end <= cur_end + 1e-3:
            continue
        ch = last_sung_char(line)
        if ch is None or ch["end"] >= new_end:
            continue
        ch["end"] = round(new_end, 3)
        # keep the containing token + line span consistent with the grown char
        for tok in line.get("tokens", []):
            if ch in tok.get("chars", []):
                tok["end"] = max(tok.get("end", new_end), round(new_end, 3))
        line["end"] = round(new_end, 3)
        changes.append((i, cur_end, new_end))
    return changes


@click.command()
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--vocals", "vocals_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "-o", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--tail-top-db", default=26.0, show_default=True,
              help="Tail must stay within this many dB of the loud ref (stricter "
                   "than segmentation's 40 dB excludes reverb/breath).")
@click.option("--max-extend", default=2.0, show_default=True,
              help="Never grow a line end by more than this many seconds.")
@click.option("--next-guard", default=0.25, show_default=True,
              help="Stay at least this many seconds before the next line's start.")
@click.option("--tail-gap", default=0.18, show_default=True,
              help="Tolerate voiced dips up to this long; a longer silence stops the tail.")
@click.option("--decay-db", default=0.0, show_default=True,
              help="Phase-0 relative-decay tail (survey B1): also follow frames "
                   "within this many dB of the note's local peak, capturing the "
                   "sung sustain decay below the absolute floor. 0 = legacy.")
def main(
    aligned_path: str,
    vocals_path: str,
    out_path: str,
    tail_top_db: float,
    max_extend: float,
    next_guard: float,
    tail_gap: float,
    decay_db: float,
) -> None:
    lines = json.loads(Path(aligned_path).read_text(encoding="utf-8"))

    sr = DEFAULTS["sr"]
    hop = DEFAULTS["hop_length"]
    print(f"decoding {vocals_path} @ {sr} Hz ...", flush=True)
    y = load_audio_mono(vocals_path, sr)
    duration = len(y) / sr
    rms_db = frame_rms_db(y, DEFAULTS["frame_length"], hop)
    hop_s = hop / sr

    changes = repair(
        lines, rms_db, hop_s, duration,
        tail_top_db=tail_top_db, max_extend=max_extend,
        next_guard=next_guard, tail_gap=tail_gap, decay_db=decay_db,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"extended {len(changes)} line end(s) -> {out_path}", flush=True)
    for i, old, new in changes:
        print(f"  line {i:2d}: {old:7.2f} -> {new:7.2f}  (+{new - old:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
