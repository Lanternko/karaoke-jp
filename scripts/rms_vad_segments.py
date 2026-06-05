"""RMS-VAD pre-segmentation for singing vocals.

Whisper's built-in (Silero) VAD is tuned for speech and routinely drops quiet
held-vowel entries and merges whole sung phrases into mega-segments — the exact
failure mode behind ``haru-hikage``'s bridge (113-147s) and chorus (160-192s)
collapse.  This script ignores Whisper's VAD entirely and derives segment
boundaries from the *separated* vocals' RMS energy, following the RMS-VAD /
"Cut & Merge" recipe from "Exploiting Music Source Separation for Automatic
Lyrics Transcription with Whisper" (arXiv:2506.15514).

Pipeline:

    decode → frame RMS (dB) → threshold to voiced frames → contiguous runs
    → drop sub-threshold blips → MERGE runs across short silences (phrase
    grouping) → pad → CUT any segment longer than Whisper's window at its
    lowest-energy valley.

The emitted ``rms_segments.json`` is consumed by ``scripts/run_asr_segmented.py``
which transcribes each segment with ``vad_filter=False``.  This is a *sidecar*:
it does not touch ``run_asr.py``, ``align_lyrics.py``, the Snakefile, or any
canonical output.

Usage:
    python scripts/rms_vad_segments.py outputs/<song>/vocals.wav \
        -o outputs/<song>/rms_segments.json
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

# ---------------------------------------------------------------------------
# Pure, testable segmentation core (no audio I/O, no Whisper).
# ---------------------------------------------------------------------------

DEFAULTS = {
    "sr": 16000,
    "frame_length": 2048,   # ~128 ms @ 16 kHz
    "hop_length": 512,      # ~32 ms  @ 16 kHz
    "top_db": 40.0,         # voiced if frame is within top_db of the loud ref
    "min_voiced_dur": 0.15,  # drop voiced blips shorter than this (transients)
    "merge_gap": 0.6,       # merge phrases separated by a silence < this
    "pad": 0.25,            # widen each segment so onsets/offsets aren't clipped
    "max_len": 28.0,        # keep under Whisper's 30 s window
    "min_cut_frac": 0.4,    # a cut never produces a left piece < this * max_len
}


def frame_rms_db(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Per-frame RMS energy of ``y`` expressed in dB relative to the loud
    reference (95th percentile of frame RMS).  0 dB ≈ loud singing, very
    negative ≈ silence/breath."""
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))
    n_frames = 1 + (len(y) - frame_length) // hop_length
    # sliding_window_view avoids materialising a copy per frame.
    windows = np.lib.stride_tricks.sliding_window_view(y, frame_length)[
        :: hop_length
    ][:n_frames]
    rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1) + 1e-12)
    ref = np.percentile(rms, 95) if np.any(rms > 0) else 1.0
    ref = max(ref, 1e-9)
    return 20.0 * np.log10(np.maximum(rms, 1e-9) / ref)


def voiced_runs(
    rms_db: np.ndarray,
    times: np.ndarray,
    top_db: float,
    min_voiced_dur: float,
) -> list[tuple[float, float]]:
    """Contiguous spans where the frame is within ``top_db`` of the reference,
    dropping spans shorter than ``min_voiced_dur`` seconds."""
    voiced = rms_db > -top_db
    runs: list[tuple[float, float]] = []
    i = 0
    n = len(voiced)
    hop = times[1] - times[0] if len(times) > 1 else 0.032
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        start = times[i]
        end = times[j - 1] + hop  # span covers the last voiced frame's hop
        if end - start >= min_voiced_dur:
            runs.append((float(start), float(end)))
        i = j
    return runs


def merge_runs(
    runs: list[tuple[float, float]], merge_gap: float
) -> list[tuple[float, float]]:
    """Glue adjacent voiced runs separated by a silence shorter than
    ``merge_gap`` — turns word-level runs into phrase-level segments."""
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        ps, pe = merged[-1]
        if start - pe <= merge_gap:
            merged[-1] = (ps, end)
        else:
            merged.append((start, end))
    return merged


def pad_and_clamp(
    runs: list[tuple[float, float]], pad: float, duration: float
) -> list[tuple[float, float]]:
    """Widen each segment by ``pad`` on both sides, clamp to [0, duration], and
    re-merge any segments that overlap after padding."""
    padded = [
        (max(0.0, s - pad), min(duration, e + pad)) for s, e in runs
    ]
    # padding can make neighbours overlap; merge_gap=0 collapses those.
    return merge_runs(padded, merge_gap=0.0)


def cut_long(
    runs: list[tuple[float, float]],
    rms_db: np.ndarray,
    times: np.ndarray,
    max_len: float,
    min_cut_frac: float,
) -> list[tuple[float, float]]:
    """Split any segment longer than ``max_len`` at its lowest-energy frame,
    searching the window ``[start + min_cut_frac*max_len, start + max_len]`` so
    each left piece stays within Whisper's limit and termination is guaranteed.
    """
    hop = times[1] - times[0] if len(times) > 1 else 0.032

    def frame_idx(t: float) -> int:
        return int(np.clip(round(t / hop), 0, len(rms_db) - 1))

    out: list[tuple[float, float]] = []
    stack = list(reversed(runs))
    while stack:
        start, end = stack.pop()
        if end - start <= max_len:
            out.append((start, end))
            continue
        lo = frame_idx(start + min_cut_frac * max_len)
        hi = frame_idx(start + max_len)
        if hi <= lo:
            cut_t = start + max_len
        else:
            cut_t = times[lo + int(np.argmin(rms_db[lo : hi + 1]))]
        out.append((start, float(cut_t)))
        stack.append((float(cut_t), end))  # re-examine the tail
    return sorted(out)


def segment_audio(
    y: np.ndarray,
    sr: int,
    *,
    frame_length: int = DEFAULTS["frame_length"],
    hop_length: int = DEFAULTS["hop_length"],
    top_db: float = DEFAULTS["top_db"],
    min_voiced_dur: float = DEFAULTS["min_voiced_dur"],
    merge_gap: float = DEFAULTS["merge_gap"],
    pad: float = DEFAULTS["pad"],
    max_len: float = DEFAULTS["max_len"],
    min_cut_frac: float = DEFAULTS["min_cut_frac"],
) -> list[tuple[float, float]]:
    """Full RMS-VAD: waveform → list of (start, end) segments in seconds."""
    rms_db = frame_rms_db(y, frame_length, hop_length)
    times = np.arange(len(rms_db)) * (hop_length / sr)
    duration = len(y) / sr
    runs = voiced_runs(rms_db, times, top_db, min_voiced_dur)
    runs = merge_runs(runs, merge_gap)
    runs = pad_and_clamp(runs, pad, duration)
    runs = cut_long(runs, rms_db, times, max_len, min_cut_frac)
    return runs


def covers(segments: list[tuple[float, float]], lo: float, hi: float) -> float:
    """Fraction of window [lo, hi] covered by ``segments`` — a quick QA probe
    for known-bad windows (e.g. haru-hikage bridge/chorus)."""
    total = 0.0
    for s, e in segments:
        a, b = max(s, lo), min(e, hi)
        if b > a:
            total += b - a
    return total / max(hi - lo, 1e-9)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.argument("vocals_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "-o", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--top-db", default=DEFAULTS["top_db"], show_default=True,
              help="Voiced if a frame is within this many dB of the loud reference.")
@click.option("--merge-gap", default=DEFAULTS["merge_gap"], show_default=True,
              help="Merge phrases separated by a silence shorter than this (s).")
@click.option("--pad", default=DEFAULTS["pad"], show_default=True,
              help="Widen each segment by this many seconds on both sides.")
@click.option("--max-len", default=DEFAULTS["max_len"], show_default=True,
              help="Cut segments longer than this (must stay < Whisper's 30 s).")
@click.option("--min-voiced-dur", default=DEFAULTS["min_voiced_dur"], show_default=True,
              help="Drop voiced blips shorter than this (s).")
@click.option("--probe", multiple=True,
              help="lo:hi window to report coverage for, e.g. --probe 113:147. Repeatable.")
def main(
    vocals_path: str,
    out_path: str,
    top_db: float,
    merge_gap: float,
    pad: float,
    max_len: float,
    min_voiced_dur: float,
    probe: tuple[str, ...],
) -> None:
    from faster_whisper.audio import decode_audio

    sr = DEFAULTS["sr"]
    print(f"decoding {vocals_path} @ {sr} Hz ...", flush=True)
    y = decode_audio(vocals_path, sampling_rate=sr)
    y = np.asarray(y, dtype=np.float32)
    duration = len(y) / sr

    segments = segment_audio(
        y, sr,
        top_db=top_db,
        min_voiced_dur=min_voiced_dur,
        merge_gap=merge_gap,
        pad=pad,
        max_len=max_len,
    )
    coverage = sum(e - s for s, e in segments)

    params = {
        "sr": sr,
        "frame_length": DEFAULTS["frame_length"],
        "hop_length": DEFAULTS["hop_length"],
        "top_db": top_db,
        "min_voiced_dur": min_voiced_dur,
        "merge_gap": merge_gap,
        "pad": pad,
        "max_len": max_len,
    }
    payload = {
        "source": vocals_path,
        "duration": round(duration, 3),
        "n_segments": len(segments),
        "voiced_sec": round(coverage, 3),
        "params": params,
        "segments": [
            {"start": round(s, 3), "end": round(e, 3)} for s, e in segments
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    longest = max((e - s for s, e in segments), default=0.0)
    print(
        f"{len(segments)} segments, voiced {coverage:.1f}s / {duration:.1f}s "
        f"({coverage / max(duration, 1e-9):.0%}), longest {longest:.1f}s -> {out_path}",
        flush=True,
    )
    for spec in probe:
        try:
            lo_s, hi_s = spec.split(":")
            lo, hi = float(lo_s), float(hi_s)
        except ValueError:
            print(f"  bad --probe {spec!r} (want lo:hi)", flush=True)
            continue
        print(f"  coverage {lo:.0f}-{hi:.0f}s: {covers(segments, lo, hi):.0%}", flush=True)


if __name__ == "__main__":
    main()
