#!/usr/bin/env python3
"""Extract reference pitch-guide notes from a karaoke video overlay.

This is a ground-truth helper for *short key segments*.  It samples the official
guide melody at the red playhead, converts the detected y-position to MIDI pitch
with a robust F0-assisted calibration, and writes a sparse MIDI plus TSV.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from karaoke_jp.melody import _write_midi  # noqa: E402
from karaoke_jp.pitch_eval import F0Track  # noqa: E402
from karaoke_jp.score_melody import read_first_tempo_bpm  # noqa: E402


@dataclass(frozen=True)
class Segment:
    label: str
    start: float
    end: float


@dataclass(frozen=True)
class Sample:
    time: float
    y: float
    playhead_x: int
    frame_path: str


def _parse_segment(spec: str) -> Segment:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise click.BadParameter("segment must be LABEL:START:END, seconds")
    label, start, end = parts
    return Segment(label=label, start=float(start), end=float(end))


def _extract_frames(video: Path, seg: Segment, out_dir: Path, fps: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.png"):
        old.unlink()
    duration = seg.end - seg.start
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{seg.start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        str(out_dir / "frame_%05d.png"),
    ]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob("frame_*.png"))


def _detect_playhead_x(arr: np.ndarray, *, x_hint: int | None = None) -> int | None:
    top = arr[20:180, :, :3].astype(np.int16)
    r, g, b = top[..., 0], top[..., 1], top[..., 2]
    mask = (r > 180) & (g < 120) & (b > 80) & (b < 230)
    scores = mask.sum(axis=0)
    lo, hi = (40, max(41, arr.shape[1] - 40))
    if x_hint is not None:
        lo = max(40, x_hint - 120)
        hi = min(arr.shape[1] - 40, x_hint + 120)
    if hi <= lo:
        return x_hint
    rel = int(np.argmax(scores[lo:hi]))
    x = lo + rel
    if scores[x] < 35:
        return x_hint
    return x


def _detect_bar_y(arr: np.ndarray, x: int) -> float | None:
    y0, y1 = 24, 176
    x0 = max(0, x - 90)
    x1 = min(arr.shape[1], x + 91)
    crop = arr[y0:y1, x0:x1, :3].astype(np.int16)
    cols = np.arange(x0, x1)

    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    # Current official guide bars may be black/gray (past/current) or pink
    # (future/current). The background is saturated blue, so do NOT use a generic
    # saturation mask here.
    colored = (r > 170) & (g < 150) & (b > 100)
    dark_bar = (r < 70) & (g < 70) & (b < 70)
    # Remove near-white grid/text speckles.
    colored &= ~((r > 210) & (g > 210) & (b > 210))
    note_pixels = colored | dark_bar

    # Drop the playhead/glow itself; then score horizontal note components by
    # both size and closeness to the playhead. This avoids selecting the next
    # note far to the right or the red cursor glow.
    note_pixels[:, np.abs(cols - x) <= 10] = False
    row_score = np.zeros(note_pixels.shape[0], dtype=np.float64)
    for row_idx, row in enumerate(note_pixels):
        xs = cols[row]
        if xs.size < 3:
            continue
        dist = float(np.min(np.abs(xs - x)))
        if dist > 36.0:
            continue
        count = float(xs.size)
        row_score[row_idx] = count * np.exp(-dist / 24.0)
    if row_score.max(initial=0) < 3:
        return None
    smooth = np.convolve(row_score, np.ones(5), mode="same")
    peak = int(np.argmax(smooth))
    thresh = max(3.0, smooth[peak] * 0.35)
    lo = peak
    while lo > 0 and smooth[lo - 1] >= thresh:
        lo -= 1
    hi = peak
    while hi + 1 < smooth.size and smooth[hi + 1] >= thresh:
        hi += 1
    ys = np.arange(lo, hi + 1, dtype=np.float64)
    weights = np.maximum(row_score[lo : hi + 1], 1.0)
    return float(y0 + np.average(ys, weights=weights))


def _nearest_midi(track: F0Track, t: float) -> float | None:
    midi = track.midi
    if track.times.size == 0:
        return None
    idx = int(np.searchsorted(track.times, t))
    candidates = []
    for j in (idx - 1, idx, idx + 1):
        if 0 <= j < track.times.size and np.isfinite(midi[j]) and abs(track.times[j] - t) <= 0.04:
            candidates.append(float(midi[j]))
    if not candidates:
        return None
    return float(np.median(candidates))


def _calibration_refs(samples: list[Sample], rmvpe: F0Track, pyin: F0Track | None) -> tuple[np.ndarray, np.ndarray]:
    ys: list[float] = []
    mids: list[float] = []
    for sample in samples:
        r = _nearest_midi(rmvpe, sample.time)
        p = _nearest_midi(pyin, sample.time) if pyin is not None else None
        if r is not None and p is not None:
            if abs(r - p) <= 2.0:
                ref = (r + p) / 2.0
            else:
                continue
        elif r is not None:
            ref = r
        elif p is not None:
            ref = p
        else:
            continue
        ys.append(sample.y)
        mids.append(ref)
    if len(ys) < 20:
        raise ValueError(f"Not enough F0-agreed samples for calibration: {len(ys)}")
    return np.array(ys, dtype=np.float64), np.array(mids, dtype=np.float64)


def _fit_y_to_midi(samples: list[Sample], rmvpe: F0Track, pyin: F0Track | None) -> tuple[float, float, float]:
    ys, mids = _calibration_refs(samples, rmvpe, pyin)
    best: tuple[float, float, float] | None = None
    for slope in np.linspace(-0.28, -0.06, 1200):
        intercept = float(np.median(mids - slope * ys))
        pred = np.rint(slope * ys + intercept)
        err = np.abs(pred - mids)
        score = float(np.median(err) + 0.25 * np.percentile(err, 80))
        if best is None or score < best[0]:
            best = (score, slope, intercept)
    assert best is not None
    score, slope, intercept = best
    return slope, intercept, score


def _samples_to_notes(
    samples: list[Sample],
    *,
    slope: float,
    intercept: float,
    fps: float,
    min_duration: float,
    merge_gap: float,
) -> list[tuple[float, float, int]]:
    if not samples:
        return []
    events = [(s.time, int(round(slope * s.y + intercept))) for s in samples]
    notes: list[tuple[float, float, int]] = []
    start_t, last_t, pitch = events[0][0], events[0][0], events[0][1]
    max_step = 1.75 / fps
    for t, p in events[1:]:
        if p == pitch and t - last_t <= max_step:
            last_t = t
            continue
        end_t = last_t + 1.0 / fps
        if end_t - start_t >= min_duration:
            notes.append((start_t, end_t, pitch))
        start_t, last_t, pitch = t, t, p
    end_t = last_t + 1.0 / fps
    if end_t - start_t >= min_duration:
        notes.append((start_t, end_t, pitch))

    merged: list[tuple[float, float, int]] = []
    for note in notes:
        if merged and note[2] == merged[-1][2] and note[0] - merged[-1][1] <= merge_gap:
            prev = merged[-1]
            merged[-1] = (prev[0], note[1], prev[2])
        else:
            merged.append(note)
    return merged


def _sort_unique_samples(samples: list[Sample]) -> list[Sample]:
    """Return samples in time order, dropping duplicate frames from overlapped segments."""
    unique: dict[float, Sample] = {}
    for sample in sorted(samples, key=lambda item: (item.time, item.frame_path)):
        # Segment overlaps can hit the same video timestamp twice.  Rounding is
        # enough here because sample times are generated on the fps grid.
        unique.setdefault(round(sample.time, 6), sample)
    return list(unique.values())


def _enforce_monophonic_notes(
    notes: list[tuple[float, float, int]],
    *,
    min_duration: float,
) -> list[tuple[float, float, int]]:
    """Trim tiny overlaps so the guide melody is safe to serialize as one track.

    The extracted karaoke guide is conceptually monophonic, but frame-level
    detection and segment overlap can leave notes that overlap by a few frames.
    ``_write_midi`` serializes one note track, so overlapping note starts would
    be delayed by the previous note-off and silently drift in the saved MIDI.
    Trimming the previous note to the next onset preserves the intended
    playhead-derived transition time.
    """
    cleaned: list[tuple[float, float, int]] = []
    for start, end, pitch in sorted(notes, key=lambda item: (item[0], item[1], item[2])):
        if end - start < min_duration:
            continue
        if cleaned and start < cleaned[-1][1]:
            prev_start, prev_end, prev_pitch = cleaned[-1]
            if start - prev_start >= min_duration:
                cleaned[-1] = (prev_start, start, prev_pitch)
            else:
                cleaned.pop()
        if cleaned and pitch == cleaned[-1][2] and start - cleaned[-1][1] <= 0.04:
            prev_start, prev_end, prev_pitch = cleaned[-1]
            cleaned[-1] = (prev_start, max(prev_end, end), prev_pitch)
        elif end - start >= min_duration:
            cleaned.append((start, end, pitch))
    return cleaned


@click.command()
@click.option("--video", "video_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--rmvpe-f0", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--pyin-f0", type=click.Path(exists=True, dir_okay=False))
@click.option("--fallback-midi", type=click.Path(exists=True, dir_okay=False))
@click.option("--segment", "segments", multiple=True, callback=lambda _ctx, _param, vals: [_parse_segment(v) for v in vals])
@click.option("--out-midi", type=click.Path(dir_okay=False), required=True)
@click.option("--out-tsv", type=click.Path(dir_okay=False), required=True)
@click.option("--out-json", type=click.Path(dir_okay=False), required=True)
@click.option("--frames-dir", type=click.Path(file_okay=False), required=True)
@click.option("--fps", type=float, default=30.0, show_default=True)
@click.option("--min-duration", type=float, default=0.06, show_default=True)
@click.option("--merge-gap", type=float, default=0.04, show_default=True)
def main(
    video_path: str,
    rmvpe_f0: str,
    pyin_f0: str | None,
    fallback_midi: str | None,
    segments: list[Segment],
    out_midi: str,
    out_tsv: str,
    out_json: str,
    frames_dir: str,
    fps: float,
    min_duration: float,
    merge_gap: float,
) -> None:
    if not segments:
        raise click.UsageError("At least one --segment LABEL:START:END is required.")
    video = Path(video_path)
    frames_root = Path(frames_dir)
    rmvpe = F0Track.from_npz(rmvpe_f0)
    pyin = F0Track.from_npz(pyin_f0) if pyin_f0 else None

    all_samples: list[Sample] = []
    segment_frame_examples: dict[str, str] = {}
    for seg in segments:
        prev_x: int | None = None
        frame_paths = _extract_frames(video, seg, frames_root / seg.label, fps)
        for idx, frame_path in enumerate(frame_paths):
            arr = np.asarray(Image.open(frame_path).convert("RGB"))
            x = _detect_playhead_x(arr, x_hint=prev_x)
            if x is None:
                continue
            prev_x = x
            y = _detect_bar_y(arr, x)
            if y is None:
                continue
            t = seg.start + idx / fps
            all_samples.append(Sample(time=t, y=y, playhead_x=x, frame_path=str(frame_path)))
        if frame_paths:
            segment_frame_examples[seg.label] = str(frame_paths[min(len(frame_paths) // 2, len(frame_paths) - 1)])

    all_samples = _sort_unique_samples(all_samples)
    slope, intercept, fit_score = _fit_y_to_midi(all_samples, rmvpe, pyin)
    notes = _samples_to_notes(
        all_samples,
        slope=slope,
        intercept=intercept,
        fps=fps,
        min_duration=min_duration,
        merge_gap=merge_gap,
    )
    notes = _enforce_monophonic_notes(notes, min_duration=min_duration)
    tempo = read_first_tempo_bpm(fallback_midi) if fallback_midi else 120.0
    _write_midi(notes, Path(out_midi), tempo=tempo)

    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_tsv).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["start", "end", "duration", "pitch", "segment"])
        for start, end, pitch in notes:
            label = next((s.label for s in segments if s.start <= start < s.end), "")
            writer.writerow([f"{start:.3f}", f"{end:.3f}", f"{end-start:.3f}", pitch, label])

    meta = {
        "video": str(video),
        "segments": [seg.__dict__ for seg in segments],
        "samples": len(all_samples),
        "notes": len(notes),
        "slope": slope,
        "intercept": intercept,
        "fit_score": fit_score,
        "fps": fps,
        "example_frames": segment_frame_examples,
    }
    Path(out_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    click.echo(
        f"[ref-video] samples={len(all_samples)} notes={len(notes)} "
        f"slope={slope:.5f} intercept={intercept:.2f} fit_score={fit_score:.3f}"
    )
    click.echo(f"[ref-video] wrote {out_midi}")
    click.echo(f"[ref-video] wrote {out_tsv}")


if __name__ == "__main__":
    main()
