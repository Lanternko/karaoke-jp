#!/usr/bin/env python3
"""Fuse two frame-wise F0 tracks for pitch-ablation experiments.

The output is another ``.npz`` with ``times`` and ``f0`` arrays, so it can be
fed to the same melody/refit tools as RMVPE or pYIN.  This is intentionally a
late-fusion sidecar: it does not average contours by default, it only fills
unvoiced primary frames from a secondary estimator.
"""
from __future__ import annotations

from pathlib import Path

import click
import numpy as np


def _midi_from_f0(f0_hz: np.ndarray) -> np.ndarray:
    midi = np.full(f0_hz.shape, np.nan, dtype=np.float64)
    voiced = np.isfinite(f0_hz) & (f0_hz > 0)
    midi[voiced] = 69.0 + 12.0 * np.log2(f0_hz[voiced] / 440.0)
    return midi


def _f0_from_midi(midi: np.ndarray) -> np.ndarray:
    f0 = np.zeros(midi.shape, dtype=np.float64)
    voiced = np.isfinite(midi)
    f0[voiced] = 440.0 * (2.0 ** ((midi[voiced] - 69.0) / 12.0))
    return f0


def _load_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        f0 = data["f0"].astype(np.float64)
        if "times" in data:
            times = data["times"].astype(np.float64)
        elif "hop_seconds" in data:
            hop = float(np.asarray(data["hop_seconds"]).reshape(-1)[0])
            times = np.arange(f0.size, dtype=np.float64) * hop
        else:
            raise ValueError(f"{path} must contain times or hop_seconds")
    if times.shape != f0.shape:
        raise ValueError(f"{path} has mismatched f0/times shapes")
    return times, f0


def _resample_midi_to_times(
    source_times: np.ndarray,
    source_midi: np.ndarray,
    target_times: np.ndarray,
    *,
    max_distance: float,
) -> np.ndarray:
    aligned = np.full(target_times.shape, np.nan, dtype=np.float64)
    for idx, time_s in enumerate(target_times):
        pos = int(np.searchsorted(source_times, time_s))
        candidates: list[float] = []
        for j in (pos - 1, pos, pos + 1):
            if 0 <= j < source_times.size and abs(source_times[j] - time_s) <= max_distance:
                value = source_midi[j]
                if np.isfinite(value):
                    candidates.append(float(value))
        if candidates:
            aligned[idx] = float(np.median(candidates))
    return aligned


def fuse_midi_tracks(
    primary_midi: np.ndarray,
    secondary_midi: np.ndarray,
    *,
    strategy: str,
    agreement_tolerance: float,
) -> np.ndarray:
    fused = primary_midi.copy()
    secondary_voiced = np.isfinite(secondary_midi)

    if strategy == "primary-fill":
        fill = ~np.isfinite(fused) & secondary_voiced
        fused[fill] = secondary_midi[fill]
        return fused

    both = np.isfinite(primary_midi) & secondary_voiced
    close = both & (np.abs(primary_midi - secondary_midi) <= agreement_tolerance)
    fused[close] = (primary_midi[close] + secondary_midi[close]) / 2.0

    if strategy == "agree-avg-else-primary":
        fill = ~np.isfinite(fused) & secondary_voiced
        fused[fill] = secondary_midi[fill]
        return fused

    if strategy == "agree-avg-else-secondary":
        far = both & ~close
        fused[far] = secondary_midi[far]
        fill = ~np.isfinite(fused) & secondary_voiced
        fused[fill] = secondary_midi[fill]
        return fused

    raise ValueError(f"Unsupported strategy: {strategy}")


@click.command()
@click.option("--primary", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--secondary", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option(
    "--strategy",
    type=click.Choice(["primary-fill", "agree-avg-else-primary", "agree-avg-else-secondary"]),
    default="primary-fill",
    show_default=True,
)
@click.option("--max-distance", type=float, default=0.04, show_default=True)
@click.option("--agreement-tolerance", type=float, default=2.0, show_default=True)
@click.option(
    "--target-times",
    type=click.Choice(["primary", "secondary"]),
    default="primary",
    show_default=True,
    help="Frame grid for the fused output. Use secondary to keep a smoother estimator's timing grid while preferring primary pitches.",
)
def main(
    primary: str,
    secondary: str,
    out_path: str,
    strategy: str,
    max_distance: float,
    agreement_tolerance: float,
    target_times: str,
) -> None:
    primary_times, primary_f0 = _load_npz(primary)
    secondary_times, secondary_f0 = _load_npz(secondary)
    if target_times == "primary":
        output_times = primary_times
        primary_midi = _midi_from_f0(primary_f0)
        secondary_midi = _resample_midi_to_times(
            secondary_times,
            _midi_from_f0(secondary_f0),
            output_times,
            max_distance=max_distance,
        )
    else:
        output_times = secondary_times
        primary_midi = _resample_midi_to_times(
            primary_times,
            _midi_from_f0(primary_f0),
            output_times,
            max_distance=max_distance,
        )
        secondary_midi = _midi_from_f0(secondary_f0)
    fused_midi = fuse_midi_tracks(
        primary_midi,
        secondary_midi,
        strategy=strategy,
        agreement_tolerance=agreement_tolerance,
    )
    fused_f0 = _f0_from_midi(fused_midi)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, times=output_times, f0=fused_f0)
    primary_voiced = int(np.count_nonzero(np.isfinite(primary_midi)))
    secondary_used = int(np.count_nonzero(np.isfinite(fused_midi) & ~np.isfinite(primary_midi)))
    fused_voiced = int(np.count_nonzero(np.isfinite(fused_midi)))
    click.echo(
        f"[fuse-f0] wrote {out} "
        f"(primary_voiced={primary_voiced}, secondary_fill={secondary_used}, fused_voiced={fused_voiced})"
    )


if __name__ == "__main__":
    main()
