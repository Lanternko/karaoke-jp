#!/usr/bin/env python3
"""Extract a pYIN F0 track to ``.npz`` for pitch QA cross-checks."""
from __future__ import annotations

from pathlib import Path

import click
import librosa
import numpy as np


@click.command()
@click.option("--wav", "wav_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--sample-rate", default=22050, type=int, show_default=True)
@click.option("--hop-length", default=512, type=int, show_default=True)
@click.option("--frame-length", default=2048, type=int, show_default=True)
@click.option("--fmin", default="C2", show_default=True)
@click.option("--fmax", default="C7", show_default=True)
def main(
    wav_path: str,
    out_path: str,
    sample_rate: int,
    hop_length: int,
    frame_length: int,
    fmin: str,
    fmax: str,
) -> None:
    y, sr = librosa.load(wav_path, sr=sample_rate, mono=True)
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(fmin),
        fmax=librosa.note_to_hz(fmax),
        sr=sr,
        hop_length=hop_length,
        frame_length=frame_length,
    )
    times = librosa.frames_to_time(np.arange(f0.size), sr=sr, hop_length=hop_length)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, f0=f0.astype(np.float32), times=times.astype(np.float32))
    print(f"[pyin] wrote {out}")


if __name__ == "__main__":
    main()
