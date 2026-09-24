"""CLI: infer RMVPE F0 via SOME's bundled modules and dump ``.npz``."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOME_DIR = ROOT / "third_party" / "SOME"
sys.path.insert(0, str(SOME_DIR))

import click
import librosa
import numpy as np

from modules.rmvpe.inference import RMVPE


@click.command()
@click.option("--wav", "wav_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--hop-length", default=160, type=int, show_default=True)
@click.option("--threshold", default=0.03, type=float, show_default=True)
@click.option("--viterbi/--no-viterbi", default=False, show_default=True)
def main(
    wav_path: str,
    model_path: str,
    out_path: str,
    device: str,
    hop_length: int,
    threshold: float,
    viterbi: bool,
) -> None:
    waveform, sample_rate = librosa.load(wav_path, sr=None, mono=True)
    rmvpe = RMVPE(model_path, hop_length=hop_length, device=device)
    f0 = rmvpe.infer_from_audio(
        waveform,
        sample_rate=sample_rate,
        thred=threshold,
        use_viterbi=viterbi,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        f0=f0.astype(np.float32),
        hop_seconds=np.array([hop_length / 16000.0], dtype=np.float32),
    )
    print(f"[rmvpe] wrote {out}")


if __name__ == "__main__":
    main()
