#!/usr/bin/env python3
"""Run BasicPitch on a separated vocals stem, saving note events CSV + MIDI.

Thin CLI wrapper so the Snakefile score_chain rule has a declared, repeatable
entry point (BasicPitch's own CLI pulls in optional backends we don't ship).
"""
from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option("--vocals", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False), required=True)
def main(vocals: str, out_dir: str) -> None:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict_and_save

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    predict_and_save(
        [vocals],
        str(out),
        True,   # save_midi
        False,  # sonify_midi
        False,  # save_model_outputs
        True,   # save_notes
        ICASSP_2022_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
