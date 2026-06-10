#!/usr/bin/env python3
"""Canonical audio-only score chain, one command.

Pins the validated recipe (Chidori humangold + byoushin cross-check,
2026-06-10) so the flags cannot drift from what was measured:

    octavefix MIDI
      -> bp_hybrid_relabel  (max-dist 1, min-score 0.02, min-ratio 1.2)
      -> score_note_postfix (--refine-boundaries --extend-sustains
                             --absorb-shakuri, plus --fill-morae when an
                             aligned JSON is available)
      -> add_midi_markers   (restores the page-marker track that
                             _write_midi-based stages drop)

Update the constants here ONLY together with a re-validation run; MEMORY.md
records the evidence behind them.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import click

SCRIPTS = Path(__file__).resolve().parent

BP_FLAGS = ["--max-dist", "1", "--min-score", "0.02", "--min-ratio", "1.2"]
POSTFIX_FLAGS = [
    "--refine-boundaries",
    "--extend-sustains",
    "--absorb-shakuri",
    "--capture-tail-falls",
]


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], check=True)


@click.command()
@click.option("--midi", "midi_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="melody_markers.octavefix.mid (or any note MIDI).")
@click.option("--basicpitch-csv", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--f0", "f0_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--bpm-file", type=click.Path(exists=True, dir_okay=False), required=True,
              help="melody_quantized.mid.bpm.txt (add_midi_markers beat grid).")
@click.option("--quarters-per-page", type=int, default=10, show_default=True,
              help="Must match the Snakefile QUARTERS_PER_PAGE bar-display scale.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
def main(midi_path: str, basicpitch_csv: str, f0_path: str, aligned_path: str | None,
         bpm_file: str, quarters_per_page: int, out_path: str) -> None:
    with tempfile.TemporaryDirectory(prefix="score-chain-") as tmp:
        bp_mid = str(Path(tmp) / "bp.mid")
        pf_mid = str(Path(tmp) / "postfix.mid")
        _run([str(SCRIPTS / "bp_hybrid_relabel.py"),
              "--midi", midi_path, "--basicpitch-csv", basicpitch_csv,
              "--out", bp_mid, *BP_FLAGS])
        postfix = [str(SCRIPTS / "score_note_postfix.py"),
                   "--midi", bp_mid, "--f0", f0_path, "--out", pf_mid,
                   *POSTFIX_FLAGS]
        if aligned_path:
            postfix += ["--aligned", aligned_path, "--fill-morae"]
        _run(postfix)
        markers = [str(SCRIPTS / "add_midi_markers.py"),
                   "--midi", pf_mid, "--out", out_path,
                   "--mode", "beat", "--bpm-file", bpm_file,
                   "--quarters-per-page", str(quarters_per_page)]
        if aligned_path:
            markers += ["--aligned", aligned_path]
        _run(markers)
    click.echo(f"[score-chain] wrote {out_path}")


if __name__ == "__main__":
    main()
