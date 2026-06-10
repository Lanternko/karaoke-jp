#!/usr/bin/env python3
"""GAME-backbone melody chain, one command.

Pins the configuration validated on Chidori humangold + byoushin
(2026-06-10):

    GAME large (-l ja) on the separated vocals
      -> score_note_postfix --extend-sustains   (refine/shakuri measurably
                                                 HURT on GAME output; fill and
                                                 tail-falls are no-ops on it)
      -> melody_union with the classic-chain MIDI as fallback (GAME clips
         soft low notes and sustain tails; the mora chain covers them)
      -> add_midi_markers

GAME runs inside its own venv (torch cu129 for RTX 5090 / sm_120); all other
stages use this interpreter. Update constants only with a re-validation run;
MEMORY.md records the evidence.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
GAME_DIR = ROOT / "third_party" / "GAME"
GAME_PY = Path.home() / "venvs" / "karaoke-jp-game" / "bin" / "python"
GAME_MODEL = GAME_DIR / "pretrained" / "GAME-1.0-large" / "model.pt"

POSTFIX_FLAGS = ["--extend-sustains"]


def _run(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(args, check=True, cwd=cwd)


@click.command()
@click.option("--vocals", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--fallback-midi", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Classic-chain melody MIDI (e.g. melody_markers.scorefix.mid).")
@click.option("--f0", "f0_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--bpm-file", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--quarters-per-page", type=int, default=10, show_default=True)
@click.option("--language", default="ja", show_default=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
def main(vocals: str, fallback_midi: str, f0_path: str, aligned_path: str,
         bpm_file: str, quarters_per_page: int, language: str, out_path: str) -> None:
    vocals_p = Path(vocals).resolve()
    with tempfile.TemporaryDirectory(prefix="game-chain-") as tmp:
        tmp_p = Path(tmp)
        # GAME writes <stem>.mid next to a COPY of the input inside tmp, so the
        # song dir is never polluted with generically named outputs.
        wav = tmp_p / vocals_p.name
        wav.symlink_to(vocals_p)
        _run([str(GAME_PY), "infer.py", "extract", str(wav),
              "-m", str(GAME_MODEL), "-l", language, "--output-formats", "mid"],
             cwd=GAME_DIR)
        game_mid = wav.with_suffix(".mid")
        pf_mid = tmp_p / "postfix.mid"
        union_mid = tmp_p / "union.mid"
        _run([sys.executable, str(SCRIPTS / "score_note_postfix.py"),
              "--midi", str(game_mid), "--f0", f0_path, "--aligned", aligned_path,
              "--out", str(pf_mid), *POSTFIX_FLAGS])
        _run([sys.executable, str(SCRIPTS / "melody_union.py"),
              "--primary", str(pf_mid), "--fallback", fallback_midi,
              "--out", str(union_mid)])
        _run([sys.executable, str(SCRIPTS / "add_midi_markers.py"),
              "--midi", str(union_mid), "--out", out_path,
              "--mode", "beat", "--bpm-file", bpm_file,
              "--quarters-per-page", str(quarters_per_page),
              "--aligned", aligned_path])
    click.echo(f"[game-chain] wrote {out_path}")


if __name__ == "__main__":
    main()
