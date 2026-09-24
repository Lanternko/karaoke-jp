#!/usr/bin/env python3
"""GAME-backbone melody chain, one command.

Pinned configuration:

    GAME large (-l ja) on the separated vocals
      -> score_note_postfix --extend-sustains   (refine/shakuri measurably
                                                 HURT on GAME output; fill and
                                                 tail-falls are no-ops on it)
      -> melody_union with the classic-chain MIDI as fallback (GAME clips
         soft low notes and sustain tails; the mora chain covers them)
      -> add_midi_markers

GAME runs inside its own venv (torch cu129 for RTX 5090 / sm_120); all other
stages use this interpreter.
"""
from __future__ import annotations

import json as _json
import shutil
import os
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
GAME_SEEDED_LAUNCHER = SCRIPTS / "game_infer_seeded.py"


def _load_canonical_profile() -> dict:
    # config/versions.json is the source of truth for pitch_chain +
    # display_grid params. Soft fallback so a fresh checkout still runs.
    try:
        data = _json.loads((ROOT / "config" / "versions.json").read_text())
        # KARAOKE_PROFILE=<id> renders a non-canonical profile
        return data["profiles"][os.environ.get("KARAOKE_PROFILE") or data["canonical"]]
    except Exception:
        return {}


_PROFILE = _load_canonical_profile()
_PITCH_CHAIN = _PROFILE.get("pitch_chain", {})

POSTFIX_FLAGS = _PITCH_CHAIN.get("postfix_flags", ["--extend-sustains", "--keep-repeats"])

# Boundary decoding threshold for GAME's segmenter. 0.3 (vs default 0.2)
# cuts spurious boundary splits on both a cappella and separated vocals.
GAME_SEG_THRESHOLD = _PITCH_CHAIN.get("game_seg_threshold", "0.3")

# GAME's segmenter is a D3PM diffusion model whose sampling loop draws
# torch.rand_like() at INFERENCE time (8 steps), and upstream seeds nothing:
# unseeded, back-to-back runs on identical audio give different note sets.
# With the seed pinned, output is bitwise identical. Onset
# times are the join key for overrides/<song>_pitch_patch.json, so this seed is
# load-bearing: change it and every hand-written pitch patch has to be re-checked.
GAME_SEED = int(_PITCH_CHAIN.get("game_seed", 0))


def _run(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(args, check=True, cwd=cwd)


@click.command()
@click.option("--vocals", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--fallback-midi", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Classic-chain melody MIDI (e.g. melody_markers.scorefix.mid).")
@click.option("--f0", "f0_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--bpm-file", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--language", default="ja", show_default=True)
@click.option("--pitch-patch", "pitch_patch_path",
              type=click.Path(exists=True, dir_okay=False), default=None,
              help="Hand-written per-song display pitch fixes "
              "(overrides/<song>_pitch_patch.json), forwarded to make_display_grid.")
@click.option("--rms-segments", "rms_segments_path",
              type=click.Path(exists=True, dir_okay=False), default=None,
              help="rms_segments.json, forwarded to make_display_grid so "
              "instrumental-break bleed notes never render.")
@click.option("--pyin-f0", "pyin_path", type=click.Path(dir_okay=False), default=None,
              help="pyin_f0.npz for display_grid.mora_cleanup's acoustic rules "
              "(default: pyin_f0.npz next to --f0, if present).")
@click.option("--seed", type=int, default=GAME_SEED, show_default=True,
              help="RNG seed for GAME's D3PM segmentation sampler. GAME is "
              "stochastic at inference; output is only reproducible "
              "from its recorded seed.")
@click.option("--deterministic-algorithms", is_flag=True, default=False,
              help="Additionally demand bitwise-deterministic CUDA kernels. "
              "Not required for reproducibility here (the seed alone suffices "
              "on sm_120); costs speed and can fail on unsupported ops.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
def main(vocals: str, fallback_midi: str, f0_path: str, aligned_path: str,
         bpm_file: str, language: str, seed: int, deterministic_algorithms: bool,
         pitch_patch_path: str | None, rms_segments_path: str | None,
         pyin_path: str | None, out_path: str) -> None:
    vocals_p = Path(vocals).resolve()
    with tempfile.TemporaryDirectory(prefix="game-chain-") as tmp:
        tmp_p = Path(tmp)
        # GAME writes <stem>.mid next to a COPY of the input inside tmp, so the
        # song dir is never polluted with generically named outputs.
        wav = tmp_p / vocals_p.name
        wav.symlink_to(vocals_p)
        # NOT `infer.py` directly: the launcher seeds torch/numpy/random in
        # the GAME venv before the D3PM sampler runs. Same argv otherwise.
        game_cmd = [str(GAME_PY), str(GAME_SEEDED_LAUNCHER),
                    "--game-dir", str(GAME_DIR), "--seed", str(seed)]
        if deterministic_algorithms:
            game_cmd.append("--deterministic-algorithms")
        game_cmd += ["--", "extract", str(wav),
                     "-m", str(GAME_MODEL), "-l", language,
                     "--output-formats", "mid",
                     "--seg-threshold", GAME_SEG_THRESHOLD]
        _run(game_cmd, cwd=GAME_DIR)
        game_mid = wav.with_suffix(".mid")
        pf_mid = tmp_p / "postfix.mid"
        union_mid = tmp_p / "union.mid"
        _run([sys.executable, str(SCRIPTS / "score_note_postfix.py"),
              "--midi", str(game_mid), "--f0", f0_path, "--aligned", aligned_path,
              "--out", str(pf_mid), *POSTFIX_FLAGS])
        _run([sys.executable, str(SCRIPTS / "melody_union.py"),
              "--primary", str(pf_mid), "--fallback", fallback_midi,
              "--out", str(union_mid)])
        # keep the REAL-TIME union melody next to the display MIDI — the
        # portrait grid (and any other real-time consumer) reads it directly
        # instead of un-warping the display timeline
        shutil.copyfile(union_mid, Path(out_path).with_suffix(".union.mid"))
        # standardized display grid: fixed quarter width, fixed
        # mora/phrase gaps, fixed page span; sync via the warp sidecar that
        # render_mp4 --time-warp consumes
        warp_path = str(Path(out_path).with_suffix(".warp.json"))
        grid_args = [sys.executable, str(SCRIPTS / "make_display_grid.py"),
                     "--midi", str(union_mid), "--bpm-file", bpm_file,
                     "--aligned", aligned_path,
                     "--out-midi", out_path, "--out-warp", warp_path]
        # Pass the canonical display grid as EXPLICIT flags (snake_case key ->
        # --kebab-case) instead of leaning on make_display_grid's click
        # defaults happening to match versions.json — config is the SoT.
        for key, val in _PROFILE.get("display_grid", {}).items():
            if key == "comment":
                continue
            grid_args += ["--" + key.replace("_", "-"), str(val)]
        if _PROFILE.get("skin", {}).get("ligatures"):
            grid_args += ["--out-ligatures",
                          str(Path(out_path).with_suffix(".ligatures.json"))]
        if _PROFILE.get("display_grid", {}).get("mora_cleanup"):
            # acoustic evidence for the mora-aware cleanup (all three or none)
            pyin = Path(pyin_path) if pyin_path else Path(f0_path).with_name("pyin_f0.npz")
            if pyin.exists():
                grid_args += ["--vocals", vocals, "--rmvpe-f0", f0_path,
                              "--pyin-f0", str(pyin)]
            else:
                click.echo(f"[game-chain] WARNING: {pyin} missing -- mora cleanup "
                           "runs merges only (no acoustic drops)")
        if pitch_patch_path:
            grid_args += ["--pitch-patch", pitch_patch_path]
        if rms_segments_path:
            grid_args += ["--rms-segments", rms_segments_path]
        _run(grid_args)
    click.echo(f"[game-chain] wrote {out_path} (+ {warp_path}; render with --time-warp) "
               f"[game_seed={seed}]")


if __name__ == "__main__":
    main()
