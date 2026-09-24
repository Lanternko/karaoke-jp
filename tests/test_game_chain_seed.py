"""GAME is stochastic at inference (D3PM sampler). Guard the seed plumbing.

If these break, outputs stop being reproducible and onset-keyed
overrides/<song>_pitch_patch.json entries start silently drifting.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "game_infer_seeded.py"


def test_canonical_profile_pins_a_game_seed():
    data = json.loads((ROOT / "config" / "versions.json").read_text())
    pitch_chain = data["profiles"][data["canonical"]]["pitch_chain"]
    assert "game_seed" in pitch_chain, "canonical profile must pin game_seed"
    assert isinstance(pitch_chain["game_seed"], int)


def test_run_game_chain_routes_game_through_the_seeded_launcher():
    src = (ROOT / "scripts" / "run_game_chain.py").read_text()
    # the raw `infer.py` call must not come back: it leaves the D3PM sampler unseeded
    assert "GAME_SEEDED_LAUNCHER" in src
    assert '"infer.py", "extract"' not in src
    assert "--seed" in src


def test_launcher_seeds_before_importing_torch():
    """Order matters: CUBLAS_WORKSPACE_CONFIG and sys.path must be set before
    torch is imported, and the seeds before infer.py is run."""
    src = LAUNCHER.read_text()
    i_env = src.index("CUBLAS_WORKSPACE_CONFIG")
    i_import = src.index("import torch")
    i_seed = src.index("torch.manual_seed")
    i_run = src.index("runpy.run_path")
    assert i_env < i_import < i_seed < i_run


@pytest.mark.parametrize("flag", ["--game-dir", "--seed"])
def test_launcher_requires_seed_and_game_dir(flag, tmp_path):
    proc = subprocess.run(
        [sys.executable, str(LAUNCHER), flag, "0"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0


def test_launcher_passes_argv_through_verbatim(tmp_path):
    """Everything after `--` must reach infer.py untouched."""
    fake = tmp_path / "infer.py"
    fake.write_text("import sys, json, pathlib\n"
                    "pathlib.Path(sys.argv[-1]).write_text(json.dumps(sys.argv))\n")
    out = tmp_path / "argv.json"
    subprocess.run(
        [sys.executable, str(LAUNCHER), "--game-dir", str(tmp_path), "--seed", "7",
         "--", "extract", "a.wav", "--seg-threshold", "0.3", str(out)],
        check=True, capture_output=True, text=True,
    )
    argv = json.loads(out.read_text())
    # runpy.run_path rewrites argv[0] to the script path; click only uses it for
    # help text, so only argv[1:] has to survive intact.
    assert Path(argv[0]).name == "infer.py"
    assert argv[1:] == ["extract", "a.wav", "--seg-threshold", "0.3", str(out)]
