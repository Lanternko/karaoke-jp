"""M2: vocals.wav -> melody.mid via openvpi/SOME.

We shell out to ``third_party/SOME/infer.py`` running inside the dedicated
``~/venvs/karaoke-jp-melody/`` interpreter so SOME's pinned legacy stack
(``librosa<0.10``, ``numpy<2``, ``lightning``, etc.) does not collide with
the main karaoke-jp venv.

The SOME repo / checkpoint is expected at::

    third_party/SOME/
    third_party/SOME/pretrained/0119_continuous256_5spk/
        config.yaml
        model_ckpt_steps_100000_simplified.ckpt
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOME_DIR = PROJECT_ROOT / "third_party" / "SOME"
DEFAULT_SOME_CKPT = (
    DEFAULT_SOME_DIR / "pretrained" / "0119_continuous256_5spk" /
    "model_ckpt_steps_100000_simplified.ckpt"
)
DEFAULT_SOME_PYTHON = Path.home() / "venvs" / "karaoke-jp-melody" / "bin" / "python"


def extract_midi(
    vocals_path: str | Path,
    midi_path: str | Path,
    *,
    tempo: float = 120.0,
    some_dir: Path | None = None,
    some_ckpt: Path | None = None,
    some_python: Path | None = None,
    cuda_device: int | None = 0,
) -> Path:
    """Run SOME on ``vocals_path`` and write a MIDI file to ``midi_path``.

    Returns the resolved MIDI path.
    """
    vocals_path = Path(vocals_path).resolve()
    if not vocals_path.is_file():
        raise FileNotFoundError(vocals_path)

    midi_path = Path(midi_path).resolve()
    midi_path.parent.mkdir(parents=True, exist_ok=True)

    some_dir = some_dir or DEFAULT_SOME_DIR
    some_ckpt = some_ckpt or DEFAULT_SOME_CKPT
    some_python = some_python or DEFAULT_SOME_PYTHON

    if not some_python.exists():
        raise FileNotFoundError(
            f"SOME python interpreter not found at {some_python}. "
            "Set up ~/venvs/karaoke-jp-melody (see README)."
        )
    if not some_ckpt.exists():
        raise FileNotFoundError(
            f"SOME checkpoint not found at {some_ckpt}. Run the M2 setup."
        )

    # Build a *minimal* env for the child Python so the parent venv's
    # PYTHONPATH / PYTHONHOME / VIRTUAL_ENV / SYS.PATH overrides do not leak
    # in and weaken the dedicated-venv isolation. Pass through only what the
    # subprocess legitimately needs (PATH for ffmpeg/ffprobe, locale for
    # filename handling, HOME for pooch / huggingface caches, LD_LIBRARY_PATH
    # for CUDA shims).
    parent = os.environ
    passthrough = (
        "PATH", "HOME", "USER", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE",
        "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH",
    )
    env = {k: parent[k] for k in passthrough if k in parent}
    env["CUDA_VISIBLE_DEVICES"] = "" if cuda_device is None else str(cuda_device)

    cmd = [
        str(some_python),
        "infer.py",
        "--model",
        str(some_ckpt),
        "--wav",
        str(vocals_path),
        "--midi",
        str(midi_path),
        "--tempo",
        str(tempo),
    ]
    subprocess.run(cmd, cwd=some_dir, env=env, check=True)
    return midi_path
