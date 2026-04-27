"""M1: vocal separation via Mel-Band-RoFormer.

Thin wrapper around `melband-roformer-infer` so the rest of the pipeline can
treat separation as `separate_file(in_path, out_dir) -> (vocals.wav, instrumental.wav)`.

Why a wrapper:

* The upstream package only exposes a folder-based CLI / `proc_folder(args)` —
  it does not have a single-file Python entry point.
* It writes outputs as ``{stem}_vocals.wav`` and ``{stem}_instrumental.wav``;
  the rest of our pipeline expects ``vocals.wav`` and ``instrumental.wav``.
* Model checkpoints + configs need to be downloaded once and cached.

The wrapper handles all three.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "karaoke-jp" / "melband-roformer"


def _ensure_model(slug: str | None, cache_dir: Path) -> tuple[Path, Path]:
    """Download model weights + config if missing. Return (config_path, ckpt_path)."""
    from mel_band_roformer import DEFAULT_MODEL, MODEL_REGISTRY
    from mel_band_roformer.download import download_model_assets

    slug = slug or DEFAULT_MODEL
    entry = MODEL_REGISTRY.get(slug)
    if entry is None:
        raise ValueError(f"Unknown model slug: {slug!r}")

    model_dir = cache_dir / "models" / entry.slug
    ckpt_path = model_dir / entry.checkpoint
    config_path = model_dir / entry.config

    if not ckpt_path.exists() or not config_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        ok = download_model_assets([entry], cache_dir / "models")
        if not ok:
            raise RuntimeError(f"Failed to download model assets for {slug!r}")

    return config_path, ckpt_path


def separate_file(
    input_path: str | Path,
    out_dir: str | Path,
    *,
    model_name: str | None = None,
    device: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Separate one audio file into ``vocals.wav`` + ``instrumental.wav``.

    Returns the two output paths.
    """
    from mel_band_roformer.inference import proc_folder

    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    config_path, ckpt_path = _ensure_model(model_name, cache_dir)

    # The library only takes a folder. Stage the input into a temp dir so we
    # don't accidentally pick up siblings.
    stage_dir = out_dir / ".staging"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    staged_input = stage_dir / input_path.name
    staged_input.symlink_to(input_path)

    args = argparse.Namespace(
        model_type="mel_band_roformer",
        config_path=config_path,
        model_path=ckpt_path,
        input_folder=stage_dir,
        store_dir=out_dir,
        device=device,
        device_ids=None,
    )
    try:
        proc_folder(args)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    stem = input_path.stem
    raw_vocals = out_dir / f"{stem}_vocals.wav"
    raw_instr = out_dir / f"{stem}_instrumental.wav"
    if not raw_vocals.exists() or not raw_instr.exists():
        raise RuntimeError(
            f"Expected outputs missing: {raw_vocals.name}, {raw_instr.name}"
        )

    vocals = out_dir / "vocals.wav"
    instrumental = out_dir / "instrumental.wav"
    raw_vocals.replace(vocals)
    raw_instr.replace(instrumental)
    return vocals, instrumental
