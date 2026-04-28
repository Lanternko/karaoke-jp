"""karaoke-jp top-level CLI dispatcher."""
from __future__ import annotations

import click

from . import __version__


@click.group()
@click.version_option(__version__)
def main() -> None:
    """karaoke-jp — JOYSOUND-style Japanese karaoke video generator."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--out-dir",
    "-o",
    type=click.Path(file_okay=False),
    required=True,
    help="Directory where vocals.wav and instrumental.wav are written.",
)
@click.option("--model", default=None, help="Override model name (default: KJ Kim).")
@click.option("--device", default="cuda", help="Torch device (cuda / mps / cpu).")
def separate(input_path: str, out_dir: str, model: str | None, device: str) -> None:
    """M1: Vocal separation via Mel-Band-RoFormer."""
    from .separate import separate_file

    separate_file(input_path, out_dir, model_name=model, device=device)


@main.command()
@click.argument("vocals_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--out",
    "-o",
    "midi_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Output melody.mid path.",
)
@click.option("--tempo", default=120.0, type=float, help="Tempo embedded in the MIDI file.")
@click.option(
    "--cuda-device",
    default=0,
    type=int,
    help="CUDA device index. Pass -1 to force CPU.",
)
def melody(vocals_path: str, midi_path: str, tempo: float, cuda_device: int) -> None:
    """M2: Vocals -> melody MIDI via openvpi/SOME."""
    from .melody import extract_midi

    extract_midi(
        vocals_path,
        midi_path,
        tempo=tempo,
        cuda_device=None if cuda_device < 0 else cuda_device,
    )


@main.command()
@click.argument("instrumental_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("vocals_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--out", "-o", "out_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Output mixed WAV path.",
)
@click.option(
    "--vocal-ratio",
    default=0.20,
    type=float,
    show_default=True,
    help="Vocal volume as a fraction of instrumental (0.0=mute, 1.0=equal).",
)
def mix(
    instrumental_path: str,
    vocals_path: str,
    out_path: str,
    vocal_ratio: float,
) -> None:
    """M5: Blend instrumental + vocals at VOCAL_RATIO (default 20 %)."""
    from .mix import mix_vocals

    mix_vocals(instrumental_path, vocals_path, out_path, vocal_ratio=vocal_ratio)
    print(f"[mix] wrote {out_path}")


if __name__ == "__main__":
    main()
