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
@click.option(
    "--backend",
    type=click.Choice(["rmvpe", "some"]),
    default="rmvpe",
    show_default=True,
    help="Pitch extraction backend.",
)
def melody(vocals_path: str, midi_path: str, tempo: float, cuda_device: int, backend: str) -> None:
    """M2: Vocals -> melody MIDI via RMVPE or SOME."""
    from .melody import extract_midi

    extract_midi(
        vocals_path,
        midi_path,
        tempo=tempo,
        backend=backend,
        cuda_device=None if cuda_device < 0 else cuda_device,
    )


@main.command("score-melody")
@click.argument("audio_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--score-midi",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Score MIDI used as pitch ground truth and DTW alignment reference.",
)
@click.option(
    "--out",
    "-o",
    "midi_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Output melody.mid path.",
)
@click.option(
    "--tempo",
    default=None,
    type=float,
    help="Tempo embedded in the output MIDI. Defaults to the first score tempo marker.",
)
@click.option(
    "--top-voice/--all-notes",
    default=True,
    show_default=True,
    help="Emit a best-effort top voice from the score, or keep every score note.",
)
@click.option(
    "--sample-rate",
    default=22050,
    type=int,
    show_default=True,
    help="Audio sample rate used for chroma extraction.",
)
@click.option(
    "--hop-length",
    default=1024,
    type=int,
    show_default=True,
    help="Hop length used for score/audio chroma DTW.",
)
def score_melody(
    audio_path: str,
    score_midi: str,
    midi_path: str,
    tempo: float | None,
    top_voice: bool,
    sample_rate: int,
    hop_length: int,
) -> None:
    """M2-score: score MIDI pitch + piano audio timing via chroma DTW."""
    from .score_melody import extract_score_aligned_melody

    extract_score_aligned_melody(
        audio_path,
        score_midi,
        midi_path,
        top_voice=top_voice,
        sample_rate=sample_rate,
        hop_length=hop_length,
        tempo=tempo,
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
