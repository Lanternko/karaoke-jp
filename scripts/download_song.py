"""Fetch audio + (optional) video into a song dir.

This is the M0 stage. Default behaviour: download both the WAV audio and a
720p video from the same YouTube URL, naming them ``source.wav`` and
``background.mp4`` so the rest of the Snakemake pipeline picks them up
without further configuration.

If the YouTube upload is a "Lyric Video" (lyrics burned into the frame),
pass ``--no-video`` and provide your own ``background.png`` / ``.mp4``
instead — the bundled lyrics layer would otherwise collide with the
backdrop.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click


def _yt_dlp(*args: str) -> None:
    cmd = [sys.executable.replace("python", "yt-dlp"), *args]
    # Fall back to PATH lookup if the sibling rename doesn't exist.
    if not Path(cmd[0]).exists():
        cmd[0] = "yt-dlp"
    subprocess.run(cmd, check=True)


@click.command()
@click.argument("url", type=str)
@click.option(
    "--out-dir",
    "-o",
    type=click.Path(file_okay=False),
    required=True,
    help="Song dir, e.g. songs/<song-id>/. Created if missing.",
)
@click.option(
    "--no-video",
    is_flag=True,
    help="Skip background video download. Use when the source is a "
    "lyric-video (lyrics already in the frame).",
)
def main(url: str, out_dir: str, no_video: bool) -> None:
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"[download] audio -> {out}/source.wav", flush=True)
    _yt_dlp(
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        str(out / "source.%(ext)s"),
        url,
    )

    if no_video:
        print("[download] --no-video; skipping background download.", flush=True)
        return

    print(f"[download] video -> {out}/background.mp4 (h264 720p preferred)", flush=True)
    # bv*[vcodec*=avc1][height<=720] picks h264 720p when available so we
    # don't need an extra re-encode pass; format 251 (audio-only opus) is
    # never picked here because we restrict to bv*.
    _yt_dlp(
        "-f",
        "bv*[vcodec*=avc1][height<=720]/bv*[height<=720]",
        "--remux-video",
        "mp4",
        "-o",
        str(out / "background.%(ext)s"),
        url,
    )


if __name__ == "__main__":
    main()
