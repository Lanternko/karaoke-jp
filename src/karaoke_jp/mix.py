"""M5: Mix instrumental + vocals at a given vocal volume ratio.

Usage
-----
    from karaoke_jp.mix import mix_vocals
    mix_vocals("instrumental.wav", "vocals.wav", "mixed.wav", vocal_ratio=0.30)

The output is a stereo wav suitable for handing directly to MID2BAR-Player as
the ``--audio`` argument.  The instrumental is kept at full volume; vocals are
attenuated by *vocal_ratio* (0.0 = pure instrumental, 1.0 = equal mix).
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def mix_vocals(
    instrumental_path: str | Path,
    vocals_path: str | Path,
    out_path: str | Path,
    *,
    vocal_ratio: float = 0.30,
) -> Path:
    """Blend instrumental + vocals and write a WAV file.

    Parameters
    ----------
    instrumental_path:
        Path to the instrumental (accompaniment) WAV.
    vocals_path:
        Path to the separated vocals WAV.
    out_path:
        Destination path for the mixed WAV.
    vocal_ratio:
        0.0 → pure instrumental, 1.0 → equal loudness, 0.3 → 30 % vocal.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    instrumental_path = Path(instrumental_path).resolve()
    vocals_path = Path(vocals_path).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not instrumental_path.is_file():
        raise FileNotFoundError(instrumental_path)
    if not vocals_path.is_file():
        raise FileNotFoundError(vocals_path)
    if not (0.0 <= vocal_ratio <= 1.0):
        raise ValueError(f"vocal_ratio must be in [0, 1], got {vocal_ratio}")

    # amix normalises by default (divides by number of inputs).  Use
    # weights=1:ratio and normalize=0 to keep the instrumental at full
    # amplitude and only attenuate the vocal track.
    filter_graph = (
        f"[0:a]volume=1.0[bg];"
        f"[1:a]volume={vocal_ratio:.6f}[voc];"
        f"[bg][voc]amix=inputs=2:duration=longest:normalize=0[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(instrumental_path),
        "-i", str(vocals_path),
        "-filter_complex", filter_graph,
        "-map", "[out]",
        "-ac", "2",          # stereo
        "-ar", "44100",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path
