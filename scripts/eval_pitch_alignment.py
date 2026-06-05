#!/usr/bin/env python3
"""Evaluate one rendered melody MIDI against F0 tracks and lyric windows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from karaoke_jp.pitch_eval import (  # noqa: E402
    F0Track,
    compare_notes_to_f0,
    fragmentation_metrics,
    lyric_char_windows,
    lyric_line_windows,
    metrics_to_dict,
    stable_char_windows,
    transition_char_windows,
)
from karaoke_jp.score_melody import read_midi_notes  # noqa: E402


def _load_aligned(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


@click.command()
@click.option("--midi", "midi_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--rmvpe-f0", type=click.Path(exists=True, dir_okay=False))
@click.option("--pyin-f0", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
def main(
    midi_path: str,
    aligned_path: str,
    rmvpe_f0: str | None,
    pyin_f0: str | None,
    out_path: str,
) -> None:
    notes = read_midi_notes(midi_path)
    aligned = _load_aligned(Path(aligned_path))
    line_windows = lyric_line_windows(aligned)
    char_windows = lyric_char_windows(aligned)
    stable_windows = stable_char_windows(char_windows)
    transition_windows = transition_char_windows(char_windows)

    result: dict[str, object] = {
        "midi": str(Path(midi_path)),
        "aligned": str(Path(aligned_path)),
        "fragmentation": metrics_to_dict(
            fragmentation_metrics(notes, line_windows=line_windows, char_windows=char_windows)
        ),
        "line_windows": len(line_windows),
        "char_windows": len(char_windows),
        "stable_windows": len(stable_windows),
        "transition_windows": len(transition_windows),
        "f0": {},
    }

    f0_inputs = {"rmvpe": rmvpe_f0, "pyin": pyin_f0}
    for name, path in f0_inputs.items():
        if not path:
            continue
        track = F0Track.from_npz(path)
        result["f0"][name] = {
            "all": metrics_to_dict(compare_notes_to_f0(notes, track)),
            "stable": metrics_to_dict(compare_notes_to_f0(notes, track, windows=stable_windows)),
            "transition": metrics_to_dict(compare_notes_to_f0(notes, track, windows=transition_windows)),
        }

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[pitch-eval] wrote {dest}")


if __name__ == "__main__":
    main()
