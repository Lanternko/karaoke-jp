"""Sidecar line-start repair from vocals RMS onset hints.

``midi_timing`` intentionally snaps mora starts to MIDI note onsets.  That is
usually right, but singing onsets can precede detected/quantized note onsets by
hundreds of milliseconds.  This sidecar moves only the first sung char of a line
earlier when:

* MIDI timing is much later than the original ASR/RMS hint;
* a local RMS valley/onset exists between the hint and the MIDI onset;
* the move stays after the previous line boundary guard.

It is an experimental ablation tool, not canonical pipeline behavior.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import click
import numpy as np

_RMS = Path(__file__).resolve().parent / "rms_vad_segments.py"
_spec = importlib.util.spec_from_file_location("_rms_vad", _RMS)
_rms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rms)  # type: ignore[union-attr]
frame_rms_db = _rms.frame_rms_db
DEFAULTS = _rms.DEFAULTS

_LER = Path(__file__).resolve().parent / "line_end_repair.py"
_ler_spec = importlib.util.spec_from_file_location("_line_end_repair", _LER)
_ler = importlib.util.module_from_spec(_ler_spec)
_ler_spec.loader.exec_module(_ler)  # type: ignore[union-attr]
load_audio_mono = _ler.load_audio_mono


def first_sung_char(line: dict) -> tuple[dict, dict] | None:
    for tok in line.get("tokens", []):
        if tok.get("is_punct"):
            continue
        for ch in tok.get("chars", []):
            if ch.get("start") is not None:
                return tok, ch
    return None


def rms_onset_between(
    rms_db: np.ndarray,
    hop_s: float,
    lower: float,
    upper: float,
    *,
    top_db: float,
    sustain: float,
) -> float | None:
    if upper <= lower:
        return None
    lo = max(0, int(math.floor(lower / hop_s)))
    hi = min(len(rms_db) - 1, int(math.ceil(upper / hop_s)))
    if hi <= lo:
        return None

    sustain_frames = max(1, int(math.ceil(sustain / hop_s)))
    valley = lo + int(np.argmin(rms_db[lo : hi + 1]))
    voiced = rms_db > -top_db
    for i in range(valley, hi + 1):
        j = min(len(voiced), i + sustain_frames)
        if j <= i:
            continue
        if bool(np.all(voiced[i:j])):
            return i * hop_s
    return None


def repair(
    lines: list[dict],
    hint_lines: list[dict],
    rms_db: np.ndarray,
    hop_s: float,
    *,
    max_shift: float,
    min_late: float,
    onset_top_db: float,
    onset_sustain: float,
    prev_guard: float,
    blend: float,
    min_move: float,
    skip_first_line: bool,
) -> list[tuple[int, float, float]]:
    changes: list[tuple[int, float, float]] = []
    for i, line in enumerate(lines):
        if skip_first_line and i == 0:
            continue
        target = first_sung_char(line)
        if target is None or i >= len(hint_lines):
            continue
        tok, ch = target
        cur_start = float(ch["start"])
        hint_start = float(hint_lines[i].get("start", cur_start))
        if cur_start - hint_start < min_late:
            continue

        prev_end = float(lines[i - 1].get("end", -1e9)) if i > 0 else -1e9
        lower = max(hint_start, cur_start - max_shift, prev_end + prev_guard)
        candidate = rms_onset_between(
            rms_db,
            hop_s,
            lower,
            cur_start,
            top_db=onset_top_db,
            sustain=onset_sustain,
        )
        if candidate is None:
            continue
        raw_start = max(candidate, lower)
        new_start = round(raw_start + blend * (cur_start - raw_start), 3)
        if cur_start - new_start < min_move:
            continue
        if new_start >= cur_start - 1e-3 or new_start >= float(ch["end"]):
            continue

        ch["start"] = new_start
        tok_start = tok.get("start")
        tok["start"] = new_start if tok_start is None else min(float(tok_start), new_start)
        line["start"] = new_start

        # Leading punctuation should remain zero-duration at the visible line
        # start, not linger before the first sung character.
        for lt in line.get("tokens", []):
            for lch in lt.get("chars", []) or []:
                if lch is ch:
                    break
                if lch.get("start") is not None and float(lch["end"]) <= cur_start:
                    lch["start"] = new_start
                    lch["end"] = new_start
            else:
                continue
            break
        changes.append((i, cur_start, new_start))
    return changes


@click.command()
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--hint-aligned", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--vocals", "vocals_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--max-shift", default=1.2, show_default=True)
@click.option("--min-late", default=0.45, show_default=True)
@click.option("--onset-top-db", default=30.0, show_default=True)
@click.option("--onset-sustain", default=0.096, show_default=True)
@click.option("--prev-guard", default=0.04, show_default=True)
@click.option(
    "--blend",
    default=0.0,
    show_default=True,
    help="0 moves to the RMS onset; larger values keep the start closer to the MIDI onset.",
)
@click.option("--min-move", default=0.0, show_default=True)
@click.option("--skip-first-line/--include-first-line", default=False, show_default=True)
def main(
    aligned_path: str,
    hint_aligned: str,
    vocals_path: str,
    out_path: str,
    max_shift: float,
    min_late: float,
    onset_top_db: float,
    onset_sustain: float,
    prev_guard: float,
    blend: float,
    min_move: float,
    skip_first_line: bool,
) -> None:
    lines = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    hint_lines = json.loads(Path(hint_aligned).read_text(encoding="utf-8"))
    sr = DEFAULTS["sr"]
    y = load_audio_mono(vocals_path, sr)
    rms_db = frame_rms_db(y, DEFAULTS["frame_length"], DEFAULTS["hop_length"])
    hop_s = DEFAULTS["hop_length"] / sr
    changes = repair(
        lines,
        hint_lines,
        rms_db,
        hop_s,
        max_shift=max_shift,
        min_late=min_late,
        onset_top_db=onset_top_db,
        onset_sustain=onset_sustain,
        prev_guard=prev_guard,
        blend=blend,
        min_move=min_move,
        skip_first_line=skip_first_line,
    )
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"moved {len(changes)} line start(s) -> {dest}")
    for idx, old, new in changes:
        print(f"  line {idx:2d}: {old:7.3f} -> {new:7.3f} ({new - old:+.3f}s)")


if __name__ == "__main__":
    main()
