#!/usr/bin/env python3
"""Genre router (survey K1): detect演歌-melisma / rap OOD from cheap acoustic +
timing features and emit per-mode pipeline-config overrides, so the canonical
J-pop chain degrades gracefully on曲風 it was never tuned for.

Design (survey): the right move is NOT to swap models (SOTA systems filter rap
out rather than solve it) but to route to a mode that turns OFF the post-process
that *backfires* on that曲風. Worst case the router mis-fires to the default
J-POP mode = exactly today's behaviour, so adopting it can only help.

Features (computed from rmvpe_f0.npz + aligned_midi.json — already in outputs/):
  * port_density: fraction of voiced frame-pairs with a 20-300 cents/frame glide
    (こぶし / portamento / melisma). J-POP baseline (this repo's 6 songs): 0.22-0.31.
  * syllable_rate: sung chars per second of sung time. J-POP baseline: 2.2-3.4.

Modes & overrides:
  * enka  (port_density > ENKA_PORT): melisma-heavy. Relax GAME note min-dur,
    widen the offset cap, DON'T merge same-pitch / interior-snap (legato runs
    are real notes), allow longer line_end_repair tails.
  * rap   (syllable_rate > RAP_RATE and port_density < RAP_PORT): fast / spoken,
    little pitch. Lean on a beat/tatum grid + mora-per-beat prior; relax or skip
    line_end_repair (no sustained tails); pitch bars are unreliable -> de-emphasize.
  * jpop  (default): the current canonical behaviour, unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np

ENKA_PORT = 0.40   # J-POP tops out ~0.31; enka's kobushi pushes well past this
RAP_RATE = 5.0     # J-POP tops out ~3.4 chars/s; rap is markedly faster
RAP_PORT = 0.18    # rap carries little sustained pitch movement


def port_density(f0: np.ndarray) -> float:
    v = f0 > 0
    if v.sum() < 2:
        return 0.0
    cents = np.zeros_like(f0)
    cents[v] = 1200 * np.log2(f0[v] / 440 + 1e-9)
    dc = np.abs(np.diff(cents))
    vv = v[1:] & v[:-1]
    return float(np.mean((dc[vv] > 20) & (dc[vv] < 300))) if vv.sum() else 0.0


def syllable_rate(aligned: list[dict]) -> float:
    chars = sum(len([c for t in l.get("tokens", []) for c in t.get("chars", [])
                     if c["end"] > c["start"]]) for l in aligned)
    sung = sum(max(0.0, l["end"] - l["start"]) for l in aligned if l.get("end"))
    return chars / sung if sung else 0.0


def classify(port: float, rate: float) -> str:
    if port > ENKA_PORT:
        return "enka"
    if rate > RAP_RATE and port < RAP_PORT:
        return "rap"
    return "jpop"


MODE_OVERRIDES = {
    "jpop": {},  # canonical default, unchanged
    "enka": {
        "game_min_dur": 0.06,          # keep short melisma sub-notes (default absorbs <0.1)
        "merge_same_pitch": False,
        "interior_snap": False,
        "line_end_decay_db": 30.0,     # longer sustained tails
        "offset_cap_extra": 0.3,
    },
    "rap": {
        "use_beat_grid": True,
        "mora_per_beat_prior": True,
        "line_end_repair": False,      # no sustained vowel tails to chase
        "pitch_bar_confidence": "low",
    },
}


def route(f0_npz: str | Path, aligned_json: str | Path) -> dict:
    d = np.load(f0_npz)
    port = port_density(d["f0"])
    aligned = json.loads(Path(aligned_json).read_text(encoding="utf-8"))
    rate = syllable_rate(aligned)
    mode = classify(port, rate)
    return {"mode": mode, "port_density": round(port, 3),
            "syllable_rate": round(rate, 2), "overrides": MODE_OVERRIDES[mode]}


@click.command()
@click.option("--f0", "f0_npz", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--aligned", "aligned_json", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--json-out", default=None)
def main(f0_npz, aligned_json, json_out):
    r = route(f0_npz, aligned_json)
    click.echo(f"[genre-router] mode={r['mode']} port_density={r['port_density']} "
               f"syllable_rate={r['syllable_rate']} overrides={r['overrides']}")
    if json_out:
        Path(json_out).write_text(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
