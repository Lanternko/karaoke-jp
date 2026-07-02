#!/usr/bin/env python3
"""Phase 4: UltraStar .txt outputs -> harness-format JSON.

Produces two files consumed by ultrasinger_eval.py:
  ultrasinger_pred.json   {song: [[t_on, t_off, midi], ...]}  (pitched notes)
  ultrasinger_morae.json  {song: [[t_on, romaji_mora_label], ...]} (one per note)

Octave convention (Phase-2 finding): THIS UltraSinger build writes notes with
`midi = ultrastar_note + 48` (src/.../ultrastar_converter.py: "C4 == 48"),
NOT the +60 the generic UltraStar spec / parse_ultrastar.py assumes. So we take
`pitch_raw + 48` here. (Verified: +48 centres EST-GT pitch delta on 0; +60 puts
it on +12.)

L axis: each pitched note gets a mora label = first mora of its syllable text.
`~` continuation notes carry no text -> inherit the last real syllable's mora
(a held note is the same mora sustained). Un-convertible text -> "?" (L fail).

Run in the fugashi venv (needs syllable_to_mora -> fugashi+UniDic):
  ~/venvs/karaoke-jp-lyrics/bin/python build_pred.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from parse_ultrastar import parse_ultrastar  # noqa: E402
from syllable_to_mora import syllable_to_mora  # noqa: E402

OUT_DIR = HERE / "out"
US_MIDI_OFFSET = 48  # THIS build's convention (see module docstring)
PITCHED = {"normal", "golden"}


def find_txt(song_dir: Path) -> Path | None:
    txts = sorted(song_dir.rglob("*.txt"))
    # UltraSinger may also drop a *_repitched or notes txt; prefer the main one
    # (the one directly named like the song, no extra suffix). Take shortest name.
    txts = [t for t in txts if not t.name.startswith(".")]
    if not txts:
        return None
    return min(txts, key=lambda p: len(p.name))


def main() -> None:
    pred: dict[str, list] = {}
    morae: dict[str, list] = {}
    stats = []
    for i in range(1, 51):
        s = f"{i:02d}"
        song_dir = OUT_DIR / s
        txt = find_txt(song_dir) if song_dir.is_dir() else None
        if txt is None:
            print(f"[{s}] NO OUTPUT — skipped (failed song)")
            continue
        try:
            song = parse_ultrastar(txt.read_text(encoding="utf-8-sig"))
        except UnicodeDecodeError:
            song = parse_ultrastar(txt.read_text(encoding="cp1252"))

        notes_out = []
        morae_out = []
        last_label = "?"
        n_carry = 0
        n_qmark = 0
        for n in song["notes"]:
            if n["type"] not in PITCHED:
                continue
            if "t_on" not in n:
                continue
            midi = n["pitch_raw"] + US_MIDI_OFFSET
            notes_out.append([round(n["t_on"], 6), round(n["t_off"], 6), midi])
            syl = (n["syllable"] or "").strip()
            if syl in ("~", ""):  # held continuation -> carry the last real mora
                label = last_label
                n_carry += 1
            else:
                label = syllable_to_mora(syl)
                if label != "?":
                    last_label = label
                else:
                    n_qmark += 1
            morae_out.append([round(n["t_on"], 6), label])

        pred[s] = notes_out
        morae[s] = morae_out
        stats.append((s, len(notes_out), n_carry, n_qmark))
        print(f"[{s}] {len(notes_out):4d} notes  carry(~)={n_carry:4d}  '?'={n_qmark:3d}")

    (HERE / "ultrasinger_pred.json").write_text(json.dumps(pred, indent=1))
    (HERE / "ultrasinger_morae.json").write_text(
        json.dumps(morae, ensure_ascii=False, indent=1))
    n_songs = len(pred)
    total_notes = sum(x[1] for x in stats)
    total_q = sum(x[3] for x in stats)
    print(f"\nwritten: ultrasinger_pred.json / ultrasinger_morae.json "
          f"({n_songs} songs, {total_notes} notes, {total_q} unconvertible '?')")


if __name__ == "__main__":
    main()
