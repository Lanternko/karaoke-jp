#!/usr/bin/env python3
"""UltraStar .txt outputs -> harness-format note JSON for the vocadito control.

Only the note axes (COn/COnP/COnPOff) matter for the language-confound test, so
we build just pred notes (no morae / no L: there is no MMS aligner for English
and L is not what the confound question is about).

Octave convention: same +48 as the Kiritan run (this UltraSinger build writes
midi = ultrastar_note + 48; see kiritan/ultrasinger/build_pred.py docstring).

  python build_pred_vocadito.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from parse_ultrastar import parse_ultrastar  # noqa: E402

OUT_DIR = HERE / "out"
US_MIDI_OFFSET = 48
PITCHED = {"normal", "golden"}


def find_txt(song_dir: Path):
    txts = [t for t in sorted(song_dir.rglob("*.txt")) if not t.name.startswith(".")]
    return min(txts, key=lambda p: len(p.name)) if txts else None


def main() -> None:
    lang = json.loads((HERE / "clip_lang.json").read_text())
    pred: dict[str, list] = {}
    for tid in sorted(lang, key=int):
        d = OUT_DIR / tid
        txt = find_txt(d) if d.is_dir() else None
        if txt is None:
            print(f"[{tid}] NO OUTPUT")
            continue
        try:
            song = parse_ultrastar(txt.read_text(encoding="utf-8-sig"))
        except UnicodeDecodeError:
            song = parse_ultrastar(txt.read_text(encoding="cp1252"))
        notes = [[round(n["t_on"], 6), round(n["t_off"], 6), n["pitch_raw"] + US_MIDI_OFFSET]
                 for n in song["notes"]
                 if n["type"] in PITCHED and "t_on" in n]
        pred[tid] = notes
        print(f"[{tid}] {len(notes):4d} notes  (lang {lang[tid]['language']})")
    (HERE / "vocadito_pred.json").write_text(json.dumps(pred, indent=1))
    print(f"\nwritten vocadito_pred.json ({len(pred)} clips, "
          f"{sum(len(v) for v in pred.values())} notes)")


if __name__ == "__main__":
    main()
