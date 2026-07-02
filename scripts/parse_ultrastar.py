#!/usr/bin/env python3
"""Parse UltraStar .txt (USDB community karaoke format) into note-level GT JSON.

Output schema per song: header metadata + notes[{t_on, t_off, midi, syllable, type, line}]
— the exact dual-axis shape COnPOff+L consumes (note pitch + syllable timing).

Format essentials (https://usdx.eu / community spec):
  #KEY:value headers; BPM may use comma decimals; GAP in ms.
  Note lines:  TYPE START DUR PITCH TEXT   (beats; pitch 0 = C4 → MIDI 60)
  ': ' normal, '*' golden, 'F' freestyle (no pitch), 'R' rap, 'G' rap-golden
  '- BEAT [BEAT]' line break, 'P1/P2/P3' duet voices, 'E' end.
  Beat → seconds: GAP/1000 + beat * 60 / (BPM * 4).
  #RELATIVE:yes songs use per-line relative beats (handled below).
"""

import argparse
import json
import re
import sys
from pathlib import Path

NOTE_TYPES = {":": "normal", "*": "golden", "F": "freestyle", "R": "rap", "G": "rap_golden"}
PITCHED_TYPES = {"normal", "golden"}  # freestyle/rap carry no usable pitch


def _to_float(s):
    return float(s.replace(",", "."))


def parse_ultrastar(text):
    header = {}
    notes = []
    warnings = []
    voice = "P1"
    line_idx = 0
    rel_offset = 0.0  # beat offset for RELATIVE mode

    for raw in text.splitlines():
        line = raw.strip("﻿").rstrip()
        if not line.strip():
            continue
        s = line.strip()
        if s.startswith("#"):
            k, _, v = s[1:].partition(":")
            header[k.strip().upper()] = v.strip()
            continue
        if s == "E":
            break
        if re.fullmatch(r"P\s*\d+", s):
            voice = s.replace(" ", "")
            rel_offset = 0.0
            continue
        if s.startswith("-"):
            m = re.match(r"-\s*(-?\d+)(?:\s+(-?\d+))?(?:\s+(.*))?$", s)
            if m:
                line_idx += 1
                if header.get("RELATIVE", "").lower() == "yes":
                    rel_offset += _to_float(m.group(2) or m.group(1))
                # some community files jam a note after the break on the same line
                if m.group(3) and m.group(3)[:1] in NOTE_TYPES:
                    s = m.group(3)
                else:
                    continue
            else:
                warnings.append(f"unparsed break line: {raw!r}")
                continue
        # '*:' is a community deviation for golden notes; tolerate the extra colon
        m = re.match(r"([:*FRG]):?\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s?(.*)$", s)
        if not m:
            warnings.append(f"unparsed line: {raw!r}")
            continue
        t, start, dur, pitch, syl = m.groups()
        notes.append({
            "voice": voice,
            "line": line_idx,
            "type": NOTE_TYPES[t],
            "beat": int(start) + rel_offset,
            "dur_beats": int(dur),
            "pitch_raw": int(pitch),
            "syllable": syl,  # keep trailing/leading spaces: they are word separators
        })

    bpm = _to_float(header.get("BPM", "0") or "0")
    gap_ms = _to_float(header.get("GAP", "0") or "0")
    if bpm <= 0:
        warnings.append("missing/invalid BPM — no absolute times")
        spb = None
    else:
        spb = 60.0 / (bpm * 4.0)  # seconds per UltraStar beat

    for n in notes:
        if spb is not None:
            n["t_on"] = round(gap_ms / 1000.0 + n["beat"] * spb, 6)
            n["t_off"] = round(n["t_on"] + n["dur_beats"] * spb, 6)
        n["midi"] = n["pitch_raw"] + 60 if n["type"] in PITCHED_TYPES else None

    return {
        "header": header,
        "bpm": bpm,
        "gap_ms": gap_ms,
        "n_notes": len(notes),
        "n_pitched": sum(1 for n in notes if n["midi"] is not None),
        "duet": len({n["voice"] for n in notes}) > 1,
        "relative": header.get("RELATIVE", "").lower() == "yes",
        "notes": notes,
        "warnings": warnings,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("txt", nargs="+", help="UltraStar .txt file(s)")
    ap.add_argument("--out", help="output JSON path (single input) or directory")
    ap.add_argument("--summary", action="store_true", help="print one-line summary only")
    args = ap.parse_args()

    for path in args.txt:
        p = Path(path)
        # community files are a codepage lottery; UTF-8 first, then cp1252
        try:
            text = p.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            text = p.read_text(encoding="cp1252")
        song = parse_ultrastar(text)
        if args.summary:
            h = song["header"]
            span = ""
            if song["notes"] and "t_on" in song["notes"][0]:
                span = f" span {song['notes'][0]['t_on']:.1f}-{song['notes'][-1]['t_off']:.1f}s"
            print(f"{p.name}: {h.get('ARTIST','?')} - {h.get('TITLE','?')} | "
                  f"{song['n_notes']} notes ({song['n_pitched']} pitched){span} | "
                  f"BPM {song['bpm']} GAP {song['gap_ms']}ms | "
                  f"duet={song['duet']} rel={song['relative']} warn={len(song['warnings'])}")
            continue
        if args.out:
            out = Path(args.out)
            out_path = out / (p.stem + ".json") if out.is_dir() else out
        else:
            out_path = p.with_suffix(".json")
        out_path.write_text(json.dumps(song, ensure_ascii=False, indent=1))
        print(f"{p.name} -> {out_path} ({song['n_notes']} notes, {len(song['warnings'])} warnings)")
        for w in song["warnings"][:5]:
            print(f"  warn: {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
