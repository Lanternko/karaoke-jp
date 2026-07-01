#!/usr/bin/env python3
"""Build the Itako note-transcription ground truth from the public labels.

Source labels (free): https://github.com/mmorise/itako_singing
  - midi_label/itakoNN.mid   Melodyne-assisted note transcription (timing in sec)
  - mono_label/itakoNN.lab   phoneme boundaries, HTK 100-ns integer units

Unlike Kiritan, the Itako `midi_label` encodes **breaths as real note events**
at an out-of-range pitch (the musicXML `/br/` lyric must carry a duration, see
the upstream README "ブレスの扱い"). Those breath notes are NOT sung pitches and
would corrupt a note-transcription GT. The breath sentinel pitch *varies per
song* (e.g. 81 / 47 / 89 / 48), so a global pitch threshold cannot remove them.

We instead drop any MIDI note whose duration overlaps a `br` / `pau` / `sil`
phoneme span in `mono_label` by >50%. The MIDI and mono labels share the
recording's clock, so this matches the breath notes near 1:1 (validated:
itako01 17 br phones -> 17 br notes, etc.).

Output: {song_id: [[onset_sec, offset_sec, midi_pitch_float], ...]} — the same
schema midi_to_json.py / evaluate.py consume for Kiritan & MIR-ST500.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mido

HTK_UNITS_PER_SEC = 1e7  # mono_label times are in 100-ns units
DROP_PHONES = {"br", "pau", "sil"}
OVERLAP_FRAC = 0.5  # drop a note if >this fraction of its duration is in a dropped phone


def midi_notes(path: Path) -> list[list[float]]:
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    # Honor the FULL tempo map: merge_tracks gives a single absolute-ordered
    # stream so set_tempo events (often in track 0) apply to notes in any track,
    # and we update `tempo` as we go. The earlier "read first set_tempo, break"
    # was wrong for the 2 Itako songs (itako03/itako47) with a mid-song tempo
    # change — everything after the change got stretched 1.5-1.7x.
    notes: list[list[float]] = []
    ons: dict[int, float] = {}
    t = 0.0
    tempo = 500000
    for msg in mido.merge_tracks(mid.tracks):
        t += mido.tick2second(msg.time, tpb, tempo)  # delta uses the tempo in force
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "note_on" and msg.velocity:
            ons[msg.note] = t
        elif msg.type == "note_off" or (msg.type == "note_on" and not msg.velocity):
            if msg.note in ons:
                notes.append([ons.pop(msg.note), t, float(msg.note)])
    notes.sort()
    return notes


def dropped_spans(lab_path: Path) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for line in lab_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        s, e, ph = parts
        if ph in DROP_PHONES:
            spans.append((float(s) / HTK_UNITS_PER_SEC, float(e) / HTK_UNITS_PER_SEC))
    return spans


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def clean_notes(notes: list[list[float]], spans: list[tuple[float, float]]):
    kept, dropped = [], []
    for n in notes:
        dur = n[1] - n[0]
        ov = max((overlap(n[0], n[1], s, e) for s, e in spans), default=0.0)
        (dropped if dur > 0 and ov > OVERLAP_FRAC * dur else kept).append(n)
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/kojiek/side_projects/itako/itako_singing",
                    help="Itako label root (has midi_label/ and mono_label/)")
    ap.add_argument("--out", default="itako_gt.json")
    ap.add_argument("--raw-out", default=None,
                    help="optional: also write the unfiltered MIDI notes here")
    args = ap.parse_args()

    src = Path(args.src)
    mid_dir, mono_dir = src / "midi_label", src / "mono_label"

    gt, raw = {}, {}
    n_drop_total = 0
    bad = []
    for f in sorted(mid_dir.glob("*.mid")):
        sid = f.stem
        notes = midi_notes(f)
        raw[sid] = notes
        lab = mono_dir / f"{sid}.lab"
        spans = dropped_spans(lab) if lab.exists() else []
        kept, dropped = clean_notes(notes, spans)
        gt[sid] = kept
        n_drop_total += len(dropped)
        pitches = [int(n[2]) for n in kept]
        # flag songs whose cleaned range still looks non-vocal (sanity)
        if pitches and (min(pitches) < 45 or max(pitches) > 84):
            bad.append((sid, min(pitches), max(pitches)))
        print(f"{sid}: midi={len(notes):3d} -> kept={len(kept):3d} "
              f"(dropped {len(dropped):2d} breath/sil) "
              f"pitch {min(pitches)}..{max(pitches)}")

    Path(args.out).write_text(json.dumps(gt))
    total = sum(len(v) for v in gt.values())
    allp = [int(n[2]) for v in gt.values() for n in v]
    print(f"\n[itako-gt] {len(gt)} songs, {total} sung notes "
          f"(dropped {n_drop_total} breath/sil), pitch {min(allp)}..{max(allp)} -> {args.out}")
    if args.raw_out:
        Path(args.raw_out).write_text(json.dumps(raw))
        print(f"[itako-gt] raw (unfiltered) -> {args.raw_out}")
    if bad:
        print(f"[warn] {len(bad)} songs with out-of-vocal-range pitches after cleaning: {bad}")


if __name__ == "__main__":
    main()
