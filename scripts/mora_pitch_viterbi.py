#!/usr/bin/env python3
"""Latent-note decoding over the aligned mora grid.

Implements the survey-identified gap directly: instead of rounding a
frame-wise F0 median per mora, each mora window gets a small candidate set
and a Viterbi pass chooses the sequence that best explains

- RMVPE F0 support with late-window emphasis (shakuri/portamento robust),
- BasicPitch note events as a second acoustic opinion,
- the LOCAL accompaniment chroma (per-window chord context, not a global
  scale mask — keeps real leading tones like the verse B naturals alive),
- interval smoothness between neighbouring morae in the same line.

Audio-side evidence only; no reference/gold files are read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import refit_melody_to_mora as rmm  # noqa: E402
from bp_hybrid_relabel import PITCH_HI, PITCH_LO, load_events as load_basicpitch_events  # noqa: E402

from karaoke_jp.melody import _write_midi  # noqa: E402
from karaoke_jp.pitch_eval import F0Track  # noqa: E402
from karaoke_jp.score_melody import read_first_tempo_bpm  # noqa: E402


class LocalChroma:
    def __init__(self, audio_path: Path):
        import librosa

        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        self.chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
        self.hop = 512 / 22050

    def weights(self, start: float, end: float) -> np.ndarray:
        i0 = max(0, int(start / self.hop))
        i1 = max(i0 + 1, int(end / self.hop))
        w = self.chroma[:, i0:i1].mean(axis=1)
        peak = w.max()
        return w / peak if peak > 0 else np.full(12, 1.0)


def window_evidence(track: F0Track, ws: float, we: float, late_frac: float = 0.4):
    trim = min(0.04, (we - ws) * 0.25)
    mask = (track.times >= ws + trim) & (track.times < we - trim)
    times = track.times[mask]
    midi = track.midi[mask]
    voiced = np.isfinite(midi)
    times, midi = times[voiced], midi[voiced]
    if midi.size == 0:
        return times, midi, np.array([])
    late_start = we - (we - ws) * late_frac
    weights = np.where(times >= late_start, 2.0, 1.0)
    return times, midi, weights


def decode(
    windows,
    track: F0Track,
    bp_events,
    chroma: LocalChroma | None,
    *,
    w_support: float = 2.0,
    w_dev: float = 0.8,
    w_bp: float = 0.9,
    w_chroma: float = 0.6,
    w_trans: float = 0.06,
    min_voiced_ratio: float = 0.25,
):
    hop = track.hop_seconds or 0.01
    entries = []  # (ws, we, line, [(pitch, emission)])
    for ws, we, _kana, line in windows:
        if we - ws <= 0.02:
            continue
        times, midi, fw = window_evidence(track, ws, we)
        dur = we - ws
        voiced_ratio = midi.size * hop / dur if dur > 0 else 0.0

        overlapping = [
            (s, e, p, conf)
            for s, e, p, conf in bp_events
            if min(e, we) - max(s, ws) >= 0.25 * min(dur, e - s)
        ]
        cands: set[int] = set()
        wmed = None
        if midi.size >= 2:
            order = np.argsort(midi)
            cum = np.cumsum(fw[order])
            wmed = float(midi[order][np.searchsorted(cum, cum[-1] / 2)])
            cands.update({int(round(wmed)) - 1, int(round(wmed)), int(round(wmed)) + 1})
            late = midi[times >= we - dur * 0.4]
            if late.size:
                cands.add(int(round(np.median(late))))
        for _s, _e, p, _c in overlapping:
            cands.add(p)
        cands = {p for p in cands if PITCH_LO <= p <= PITCH_HI}
        if not cands or (voiced_ratio < min_voiced_ratio and not overlapping):
            entries.append((ws, we, line, []))
            continue

        cw = chroma.weights(ws - 0.15, we + 0.15) if chroma else None
        scored = []
        for p in sorted(cands):
            if midi.size:
                support = float(np.sum(fw[np.abs(midi - p) <= 0.6]) / np.sum(fw))
                dev = min(abs(wmed - p), 2.0) if wmed is not None else 2.0
            else:
                support, dev = 0.0, 2.0
            bp = 0.0
            for s, e, ep, conf in overlapping:
                if ep == p:
                    bp = max(bp, conf * min(1.0, (min(e, we) - max(s, ws)) / dur))
            ch = float(cw[p % 12]) if cw is not None else 0.0
            emission = w_support * support - w_dev * dev + w_bp * bp + w_chroma * ch
            scored.append((p, emission))
        entries.append((ws, we, line, scored))

    # Viterbi over emitted windows; transitions only inside a line and
    # across gaps shorter than 0.6 s.
    notes: list[tuple[float, float, int]] = []
    best_prev: dict[int, float] = {}
    prev_meta: tuple[float, int] | None = None  # (end, line)
    back: list[dict[int, tuple[float, int | None]]] = []
    seq_entries = []
    for ws, we, line, scored in entries:
        if not scored:
            continue
        linked = (
            prev_meta is not None
            and line == prev_meta[1]
            and ws - prev_meta[0] < 0.6
            and best_prev
        )
        column: dict[int, tuple[float, int | None]] = {}
        for p, emission in scored:
            if linked:
                choices = [
                    (score + emission - w_trans * min(abs(p - q), 12), q)
                    for q, score in best_prev.items()
                ]
                column[p] = max(choices)
            else:
                column[p] = (emission, None)
        back.append(column)
        seq_entries.append((ws, we))
        best_prev = {p: sc for p, (sc, _q) in column.items()}
        prev_meta = (we, line)

    # backtrack
    path: list[int] = []
    nxt: int | None = None
    for column in reversed(back):
        if nxt is None or nxt not in column:
            nxt = max(column, key=lambda p: column[p][0])
        path.append(nxt)
        nxt = column[nxt][1]
    path.reverse()
    for (ws, we), p in zip(seq_entries, path):
        notes.append((ws, we, p))
    return notes


@click.command()
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--f0", "f0_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--basicpitch-csv", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--chroma-audio", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--tempo-from-midi", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
def main(aligned_path, f0_path, basicpitch_csv, chroma_audio, tempo_from_midi, out_path):
    track = F0Track.from_npz(f0_path)
    aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    windows = rmm._mora_windows(aligned)
    bp_events = load_basicpitch_events(Path(basicpitch_csv))
    chroma = LocalChroma(Path(chroma_audio)) if chroma_audio else None
    notes = decode(windows, track, bp_events, chroma)
    tempo = read_first_tempo_bpm(Path(tempo_from_midi)) if tempo_from_midi else 120.0
    _write_midi(notes, Path(out_path), tempo=tempo)
    print(f"[mora-pitch-viterbi] wrote {out_path} notes={len(notes)}")


if __name__ == "__main__":
    main()
