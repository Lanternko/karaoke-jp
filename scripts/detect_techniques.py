"""Detect the original singer's techniques (DAM 精密採点 vocabulary) from F0.

Offline render can't score the user's singing, so the HUD instead reports what
the *original vocal* does, using the four techniques DAM 精密採点 counts live
(DX-G/Ai) with DAM's colour convention (orange しゃくり, blue こぶし,
purple フォール, green ビブラート):

  shakuri  (しゃくり)  note starts below target and slides up into it
  kobushi  (こぶし)    short up-and-back turn inside a held note
  fall     (フォール)  note is held on pitch then slides down as the voice ends
  vibrato  (ビブラート) periodic 4.5-8 Hz pitch oscillation held >= 0.5 s

Detection runs on the REAL-TIME melody (``<display>.union.mid``, the GAME
union notes run_game_chain keeps next to the display MIDI): the display grid's
constant-speed slots split/merge notes, so its bar edges don't sit on the
sung pitch changes. Each event carries its real time and, via the time warp,
its display time (what the HUD compares against). Thresholds are heuristics, NOT fitted to DAM (whose detector is closed):
treat counts as "what this renderer calls a technique", not a score.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pretty_midi
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

# --- thresholds (semitones / seconds) ---------------------------------------
SHAKURI_WIN = 0.25        # scoop must resolve within this much of note start
SHAKURI_MIN_DEPTH = 0.5   # starts at least this far below target...
SHAKURI_MAX_DEPTH = 4.0   # ...but not so far it is really another note
ON_PITCH = 0.35           # "on pitch" tolerance
SHAKURI_MIN_GLIDE = 0.03  # the rise must take time (a jump is a new note)
FALL_MIN_DROP = 0.7
FALL_MIN_HOLD = 0.10      # must be on pitch this long before falling
FALL_MIN_GLIDE = 0.05     # the descent itself must last this long (not a dive)
FALL_EDGE = 2             # offset frames ignored (RMVPE end dive)
VIB_RATE = (4.5, 8.5)     # Hz
VIB_MIN_P2P = 0.25        # peak-to-peak extent
VIB_MIN_DUR = 0.4
VIB_MIN_HALVES = 3       # half-cycles
KOB_MIN_PROM = 0.4
KOB_WIDTH = (0.06, 0.25)
KOB_MIN_NOTE = 0.20       # kobushi only inside a bar held at least this long
KOB_GUARD = 0.05          # ...and away from the bar's own edges
EDGE = 3                  # frames trimmed at voicing on/offset (RMVPE dives)


def _load_f0(path: str):
    d = np.load(path)
    f0 = d["f0"].astype(float)
    hop = float(np.asarray(d["hop_seconds"]).ravel()[0])
    midi = np.full_like(f0, np.nan)
    v = f0 > 0
    midi[v] = 69 + 12 * np.log2(f0[v] / 440.0)
    # 3-frame median kills single-frame RMVPE glitches, keeps NaN gaps.
    filled = np.where(np.isnan(midi), 0.0, midi)
    med = median_filter(filled, size=3)
    midi = np.where(v & (med > 0), med, np.nan)
    return midi, hop


def _dev(midi_seg: np.ndarray, pitch: int) -> np.ndarray:
    d = midi_seg - pitch
    # fold octave errors back toward the note (RMVPE occasionally jumps 12)
    return d - 12 * np.round(d / 12)


def _runs(mask: np.ndarray):
    """(start, stop) index pairs of True runs."""
    if not mask.any():
        return []
    m = np.concatenate([[False], mask, [False]]).astype(int)
    edges = np.flatnonzero(np.diff(m))
    return list(zip(edges[::2], edges[1::2]))


def _vibrato_spans(x: np.ndarray, hop: float):
    """(i0, i1) frame spans of sustained periodic oscillation in x."""
    hp = x - uniform_filter1d(x, size=max(3, int(0.25 / hop)), mode="nearest")
    dist = max(1, int(1 / VIB_RATE[1] / 2 / hop))
    pk, _ = find_peaks(hp, distance=dist)
    tr, _ = find_peaks(-hp, distance=dist)
    ext = sorted([(i, 1) for i in pk] + [(i, -1) for i in tr])
    # chains of consecutive half-cycles at vibrato rate and depth
    chains, cur = [], []
    for (i0, s0), (i1, s1) in zip(ext, ext[1:]):
        half = (i1 - i0) * hop
        ok = (s0 != s1
              and 1 / (2 * VIB_RATE[1]) <= half <= 1 / (2 * VIB_RATE[0])
              and abs(hp[i1] - hp[i0]) >= VIB_MIN_P2P)
        if ok:
            cur.append((i0, i1, abs(hp[i1] - hp[i0])))
        elif cur:
            chains.append(cur)
            cur = []
    if cur:
        chains.append(cur)
    spans = []
    for c in chains:
        amps = [x[2] for x in c]
        # >= 2 full cycles of similar depth: a single pitch step detrends
        # into one big peak/trough pair and must not count as vibrato.
        if (len(c) >= VIB_MIN_HALVES and max(amps) <= 2.5 * min(amps)
                and (c[-1][1] - c[0][0]) * hop >= VIB_MIN_DUR):
            spans.append((c[0][0], c[-1][1]))
    return spans


def detect(midi_f0, hop, notes):
    """notes: list of dicts with start/end (real seconds) + pitch + idx."""
    ev = []
    n_frames = len(midi_f0)
    starts = np.asarray([n["start"] for n in notes])

    def seg(t0, t1):
        a = max(0, int(round(t0 / hop)))
        b = min(n_frames, int(round(t1 / hop)))
        return a, b

    def note_at(t):
        k = int(np.searchsorted(starts, t, side="right")) - 1
        return max(0, k)

    # ---- vibrato: on continuous voiced runs (a held vowel is often split
    # into several bars), relative to the run's own slow trend.
    vib = np.zeros(n_frames, bool)
    for r0, r1 in _runs(~np.isnan(midi_f0)):
        r0, r1 = r0 + EDGE, r1 - EDGE
        if (r1 - r0) * hop < VIB_MIN_DUR:
            continue
        for a, b in _vibrato_spans(midi_f0[r0:r1], hop):
            t = (r0 + a) * hop
            ev.append({"type": "vibrato", "note": notes[note_at(t)]["idx"],
                       "t": t, "dur": round(float((b - a) * hop), 2)})
            vib[r0 + a:r0 + b + 1] = True

    # ---- fall: where the voice actually stops (union notes are sustained
    # up to the next onset, so bar gaps can't tell phrase ends). The run's
    # last bar must be held on pitch, then glide down before voicing ends,
    # and the melody must not explain the descent (no lower bar starting
    # inside it).
    for r0, r1 in _runs(~np.isnan(midi_f0)):
        r1 -= FALL_EDGE
        if (r1 - r0) * hop < FALL_MIN_HOLD + FALL_MIN_GLIDE:
            continue
        k = note_at((r1 - 1) * hop)
        n = notes[k]
        a = max(r0, int(round(n["start"] / hop)))
        d = _dev(midi_f0[a:r1], n["pitch"])
        on = np.flatnonzero(np.abs(d) <= ON_PITCH)
        if on.size * hop < FALL_MIN_HOLD:
            continue
        after = d[on[-1]:]
        below = np.flatnonzero(after <= -0.5)
        if after.size < 3 or not below.size:
            continue
        tail = float(np.median(after[-3:]))
        t_fall = (a + on[-1]) * hop
        nxt = notes[k + 1] if k + 1 < len(notes) else None
        melodic = (nxt is not None and nxt["start"] < r1 * hop
                   and nxt["pitch"] < n["pitch"])
        if (-7 <= tail <= -FALL_MIN_DROP
                and (after.size - below[0]) * hop >= FALL_MIN_GLIDE
                and not melodic and not vib[a + on[-1]]):
            ev.append({"type": "fall", "note": n["idx"], "t": t_fall,
                       "drop": round(-tail, 2)})

    for k, n in enumerate(notes):
        s, e, p = n["start"], n["end"], n["pitch"]
        prv = notes[k - 1] if k > 0 else None

        # ---- shakuri: a voiced run that *starts* at the bar (scoops often
        # begin a little early, never reaching into the previous bar), whose
        # first reliable frames sit below target and then glide up onto it.
        lead = 0.06 if (prv is None or s - prv["end"] > 0.06) else 0.0
        a, b = seg(s - lead, min(e, s + SHAKURI_WIN))
        d = _dev(midi_f0[a:b], p)
        runs = [r for r in _runs(~np.isnan(d)) if r[0] * hop <= lead + 0.10]
        legato_from_below = (prv is not None and s - prv["end"] < 0.05
                             and prv["pitch"] < p)
        if runs and not legato_from_below:
            r0, r1 = runs[0]
            onset_cut = r0 > 0 or a == 0 or np.isnan(midi_f0[a - 1])
            dv = d[r0 + (EDGE if onset_cut else 0):r1]
            if len(dv) >= 5:
                head = float(np.median(dv[:3]))
                arrive = np.flatnonzero(np.abs(dv) <= ON_PITCH)
                if (-SHAKURI_MAX_DEPTH <= head <= -SHAKURI_MIN_DEPTH
                        and arrive.size and arrive[0] * hop >= SHAKURI_MIN_GLIDE
                        and np.all(np.diff(dv[:arrive[0] + 1]) > -0.3)):
                    ev.append({"type": "shakuri", "note": n["idx"],
                               "t": s, "depth": round(-head, 2)})

        # ---- kobushi: an isolated up-and-back turn in the middle of a held
        # bar, returning to the same level, not part of a vibrato.
        if e - s < KOB_MIN_NOTE:
            continue
        a, b = seg(s + KOB_GUARD, e - KOB_GUARD)
        d = _dev(midi_f0[a:b], p)
        if np.isnan(d).any() or len(d) < 8:
            continue  # voicing break inside the bar -> not a sung turn
        pk, props = find_peaks(
            d, prominence=KOB_MIN_PROM,
            width=(KOB_WIDTH[0] / hop, KOB_WIDTH[1] / hop))
        for i, lb, rb in zip(pk, props["left_bases"], props["right_bases"]):
            if vib[a + i] or abs(d[lb] - d[rb]) > ON_PITCH:
                continue
            ev.append({"type": "kobushi", "note": n["idx"], "t": (a + i) * hop})
    ev.sort(key=lambda x: x["t"])
    return ev


@click.command()
@click.option("--f0", "f0_path", required=True, type=click.Path(exists=True))
@click.option("--midi", "midi_path", required=True, type=click.Path(exists=True),
              help="REAL-TIME melody MIDI (e.g. melody_markers.gamescore.mms.union.mid)")
@click.option("--time-warp", "warp_path", type=click.Path(exists=True), default=None,
              help="warp json of the display MIDI; adds t_display to every event")
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False))
def main(f0_path, midi_path, warp_path, out_path):
    midi_f0, hop = _load_f0(f0_path)
    pmn = pretty_midi.PrettyMIDI(midi_path)
    raw = sorted((n for inst in pmn.instruments for n in inst.notes),
                 key=lambda n: (n.start, n.pitch))
    if warp_path:
        w = json.loads(Path(warp_path).read_text())
        real, disp = np.asarray(w["real"], float), np.asarray(w["display"], float)
        to_disp = lambda t: float(np.interp(t, real, disp))  # noqa: E731
    else:
        to_disp = lambda t: float(t)  # noqa: E731
    notes = [{"idx": i, "pitch": n.pitch, "start": n.start, "end": n.end}
             for i, n in enumerate(raw)]
    ev = detect(midi_f0, hop, notes)
    for x in ev:
        x["t_display"] = round(to_disp(x["t"]), 3)
        x["t"] = round(x["t"], 3)
    counts = {k: sum(1 for x in ev if x["type"] == k)
              for k in ("shakuri", "kobushi", "fall", "vibrato")}
    vib_sec = round(sum(x.get("dur", 0) for x in ev if x["type"] == "vibrato"), 1)
    Path(out_path).write_text(json.dumps(
        {"counts": counts, "vibrato_seconds": vib_sec, "n_notes": len(notes),
         "events": ev}, ensure_ascii=False, indent=1))
    print(f"[techniques] {counts} vibrato {vib_sec}s over {len(notes)} bars")


if __name__ == "__main__":
    main()
