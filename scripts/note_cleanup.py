#!/usr/bin/env python3
"""Mora-aware display-note cleanup (whale survey, 2026-07-02).

GAME segments acoustically, so three families of display bugs survive the
existing drop_fragments/absorb_wiggles pass (all ear-confirmed on whale):

  1. SCOOP mis-splits — a しゃくり approach (sing a step low, slide up into
     the sustain) becomes two notes on ONE mora (の @5.97: 65 then 67).
  2. same-pitch shatter — one held note splits into long + short pieces a
     few tens of ms apart (か @18.84: 67 + 67 with a 20 ms gap).
  3. phantom tails — reverb/echo in the separated stem after the voice
     stops transcribes as short fragments (に @66.92/67.43: the stem is
     -30..-60 dB and neither f0 tracker supports the note's pitch).

All rules are MORA-AWARE (a merge never crosses an MMS char onset, so a
genuine one-mora melisma of distinct held pitches is untouched) and the
destructive rule (3) additionally requires ACOUSTIC evidence to fire:
`f0 > 0` from one tracker alone is NOT trusted (RMVPE hallucinated a 57 in
whale's dead-silent 240 s interlude and octave-jumped +12 on the breathy
outro); a note survives only with stem energy AND a tracker at its pitch.

Display-layer only — never feed cleaned notes to the eval harness.
"""
from __future__ import annotations

import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from karaoke_jp.lrc_export import split_furigana  # noqa: E402
from karaoke_jp.ruby import kata_to_hira  # noqa: E402

Note = tuple[float, float, int]
Span = tuple[float, float]

# a note overlapping no char window even padded this much is an ORPHAN
# (mirrors make_display_grid._char_windows pad)
ORPHAN_PAD = 0.35
# rule 1 (scoop): approach note at most this long ...
SCOOP_MAX_DUR = 0.25
# ... rising 1..3 semitones into a sustain at least this many times longer
# (whale の: 0.19 s -> 0.58 s, +2 st; the real descending melisma at 246.9 s
# is blocked by the ascending-only gate, not by these ratios)
SCOOP_MAX_STEP = 3
SCOOP_MIN_RATIO = 1.5
SCOOP_MAX_GAP = 0.04
# rule 2 (shatter): same pitch, same mora, gap at most this
SAMEPITCH_MAX_GAP = 0.08
# rule 3 (tails): a fragment this short, after a real gap, PAST the mora's
# main note must earn its place acoustically
TAIL_MAX_DUR = 0.35
TAIL_MIN_GAP = 0.10
# acoustic thresholds (dB relative to the stem's voiced-median RMS)
SUPPORT_MIN = 0.30       # fraction of frames with a tracker at the pitch
TAIL_REL_DB = -25.0      # tail may be at most this far below its mora's main note
ORPHAN_ABS_DB = -40.0    # orphan floor (whale 240 s ghost: -89 dB; real soft outro: -26)
PITCH_TOL = 1.0          # semitones for "tracker agrees with note pitch"
# User-facing default: one mora has one displayed pitch.  Variants within
# this distance are treated as intonation/portamento around one score note,
# not as a new note attack.  Wider changes remain eligible melisma.
MORA_PRIMARY_MAX_STEP = 3
MORA_PRIMARY_MAX_GAP = 0.12
MORA_SPILL_MIN_OVERLAP = 0.50


@dataclass(frozen=True)
class MoraSlot:
    """A real reading mora, retaining its parent orthographic-char span.

    MMS writes timings back to surface characters, so a kanji such as
    ``静（しず）`` has one char span but two morae.  The previous cleanup used
    the char span itself as a "mora" and therefore could not distinguish the
    leading し note from a ず scoop.  ``mora_id`` is global song order.
    """

    mora_id: int
    start: float
    end: float
    kana: str
    line_index: int
    char_id: int


def flatten_chars(aligned: list[dict]) -> list[Span]:
    """Sorted (start, end) for every aligned char with real duration."""
    return sorted(
        (float(ch["start"]), float(ch["end"]))
        for line in aligned for tok in (line.get("tokens") or [])
        for ch in tok.get("chars", []) if ch["end"] > ch["start"])


def apply_recut_to_aligned(aligned: list[dict], patches: list[dict]) -> int:
    """Apply lyric_recut char windows to the aligned data itself.

    make_portrait_grid.apply_lyric_recut fixes the LYRIC wipe, but the note
    pipeline (gating windows, char-onset splitting, mora assignment) kept
    reading the raw collapsed alignment — whale's 睡 spanned 234.4..248.1 s,
    so its char window blessed ghost notes across a 10 s instrumental.
    Patching the source once, before anything derives windows from it, fixes
    every consumer. Match/format identical to apply_lyric_recut.
    """
    applied = 0
    for sp in patches:
        if "lyric_recut" not in sp:
            continue
        target = float(sp["lyric_recut"])
        times = sp["chars"]
        for line in aligned:
            chars = [ch for tok in (line.get("tokens") or [])
                     for ch in tok.get("chars", [])]
            if not chars or abs(float(chars[0]["start"]) - target) > 0.5:
                continue
            if len(chars) != len(times):
                raise ValueError(
                    f"lyric_recut for line @{target}: {len(times)} time pairs "
                    f"but line {line.get('text')!r} has {len(chars)} chars")
            for ch, (s, e) in zip(chars, times):
                ch["start"], ch["end"] = float(s), float(e)
            line["start"], line["end"] = float(times[0][0]), float(times[-1][1])
            applied += 1
    return applied


def assign_moras(notes: list[Note], chars: list[Span]) -> list[int]:
    """Index of the max-overlap char per note; -1 = orphan.

    Overlap is measured against the UNPADDED char windows — padding here
    blurred adjacent morae into ties and mis-assigned whale's の scoop to
    the preceding た. The pad only rescues notes that touch no window at
    all (sustain spill / tail fragments), assigning them to the nearest
    char within ORPHAN_PAD; farther than that is an orphan (-1)."""
    out = []
    for s, e, _p in notes:
        best, best_ov = -1, 0.0
        for k, (cs, ce) in enumerate(chars):
            ov = min(e, ce) - max(s, cs)
            if ov > best_ov:
                best, best_ov = k, ov
        if best == -1:
            gap = ORPHAN_PAD
            for k, (cs, ce) in enumerate(chars):
                d = max(cs - e, s - ce, 0.0)
                if d < gap:
                    best, gap = k, d
        out.append(best)
    return out


def _partition_char_notes(notes: list[Note], indices: list[int], n_morae: int) -> list[list[int]]:
    """Partition one multi-mora char's notes into contiguous mora groups.

    Equal time splitting is wrong for sung kanji (Whale's 静: し is short,
    ず is long).  Choose the monotone partition with the lowest duration-
    weighted within-group pitch variance.  Thus 60 | 67,69 becomes
    ``し=60`` and ``ず=67,69`` instead of the geometrically tempting
    ``し=60,67`` / ``ず=69``.
    """
    n = len(indices)
    if n_morae <= 1:
        return [indices]
    if n < n_morae:
        # Not enough acoustic events to represent every mora.  Keep every
        # event owned once; cleanup must never manufacture attacks here.
        return [[idx] for idx in indices]

    def cost(lo: int, hi: int) -> float:
        group = [notes[indices[j]] for j in range(lo, hi)]
        weights = [max(e - s, 0.01) for s, e, _p in group]
        total = sum(weights)
        mean = sum(w * p for w, (_s, _e, p) in zip(weights, group)) / total
        variance = sum(w * (p - mean) ** 2 for w, (_s, _e, p) in zip(weights, group))
        # A silence inside one mora is possible but less likely than a cut.
        gaps = sum(max(0.0, group[j][0] - group[j - 1][1]) for j in range(1, len(group)))
        return variance + 2.0 * gaps

    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(n_morae + 1)]
    back = [[-1] * (n + 1) for _ in range(n_morae + 1)]
    dp[0][0] = 0.0
    for k in range(1, n_morae + 1):
        for end in range(k, n + 1):
            for start in range(k - 1, end):
                candidate = dp[k - 1][start] + cost(start, end)
                if candidate < dp[k][end]:
                    dp[k][end] = candidate
                    back[k][end] = start
    groups: list[list[int]] = []
    k, end = n_morae, n
    while k:
        start = back[k][end]
        if start < 0:
            return [indices]
        groups.append(indices[start:end])
        k, end = k - 1, start
    return list(reversed(groups))


def _is_sung_char(ch: str) -> bool:
    if ch.isspace() or ch == "　":
        return False
    return unicodedata.category(ch)[0] not in {"P", "S"}


def _expand_line_to_morae(line: dict) -> list[dict]:
    """Dependency-light reading expansion used by display cleanup.

    Mirrors ``midi_timing.expand_line_to_morae`` but deliberately does not
    import that CLI module (which imports mido even when no MIDI I/O occurs).
    Keeping cleanup importable in the lightweight test/render environment is
    part of the contract.
    """
    morae: list[dict] = []
    for tok in line.get("tokens", []):
        chars = tok.get("chars") or []
        if not chars:
            continue
        reading = tok.get("reading")
        if not reading or tok.get("kana_only"):
            # Kana/okurigana: the aligned chars are already the mora grid.
            # Do not slice by surface length: synthetic fixtures and a few
            # repaired tokens may carry a normalized surface string.
            for ch in chars:
                if _is_sung_char(ch["char"]):
                    morae.append({"kana": ch["char"], "char": ch})
            continue
        segments = split_furigana(tok["surface"], kata_to_hira(reading))
        for _seg_text, seg_reading, c_start, c_end in segments:
            seg_chars = chars[c_start:c_end]
            sung = [c for c in seg_chars if _is_sung_char(c["char"])]
            if not sung:
                continue
            if seg_reading is None:
                for ch in sung:
                    morae.append({"kana": ch["char"], "char": ch})
                continue
            seq = list(seg_reading)
            base, rem = divmod(len(seq), len(sung))
            pos = 0
            for idx, ch in enumerate(sung):
                count = base + (1 if idx < rem else 0)
                for kana in seq[pos:pos + count]:
                    morae.append({"kana": kana, "char": ch})
                pos += count
    return morae


def assign_true_moras(notes: list[Note], aligned: list[dict]) -> tuple[list[int], list[MoraSlot]]:
    """Assign notes to reading morae, not surface characters.

    Notes first choose their max-overlap parent char.  Notes inside a kanji
    carrying multiple reading morae are then partitioned monotonically by
    pitch coherence.  This keeps neighbouring chars from stealing tails and
    fixes the old ``静（しず）`` char/mora category error.
    """
    slots: list[MoraSlot] = []
    char_slots: dict[int, list[int]] = {}
    chars: list[tuple[int, float, float]] = []
    seen_chars: set[int] = set()
    for line_index, line in enumerate(aligned):
        for mora in _expand_line_to_morae(line):
            ch = mora["char"]
            cid = id(ch)
            if cid not in seen_chars:
                chars.append((cid, float(ch["start"]), float(ch["end"])))
                seen_chars.add(cid)
            mid = len(slots)
            slots.append(MoraSlot(mid, float(ch["start"]), float(ch["end"]),
                                  str(mora.get("kana", "")), line_index, cid))
            char_slots.setdefault(cid, []).append(mid)

    assignments = [-1] * len(notes)
    by_char: dict[int, list[int]] = {}
    for note_idx, (s, e, _p) in enumerate(notes):
        best_cid, best_ov = -1, 0.0
        for cid, cs, ce in chars:
            ov = min(e, ce) - max(s, cs)
            if ov > best_ov:
                best_cid, best_ov = cid, ov
        if best_cid == -1:
            best_gap = ORPHAN_PAD
            for cid, cs, ce in chars:
                gap = max(cs - e, s - ce, 0.0)
                if gap < best_gap:
                    best_cid, best_gap = cid, gap
        if best_cid != -1:
            by_char.setdefault(best_cid, []).append(note_idx)

    for cid, indices in by_char.items():
        mids = char_slots[cid]
        groups = _partition_char_notes(notes, indices, len(mids))
        # When notes < morae, map the available events in temporal order to
        # the closest evenly-spaced mora indices; no synthetic note is made.
        if len(groups) < len(mids):
            cs, ce = slots[mids[0]].start, slots[mids[0]].end
            for group in groups:
                centre = sum((notes[i][0] + notes[i][1]) * 0.5 for i in group) / len(group)
                rel = (centre - cs) / max(ce - cs, 1e-6)
                target = min(len(mids) - 1, max(0, int(rel * len(mids))))
                for i in group:
                    assignments[i] = mids[target]
        else:
            for mid, group in zip(mids, groups, strict=True):
                for i in group:
                    assignments[i] = mid
    return assignments, slots


class AcousticEvidence:
    """Frame-level stem evidence: RMS energy + RMVPE + pYIN, cross-checked.

    Whale ground truth for why one tracker is never enough: RMVPE reports
    f0>0 (MIDI 57) inside a -89 dB dead interlude, and reads the breathy
    outro an octave up (80.5 where pYIN holds 68.6, pYIN's range topping out
    at MIDI 95 so it COULD have followed). Energy alone also fails: reverb
    tails sit at -30 dB with no stable pitch.
    """

    def __init__(self, rms_t, rms_db, rmvpe_midi, rmvpe_hop, pyin_t, pyin_midi):
        self.rms_t = rms_t
        self.rms_db = rms_db
        self.rmvpe_midi = rmvpe_midi  # NaN where unvoiced
        self.rmvpe_hop = rmvpe_hop
        self.pyin_t = pyin_t
        self.pyin_midi = pyin_midi    # NaN where unvoiced

    @classmethod
    def load(cls, vocals_path, rmvpe_path, pyin_path) -> "AcousticEvidence | None":
        if not (vocals_path and rmvpe_path and pyin_path):
            return None
        import numpy as np
        import soundfile as sf

        y, sr = sf.read(str(vocals_path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        hop, win = int(0.010 * sr), int(0.025 * sr)
        n = max((len(y) - win) // hop, 1)
        idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
        rms = np.sqrt(np.mean(y[idx] ** 2, axis=1))
        rms_t = np.arange(n) * hop / sr + win / (2 * sr)

        z = np.load(str(rmvpe_path))
        rf0 = np.asarray(z["f0"], dtype=float)
        rhop = float(np.atleast_1d(z["hop_seconds"])[0])
        rmvpe = np.where(rf0 > 0, 69 + 12 * np.log2(np.maximum(rf0, 1e-6) / 440.0),
                         np.nan)

        zp = np.load(str(pyin_path))
        pf0 = np.asarray(zp["f0"], dtype=float)
        pt = np.asarray(zp["times"], dtype=float)
        pyin = np.where(pf0 > 0, 69 + 12 * np.log2(np.maximum(pf0, 1e-6) / 440.0),
                        np.nan)

        # dB relative to the voiced-median stem level (voiced = RMVPE f0>0)
        rt = np.arange(len(rf0)) * rhop
        voiced_on_rms = np.interp(rms_t, rt, (rf0 > 0).astype(float)) > 0.5
        ref = np.median(rms[voiced_on_rms & (rms > 0)]) or 1e-9
        rms_db = 20 * np.log10(np.maximum(rms, 1e-9) / ref)
        return cls(rms_t, rms_db, rmvpe, rhop, pt, pyin)

    def _frames(self, s: float, e: float):
        import numpy as np
        n = max(int(round((e - s) / 0.01)), 3)
        return np.linspace(s, e, n)

    def note_stats(self, s: float, e: float, pitch: int) -> dict:
        import numpy as np
        t = self._frames(s, e)
        med_db = float(np.median(np.interp(t, self.rms_t, self.rms_db)))
        ri = np.clip(np.round(t / self.rmvpe_hop).astype(int), 0,
                     len(self.rmvpe_midi) - 1)
        rm = self.rmvpe_midi[ri]
        pi = np.clip(np.searchsorted(self.pyin_t, t), 0, len(self.pyin_midi) - 1)
        pm = self.pyin_midi[pi]

        def sup(track, p):
            with np.errstate(invalid="ignore"):
                return float(np.mean(np.abs(track - p) <= PITCH_TOL))

        return {
            "med_db": med_db,
            "sup_rmvpe": sup(rm, pitch),
            "sup_pyin": sup(pm, pitch),
            "sup_pyin_oct_down": sup(pm, pitch - 12),
            "support": float(np.mean((np.abs(rm - pitch) <= PITCH_TOL)
                                     | (np.abs(pm - pitch) <= PITCH_TOL))),
        }


def fix_octave_errors(
    notes: list[Note], ev: AcousticEvidence | None,
) -> tuple[list[Note], int, list[str]]:
    """RMVPE octave-up artifacts: RMVPE tracks the note's (high) pitch but
    pYIN — whose range comfortably covers it — holds pitch-12 throughout.
    Whale outro: 81 @250.93 and 79 @251.38 are really 69 / 67.

    Scoped to notes IMPLAUSIBLY HIGH for the song's own register (above the
    95th-percentile pitch + 3 st): pYIN also reads whole breathy mid-range
    sustains a subharmonic octave DOWN (whale 65.1-66.8 s, pyin@53 vs
    GAME+RMVPE@65), and the trust hierarchy is transcription >> any single
    f0 tracker — a tracker disagreement alone must not repitch the melody.
    """
    if ev is None or not notes:
        return notes, 0, []
    ceiling = sorted(p for _s, _e, p in notes)[int(0.95 * (len(notes) - 1))] + 3
    out, fixed, log = [], 0, []
    for idx, (s, e, p) in enumerate(notes):
        local_island = False
        if 0 < idx < len(notes) - 1:
            ps, pe, pp = notes[idx - 1]
            ns, ne, npitch = notes[idx + 1]
            shifted = p - 12
            # An octave-up island can sit inside the song's normal global
            # range (Whale 129.36 p70).  Repair it when shifting -12 makes it
            # locally smooth on BOTH sides and the unshifted pitch is a large
            # jump.  Time guards keep this within one continuous phrase.
            local_island = (
                s - pe <= 0.30 and ns - e <= 0.30
                and max(abs(pp - shifted), abs(npitch - shifted)) <= 3
                and min(abs(pp - p), abs(npitch - p)) >= 8
            )
        if p > ceiling or local_island:
            st = ev.note_stats(s, e, p)
            if (st["sup_pyin"] < 0.2 and st["sup_pyin_oct_down"] >= 0.5
                    and st["sup_rmvpe"] >= 0.3):
                log.append(f"octave-fix {s:.2f}-{e:.2f}s {p}->{p - 12} "
                           f"(pyin@{p - 12} {st['sup_pyin_oct_down']:.2f}"
                           f"{' local-island' if local_island else ''})")
                p -= 12
                fixed += 1
        out.append((s, e, p))
    return out, fixed, log


def merge_same_pitch(
    notes: list[Note], moras: list[int], *, max_gap: float = SAMEPITCH_MAX_GAP,
) -> tuple[list[Note], int]:
    """One held note shattered into pieces: same pitch, same mora, tiny gap."""
    out: list[list] = []
    out_mora: list[int] = []
    merged = 0
    for (s, e, p), m in zip(notes, moras):
        if (out and p == out[-1][2] and m == out_mora[-1] and m != -1
                and s - out[-1][1] <= max_gap):
            out[-1][1] = max(out[-1][1], e)
            merged += 1
            continue
        out.append([s, e, p])
        out_mora.append(m)
    return [tuple(n) for n in out], merged


def merge_scoops(
    notes: list[Note], moras: list[int],
) -> tuple[list[Note], int, list[str]]:
    """Fold a しゃくり approach note into the sustain it slides into.

    Fires only when ALL hold: same mora, the short note is the mora's FIRST
    note, it rises 1..SCOOP_MAX_STEP semitones into a note >= SCOOP_MIN_RATIO
    times longer, with essentially no gap. Descending steps never merge —
    that is what keeps whale's real 69-67-67-69 outro melisma intact.
    """
    out = [list(n) for n in notes]
    out_mora = list(moras)
    merged, log = 0, []
    i = 0
    while i + 1 < len(out):
        (s1, e1, p1), (s2, e2, p2) = out[i], out[i + 1]
        m = out_mora[i]
        first_of_mora = m != -1 and not any(
            out_mora[j] == m for j in range(i))
        d1, d2 = e1 - s1, e2 - s2
        if (m == out_mora[i + 1] and first_of_mora
                and d1 <= SCOOP_MAX_DUR
                and d2 >= SCOOP_MIN_RATIO * d1
                and 1 <= p2 - p1 <= SCOOP_MAX_STEP
                and s2 - e1 <= SCOOP_MAX_GAP):
            log.append(f"scoop-merge {s1:.2f}s {p1}({d1:.2f}s)->{p2}")
            out[i + 1] = [s1, e2, p2]
            del out[i], out_mora[i]
            merged += 1
            continue
        i += 1
    return [tuple(n) for n in out], merged, log


def consolidate_mora_primary(
    notes: list[Note], moras: list[int], slots: list[MoraSlot],
    ev: AcousticEvidence | None = None,
) -> tuple[list[Note], int, int, list[str]]:
    """Default one displayed pitch per mora for nearby pitch variants.

    The dominant in-mora plateau is the score pitch.  Adjacent variants within
    three semitones are intonation/portamento and collapse into that pitch.
    A different-pitch tail that lives mostly beyond the mora span is discarded
    instead of extending the bar (Whale 波に横たえながら @168.56).  Changes wider
    than three semitones are left alone as explicit melisma candidates.

    Dominance is TRACKER-SUPPORT-WEIGHTED duration when acoustics are armed:
    raw duration alone picked whale's く drift (69, 0.42 s of slide) over its
    sung attack plateau (70, 0.39 s, both trackers pinned) by a 0.03 s margin.
    The plateau the trackers actually hold is the score note; a same-length
    portamento away from it is not.  Geometry (overlap, duration) remains the
    fallback without sidecars.
    """
    by_mora: dict[int, list[int]] = {}
    for i, m in enumerate(moras):
        if m != -1:
            by_mora.setdefault(m, []).append(i)

    consumed: set[int] = set()
    replacements: dict[int, Note] = {}
    merged = spills = 0
    log: list[str] = []
    for m, indices in by_mora.items():
        if len(indices) < 2:
            continue
        slot = slots[m]

        def overlap(i: int) -> float:
            s, e, _p = notes[i]
            return max(0.0, min(e, slot.end) - max(s, slot.start))

        if ev is not None:
            def dominance(i: int) -> tuple[float, float]:
                s, e, p = notes[i]
                sup = ev.note_stats(s, e, p)["support"]
                return ((e - s) * max(sup, 0.05), overlap(i))
            main_i = max(indices, key=dominance)
        else:
            main_i = max(indices,
                         key=lambda i: (overlap(i), notes[i][1] - notes[i][0]))
        ms, me, mp = notes[main_i]
        new_s, new_e = ms, me
        changed: list[int] = []
        for i in indices:
            if i == main_i:
                continue
            s, e, p = notes[i]
            if abs(p - mp) > MORA_PRIMARY_MAX_STEP:
                continue
            dur = max(e - s, 1e-6)
            in_ratio = overlap(i) / dur
            gap = max(s - new_e, new_s - e, 0.0)
            if p != mp and i > main_i and in_ratio < MORA_SPILL_MIN_OVERLAP:
                consumed.add(i)
                spills += 1
                log.append(f"mora-spill-drop {s:.2f}-{e:.2f}s p{p} "
                           f"after p{mp} ({slot.kana}, overlap={in_ratio:.2f})")
                continue
            if gap <= MORA_PRIMARY_MAX_GAP:
                new_s, new_e = min(new_s, s), max(new_e, e)
                consumed.add(i)
                changed.append(i)
                merged += 1
        if changed:
            replacements[main_i] = (new_s, new_e, mp)
            pitches = ",".join(str(notes[i][2]) for i in sorted([main_i] + changed))
            log.append(f"mora-primary {new_s:.2f}-{new_e:.2f}s "
                       f"[{pitches}]->{mp} ({slot.kana})")

    out = []
    for i, note in enumerate(notes):
        if i in consumed:
            continue
        out.append(replacements.get(i, note))
    return sorted(out), merged, spills, log


def validate_wide_melismas(
    notes: list[Note], moras: list[int], ev: AcousticEvidence | None,
) -> tuple[list[Note], int, list[str]]:
    """Keep the >3-semitone exception only when every plateau is real.

    Wide motion is the explicit exception to one-pitch-per-mora, but GAME can
    still append an octave blip or bleed fragment.  With acoustic sidecars,
    a secondary plateau must have tracker support and reasonable stem energy;
    otherwise it is not a defensible melisma note.
    """
    if ev is None:
        return notes, 0, []
    by_mora: dict[int, list[int]] = {}
    for i, m in enumerate(moras):
        if m != -1:
            by_mora.setdefault(m, []).append(i)
    drop: set[int] = set()
    log: list[str] = []
    for _m, indices in by_mora.items():
        if len(indices) < 2:
            continue
        pitches = [notes[i][2] for i in indices]
        if max(pitches) - min(pitches) <= MORA_PRIMARY_MAX_STEP:
            continue
        stats = {i: ev.note_stats(*notes[i]) for i in indices}
        main_i = max(indices, key=lambda i: (
            stats[i]["support"], notes[i][1] - notes[i][0]))
        main_db = stats[main_i]["med_db"]
        for i in indices:
            if i == main_i:
                continue
            st = stats[i]
            if st["support"] < SUPPORT_MIN or st["med_db"] < main_db + TAIL_REL_DB:
                s, e, p = notes[i]
                drop.add(i)
                log.append(f"melisma-weak-drop {s:.2f}-{e:.2f}s p{p} "
                           f"({st['med_db']:.0f}dB sup{st['support']:.2f})")
    return [n for i, n in enumerate(notes) if i not in drop], len(drop), log


def drop_unsupported_tails(
    notes: list[Note], moras: list[int], ev: AcousticEvidence | None,
) -> tuple[list[Note], list[int], int, int, list[str]]:
    """Kill phantom fragments, acoustically.

    ORPHANS (no mora/char window even padded) always drop once acoustic
    cleanup is armed.  Lyrics are the product truth: a well-pitched bleed or
    uncharted ad-lib still must not paint a karaoke bar.  This matters at
    Whale's outro, where accompaniment is strong enough to fool BOTH trackers
    after the final lyric has ended.

    TAIL fragments (short, after a real gap, past their mora's main note)
    must have tracker support and sit within TAIL_REL_DB of the mora's main
    note — reverb tails (に: -23/-43 dB below, tracker elsewhere) die, the
    soft-but-pitched outro echoes of ら (both trackers in agreement) stay.
    Without acoustic sidecars this rule does not fire at all.
    """
    if ev is None:
        return notes, moras, 0, 0, []
    # main note per mora = its longest note
    main: dict[int, int] = {}
    for i, (s, e, _p) in enumerate(notes):
        m = moras[i]
        if m != -1 and (m not in main
                        or e - s > notes[main[m]][1] - notes[main[m]][0]):
            main[m] = i
    out, out_mora = [], []
    tails = orphans = 0
    log: list[str] = []
    for i, (s, e, p) in enumerate(notes):
        m = moras[i]
        st = ev.note_stats(s, e, p)
        if m == -1:
            orphans += 1
            log.append(f"orphan-drop {s:.2f}-{e:.2f}s p{p} "
                       f"({st['med_db']:.0f}dB sup{st['support']:.2f})")
            continue
        j = main[m]
        is_tail = (i != j and e - s <= TAIL_MAX_DUR
                   and s >= notes[j][1]
                   and i > 0 and s - notes[i - 1][1] > TAIL_MIN_GAP)
        if is_tail:
            ref_db = ev.note_stats(*notes[j])["med_db"]
            if (st["support"] < SUPPORT_MIN
                    or st["med_db"] < ref_db + TAIL_REL_DB):
                tails += 1
                log.append(f"tail-drop {s:.2f}-{e:.2f}s p{p} "
                           f"({st['med_db']:.0f}dB vs main {ref_db:.0f}dB "
                           f"sup{st['support']:.2f})")
                continue
        out.append((s, e, p)); out_mora.append(m)
    return out, out_mora, tails, orphans, log


def cleanup(
    notes: list[Note], aligned: list[dict], ev: AcousticEvidence | None,
) -> tuple[list[Note], dict, list[str]]:
    """Full mora-aware pass. Order matters: octave fixes first (so the
    same-pitch merge sees corrected pitches), merges next, and the
    acoustically-gated drops last (merged notes are no longer 'fragments')."""
    notes = sorted(notes)
    notes, oct_fixed, log = fix_octave_errors(notes, ev)
    moras, slots = assign_true_moras(notes, aligned)
    notes, sp_merged = merge_same_pitch(notes, moras)
    moras, slots = assign_true_moras(notes, aligned)
    notes, scoops, slog = merge_scoops(notes, moras)
    moras, slots = assign_true_moras(notes, aligned)
    notes, primary, spills, plog = consolidate_mora_primary(notes, moras, slots, ev)
    moras, slots = assign_true_moras(notes, aligned)
    notes, melisma_weak, mlog = validate_wide_melismas(notes, moras, ev)
    moras, slots = assign_true_moras(notes, aligned)
    notes, moras, tails, orphans, dlog = drop_unsupported_tails(notes, moras, ev)
    stats = {"octave_fixed": oct_fixed, "same_pitch_merged": sp_merged,
             "scoops_merged": scoops, "tails_dropped": tails,
             "orphans_dropped": orphans, "mora_primary_merged": primary,
             "mora_spills_dropped": spills,
             "melisma_weak_dropped": melisma_weak}
    return notes, stats, log + slog + plog + mlog + dlog


def load_patches(path) -> list[dict]:
    if not path:
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))
