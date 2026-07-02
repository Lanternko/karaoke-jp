#!/usr/bin/env python3
"""COnPOff+L — lyric-conditioned note transcription metric (Itako adaptation).

Second-dataset (N=2) validation of the metric from ../kiritan/conpoff_l.py. Same
four-condition matching (onset/pitch/offset/lyric), same mora-attribution ("loose"
karaoke reading), same discrimination bootstrap. Only the DATA plumbing changes:

  - Itako mono_label is HTK 100-ns units (÷1e7 to seconds); Kiritan's was seconds.
  - Two GT variants are looped: raw (itako_gt.json) and defectfix (gt_defectfix.json).
    defectfix = itako50 octave (−12) + itako01 onset lag (+50 ms). The +50 ms itako01
    note shift is NOT mirrored in mono_label, so for the defectfix variant we shift
    itako01's GT morae onsets +0.05 s to keep mora ownership intervals aligned with
    the shifted notes. The aligner labs are NOT shifted (they align to the audio,
    which never changed). itako50's octave fix is a pitch-only edit → no mora change.

  primary = defectfix (role-analogue of Kiritan's gt_timefix)
  secondary robustness = raw (itako_gt.json)

    ~/venvs/karaoke-jp/bin/python benchmarks/itako/conpoff_l.py
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from mir_eval.util import _bipartite_match  # noqa: E402

ITAKO = Path.home() / "side_projects/itako/itako_singing"
MONO_UNIT = 1e7             # HTK 100-ns -> seconds
VOWELS = set("aiueo")
STANDALONE = {"N"}          # singable moraic nasal
MERGE_FORWARD = {"cl"}      # geminate closure folds into the next mora
SILENCE = {"pau", "sil", "br", "SP", "AP"}

ONSET_TOL = 0.05
PITCH_TOL = 50.0
OFFSET_RATIO = 0.2
OFFSET_MIN = 0.05

# defectfix shifts itako01 notes +50 ms but leaves mono_label untouched; mirror it
# on the GT morae so ownership intervals stay aligned with the shifted notes.
DEFECTFIX_MORA_SHIFT = {"itako01": 0.05}


# ---------------------------------------------------------------- data loading

def load_notes(path: Path) -> dict[str, list[tuple[float, float, float]]]:
    raw = json.loads(path.read_text())
    return {k: [(float(s), float(e), float(p)) for s, e, p in v] for k, v in raw.items()}


def read_lab(path: Path, unit: float = 1.0) -> list[tuple[float, float, str]]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        s, e, ph = parts
        rows.append((float(s) / unit, float(e) / unit, ph))
    return rows


def group_morae(phones: list[tuple[float, float, str]]) -> list[tuple[float, str]]:
    """(onset, label) per mora. Consonants accumulate until a vowel closes the
    mora; N is standalone; cl merges into the following mora (span from cl
    onset, label without cl)."""
    morae: list[tuple[float, str]] = []
    cur_onset: float | None = None
    cur_phones: list[str] = []
    for s, _e, ph in phones:
        if ph in SILENCE:
            continue
        if ph in STANDALONE:
            if cur_phones:  # dangling consonants (shouldn't happen) — flush
                morae.append((cur_onset, "".join(cur_phones)))
                cur_onset, cur_phones = None, []
            morae.append((s, ph))
            continue
        if cur_onset is None:
            cur_onset = s
        if ph in MERGE_FORWARD:
            continue  # keep onset, label starts with next consonant
        cur_phones.append(ph)
        if ph in VOWELS:
            morae.append((cur_onset, "".join(cur_phones)))
            cur_onset, cur_phones = None, []
    if cur_phones:
        morae.append((cur_onset, "".join(cur_phones)))
    return morae


def attribute(onset: float, morae: list[tuple[float, str]]) -> str:
    """Ownership interval [mora_onset_i, mora_onset_{i+1}); pre-first -> first."""
    if not morae:
        return "?"
    lo, hi = 0, len(morae) - 1
    if onset < morae[0][0]:
        return morae[0][1]
    while lo < hi:  # last mora with onset <= note onset
        mid = (lo + hi + 1) // 2
        if morae[mid][0] <= onset:
            lo = mid
        else:
            hi = mid - 1
    return morae[lo][1]


# ------------------------------------------------------------------- matching

def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def match_notes(ref, est, *, pitch=True, offset=True,
                ref_lab=None, est_lab=None) -> list[tuple[int, int]]:
    """Candidate pairs under the active conditions -> maximum bipartite matching.
    ref/est: [(on, off, midi)]. Returns matched (ref_i, est_j) pairs."""
    graph: dict[int, list[int]] = defaultdict(list)
    for i, (rs, re, rp) in enumerate(ref):
        for j, (es, ee, ep) in enumerate(est):
            if abs(rs - es) > ONSET_TOL:
                continue
            if pitch and abs(1200 * math.log2(hz(ep) / hz(rp))) > PITCH_TOL:
                continue
            if offset and abs(re - ee) > max(OFFSET_MIN, OFFSET_RATIO * (re - rs)):
                continue
            if ref_lab is not None and ref_lab[i] != est_lab[j]:
                continue
            graph[i].append(j)
    matching = _bipartite_match(graph)  # right(est_j) -> left(ref_i)
    return [(ri, ej) for ej, ri in matching.items()]


def f1(n_match: int, n_ref: int, n_est: int) -> float:
    if not n_ref or not n_est or not n_match:
        return 0.0
    p, r = n_match / n_est, n_match / n_ref
    return 2 * p * r / (p + r)


# ----------------------------------------------------------------------- eval

def eval_song(ref, est, ref_morae, est_morae):
    ref_lab = [attribute(s, ref_morae) for s, _e, _p in ref]
    est_lab = [attribute(s, est_morae) for s, _e, _p in est]
    m_on = match_notes(ref, est, pitch=False, offset=False)
    m_onp = match_notes(ref, est, offset=False)
    m_onpoff = match_notes(ref, est)
    m_l = match_notes(ref, est, ref_lab=ref_lab, est_lab=est_lab)
    lyric_ok = sum(ref_lab[i] == est_lab[j] for i, j in m_onpoff)
    return {
        "COn": f1(len(m_on), len(ref), len(est)),
        "COnP": f1(len(m_onp), len(ref), len(est)),
        "COnPOff": f1(len(m_onpoff), len(ref), len(est)),
        "COnPOff+L": f1(len(m_l), len(ref), len(est)),
        "n_matched": len(m_onpoff),
        "n_lyric_ok": lyric_ok,
    }


def macro(per_song: list[dict], key: str) -> float:
    return statistics.mean(s[key] for s in per_song)


def bootstrap_diff(a: list[dict], b: list[dict], key: str, iters: int = 2000,
                   seed: int = 7) -> tuple[float, float, float]:
    """Paired bootstrap over songs for macro(a[key]) - macro(b[key])."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(statistics.mean(a[i][key] for i in idx)
                     - statistics.mean(b[i][key] for i in idx))
    diffs.sort()
    return (statistics.mean(diffs), diffs[int(0.025 * iters)], diffs[int(0.975 * iters)])


def shift_morae(morae: list[tuple[float, str]], dt: float) -> list[tuple[float, str]]:
    return [(o + dt, lab) for o, lab in morae]


def run_variant(variant: str, gt_path: Path) -> dict[tuple[str, str], list[dict]]:
    gt = load_notes(gt_path)
    note_models = {
        "GAME": load_notes(HERE / "game_raw_ja.json"),
        "CE+CTC": load_notes(HERE / "ctcce_pred.json"),
    }
    aligners = {
        "oracle": None,  # GT mono_label morae for BOTH sides = perfect aligner
        "MMS_FA": HERE / "phone_boundary/mms_fa_htk",
        "MMS-JA": HERE / "phone_boundary/mms_ja_htk",
    }

    songs = sorted(gt)
    # GT morae from mono_label (÷1e7); for defectfix, shift itako01 morae +50 ms
    gt_morae: dict[str, list[tuple[float, str]]] = {}
    for s in songs:
        morae = group_morae(read_lab(ITAKO / f"mono_label/{s}.lab", unit=MONO_UNIT))
        if variant == "defectfix" and s in DEFECTFIX_MORA_SHIFT:
            morae = shift_morae(morae, DEFECTFIX_MORA_SHIFT[s])
        gt_morae[s] = morae

    mismatch: dict[str, list[tuple[str, int, int]]] = {"MMS_FA": [], "MMS-JA": []}
    results: dict[tuple[str, str], list[dict]] = {}
    for nm_name, nm in note_models.items():
        for al_name, al_dir in aligners.items():
            per_song = []
            for s in songs:
                if al_dir is None:
                    est_morae = gt_morae[s]
                else:
                    est_morae = group_morae(read_lab(al_dir / f"{s}.lab"))  # sec already
                    g_seq = [m[1] for m in gt_morae[s]]
                    e_seq = [m[1] for m in est_morae]
                    if g_seq != e_seq and nm_name == "GAME":  # count once per aligner
                        mismatch[al_name].append((s, len(g_seq), len(e_seq)))
                per_song.append(eval_song(gt[s], nm[s], gt_morae[s], est_morae))
            results[(nm_name, al_name)] = per_song
    return results, mismatch


def main() -> None:
    variants = {
        "defectfix": HERE / "gt_defectfix.json",   # primary
        "raw": HERE / "itako_gt.json",             # secondary robustness
    }

    all_out = {}
    for variant, gt_path in variants.items():
        results, mismatch = run_variant(variant, gt_path)

        print(f"\n{'='*72}")
        print(f"=== COnPOff+L ladder (Itako N=50, GT={variant}, macro per-song F1) ===")
        print(f"{'note model':10s} {'aligner':8s}  {'COn':>6s} {'COnP':>6s} {'COnPOff':>8s} "
              f"{'COnPOff+L':>10s}  {'P(L|match)':>10s}")
        for (nm_name, al_name), per_song in results.items():
            matched = sum(s["n_matched"] for s in per_song)
            lok = sum(s["n_lyric_ok"] for s in per_song)
            print(f"{nm_name:10s} {al_name:8s}  {macro(per_song, 'COn'):>6.3f} "
                  f"{macro(per_song, 'COnP'):>6.3f} {macro(per_song, 'COnPOff'):>8.3f} "
                  f"{macro(per_song, 'COnPOff+L'):>10.3f}  {lok/matched if matched else 0:>10.1%}")

        print(f"\n--- Discrimination (paired bootstrap over 50 songs, 95% CI, GT={variant}) ---")
        pairs = [
            (("GAME", "MMS_FA"), ("GAME", "MMS-JA"), "same notes, aligner A/B"),
            (("GAME", "oracle"), ("GAME", "MMS_FA"), "aligner tax (oracle - MMS_FA)"),
            (("GAME", "MMS_FA"), ("CE+CTC", "MMS_FA"), "same aligner, note model A/B"),
        ]
        for a_key, b_key, desc in pairs:
            m, lo, hi = bootstrap_diff(results[a_key], results[b_key], "COnPOff+L")
            sig = "SIGNIFICANT" if lo > 0 or hi < 0 else "n.s."
            print(f"{a_key[0]}x{a_key[1]} - {b_key[0]}x{b_key[1]}: "
                  f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]  {sig}   ({desc})")

        print(f"\n--- Tax uniformity (COnPOff -> COnPOff+L drop per system, GT={variant}) ---")
        for (nm_name, al_name), per_song in results.items():
            drop = macro(per_song, "COnPOff") - macro(per_song, "COnPOff+L")
            print(f"{nm_name:10s} x {al_name:8s}: -{drop:.4f}")

        print(f"\n--- Mora sequence mismatch (GT vs aligner, GT={variant}) ---")
        for al_name, rows in mismatch.items():
            print(f"{al_name}: {len(rows)} songs mismatch" +
                  (": " + ", ".join(f"{s}({g}vs{e})" for s, g, e in rows) if rows else ""))

        all_out[variant] = {
            "ladder": {f"{nm}x{al}": {
                "COn": round(macro(ps, "COn"), 4), "COnP": round(macro(ps, "COnP"), 4),
                "COnPOff": round(macro(ps, "COnPOff"), 4),
                "COnPOff+L": round(macro(ps, "COnPOff+L"), 4),
                "P_L_given_match": round(sum(s["n_lyric_ok"] for s in ps) /
                                         max(1, sum(s["n_matched"] for s in ps)), 4),
            } for (nm, al), ps in results.items()},
            "discrimination": {
                f"{a[0]}x{a[1]} - {b[0]}x{b[1]}": dict(zip(
                    ("mean", "lo", "hi"),
                    [round(x, 4) for x in bootstrap_diff(results[a], results[b], "COnPOff+L")]))
                for a, b, _ in [
                    (("GAME", "MMS_FA"), ("GAME", "MMS-JA"), ""),
                    (("GAME", "oracle"), ("GAME", "MMS_FA"), ""),
                    (("GAME", "MMS_FA"), ("CE+CTC", "MMS_FA"), "")]
            },
            "mismatch": {al: [{"song": s, "gt": g, "est": e} for s, g, e in rows]
                         for al, rows in mismatch.items()},
        }

    (HERE / "conpoff_l_results.json").write_text(json.dumps(all_out, indent=2))
    print("\nwritten: conpoff_l_results.json")


if __name__ == "__main__":
    main()
