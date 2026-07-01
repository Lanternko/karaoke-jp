#!/usr/bin/env python3
"""Push sung-char onsets from acoustic/CTC onset toward the perceptual *visual*
onset (≈ vowel onset), per the survey's perceptual-onset finding (PAT / p-center
≈ vowel onset; Polfreman 2013, Sundberg 2007, Marcus 1981, Huggins 1972).

The MMS CTC aligner (and F0) place a mora's onset at the *consonant* start. The
ear — and the karaoke wipe people read as "the syllable starts now" — locks to
the *vowel* onset, which trails the consonant by a class-dependent delay
(voiceless stops/fricatives lead the vowel most; voiced/sonorant little; pure
vowels not at all). This is exactly why every "F0 hits dead-on" experiment got
ear-vetoed: the F0/consonant onset is *earlier* than the perceived attack.

This is the離線, zero-model "consonant-class → delta push-back" recipe. It is
NOT the rejected F0 re-entry guard (that snapped TO an F0 voicing onset, which
sits in breath/pre-phonation — the opposite end): here we move AWAY from the
consonant onset toward the vowel core, using only the kana's leading consonant.

Per char: new_start = start + delta(class), capped so it never crosses the
vowel core (<= start + cap_frac * consonant_budget) nor the char's own end.
delta values are literature-magnitude defaults; tune on ear/visual gold.
"""
from __future__ import annotations

import json
from pathlib import Path

import click

# Leading-consonant class per hiragana (the *onset* of the mora). Moraic units
# with no consonant onset (pure vowels, ん, っ, ー, small vowels) get delta 0.
_VOICELESS_STOP = set("かきくけこたてとぱぴぷぺぽ")            # k, t, p  (ち/つ are affricates -> fricative class)
_VOICELESS_FRIC = set("さしすせそはひふへほ") | set("ちつ")    # s/sh, h/f, ch/ts (affricates lead the vowel like fricatives)
_VOICED_OBSTRUENT = set("がぎぐげござじずぜぞだぢづでどばびぶべぼゔ")  # g, z, j, d, b, v
_SONORANT = set("まみむめもなにぬねのらりるれろわやゆよ")       # m, n(な行), r, w, y
_ZERO = set("あいうえおぁぃぅぇぉんっーゝゞ")                  # vowel-initial / moraic / long / small

# Combining small forms (ゃゅょ) inherit the host consonant's class, so we only
# ever look at the first (host) kana of a mora cluster.
_SMALL = set("ゃゅょぁぃぅぇぉ")


def consonant_class(kana: str) -> str:
    ch = kana[0] if kana else ""
    if ch in _VOICELESS_STOP:
        return "stop"
    if ch in _VOICELESS_FRIC:
        return "fric"
    if ch in _VOICED_OBSTRUENT:
        return "voiced"
    if ch in _SONORANT:
        return "sonorant"
    return "zero"


DEFAULT_DELTAS = {  # seconds, literature-magnitude
    "stop": 0.040,
    "fric": 0.035,
    "voiced": 0.010,
    "sonorant": 0.010,
    "zero": 0.0,
}


def push_char(start: float, end: float, kana: str, deltas: dict,
              *, cap_frac: float, min_vowel: float) -> float:
    cls = consonant_class(kana)
    delta = deltas[cls]
    if delta <= 0:
        return start
    dur = end - start
    if dur <= 0:
        return start
    # never push past the vowel core: cap at a fraction of the char's span, and
    # always leave at least min_vowel of vowel after the new onset.
    capped = min(delta, cap_frac * dur, max(0.0, dur - min_vowel))
    return start + capped


def apply_visual_onset(lines: list[dict], deltas: dict, *,
                       cap_frac: float, min_vowel: float) -> int:
    moved = 0
    for line in lines:
        sung_first = None
        sung_last = None
        for tok in line.get("tokens", []):
            if tok.get("is_punct"):
                continue
            for ch in tok.get("chars") or []:
                kana = ch.get("char", "")
                s, e = float(ch["start"]), float(ch["end"])
                ns = push_char(s, e, kana, deltas, cap_frac=cap_frac, min_vowel=min_vowel)
                if ns > s + 1e-6:
                    ch["start"] = ns
                    moved += 1
                if sung_first is None:
                    sung_first = ch
                sung_last = ch
        # keep the line span consistent with its (possibly pushed) first char
        if sung_first is not None:
            line["start"] = float(sung_first["start"])
        if sung_last is not None:
            line["end"] = float(sung_last["end"])
    return moved


@click.command()
@click.argument("aligned_in", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", required=True)
@click.option("--stop-delta", default=DEFAULT_DELTAS["stop"], show_default=True)
@click.option("--fric-delta", default=DEFAULT_DELTAS["fric"], show_default=True)
@click.option("--voiced-delta", default=DEFAULT_DELTAS["voiced"], show_default=True)
@click.option("--sonorant-delta", default=DEFAULT_DELTAS["sonorant"], show_default=True)
@click.option("--cap-frac", default=0.5, show_default=True,
              help="delta is capped at this fraction of the char's duration")
@click.option("--min-vowel", default=0.04, show_default=True,
              help="always leave at least this much vowel after the pushed onset")
def main(aligned_in, out, stop_delta, fric_delta, voiced_delta, sonorant_delta,
         cap_frac, min_vowel):
    deltas = {"stop": stop_delta, "fric": fric_delta, "voiced": voiced_delta,
              "sonorant": sonorant_delta, "zero": 0.0}
    lines = json.loads(Path(aligned_in).read_text(encoding="utf-8"))
    moved = apply_visual_onset(lines, deltas, cap_frac=cap_frac, min_vowel=min_vowel)
    Path(out).write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(f"[visual-onset] pushed {moved} char onsets -> {out}")


if __name__ == "__main__":
    main()
