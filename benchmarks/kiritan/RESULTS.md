# Kiritan Benchmark — GAME -l ja (2026-06-11)

**Setup.** Tohoku Kiritan singing DB, 50 Japanese pop songs, **a cappella**
(96 kHz mono, no separation). GAME-1.0-large `extract -l ja`, **zero-shot**.
GT = the database's `midi_label` (Melodyne-assisted manual transcription).
Official MIR-ST500 `evaluate.py` (mir_eval, onset 50 ms, pitch 50 cents,
offset max(50ms, 0.2×dur)). 10,370 GT notes, ~11,510 predicted.
Internal MIR-style protocol (no public Kiritan AST leaderboard exists).

## Result

| | COn | COnP | COnPOff |
|---|---|---|---|
| GAME -l ja, raw GT | **0.793** | 0.531 | 0.400 |
| GAME -l ja, **transposition-corrected GT** | **0.794** | **0.579** | **0.438** |

MIDI-int vs CSV-float pitch were near-identical (COnP 0.531 vs 0.534) — so the
int rounding was *not* the cause of the low COnP. The cause is below.

## The finding: Kiritan GT has per-song semitone transposition errors

Raw COn→COnP collapses (0.793 → 0.531): ~33% of correctly-onset notes "fail"
pitch, **23% exactly 1 semitone flat**. This is NOT a GAME weakness:

1. **Global tuning is fine** — mean pitch deviation +2.0 cents (in tune to A440).
2. **Not octave errors** — 1 octave miss in 8,746 matched notes.
3. **Concentrated, not uniform** — 13/50 songs have >40% of notes 1 semitone
   flat; 30/50 have <10%. Bimodal ⇒ per-song key offset, not a model property.
4. **Verified against an independent F0 tracker (RMVPE), not against GAME.**
   Per song, median(GT − RMVPE_sung_F0) rounded to integer:
   **34 songs offset 0 · 14 songs +1 · 1 song +3 · 1 song −2** ⇒ 16/50 GT
   transposed relative to the actual recording.
5. **Song 08 spot-check** (worst, 75% "flat"): GAME vs RMVPE median **+0.00**;
   GT vs RMVPE median **+0.88**. GAME transcribes the sung pitch exactly; the
   GT label sits ~1 semitone above the audio.

Correcting the 16 songs to match the audio (shift from RMVPE, independent of
GAME) lifts COnP 0.531 → 0.579, COnPOff 0.400 → 0.438.

## Read

1. **Onset on clean Japanese a cappella is excellent: COn 0.794** — beats
   GAME's own MIR-ST500 (0.732, separated Mandarin) and approaches in-domain
   *supervised* MIR-ST500 baselines (EFN 0.754). Clean audio + correct language
   + no separation = best onset.
2. **Even a curated academic SVS dataset has audio-label mismatches** (16/50
   transposed) — the exact "annotation vs audio" failure our human-ear gold
   methodology catches, found here by cross-checking GT against RMVPE.
3. **Residual COn→COnP gap (0.794→0.579) is the score-vs-F0 distinction.**
   GAME measures actual sung pitch (≈0.0 vs RMVPE); where singers sing flat, a
   Melodyne-snapped "score" GT disagrees — the professor's pitch-vs-note concern,
   quantified on a public dataset.

## Caveats

- N=50, zero-shot, no separation. +3/−2 shifts on 2 songs may be RMVPE
  estimation noise; the 14 songs at +1 are the clean pattern.
- Pred ~11,510 vs GT 10,370 (+11%): GAME over-segments slightly (recall>prec).
- A −20 ms global prediction shift nudges raw COn 0.793→0.803 (minor timing
  convention effect; not applied in the headline numbers).

## Artifacts

- GT: `kiritan_gt.json` (raw), `gt_transcorr.json` (transposition-corrected)
- GAME: `game_raw_ja.json` (MIDI-int), `game_ja_float.json` (CSV-float)
- RMVPE F0 (for the independent transposition check): `f0/*.npz`
- Eval outputs: `eval_game_raw_ja.txt`

## Note-cleanup ablation (幻覺小顆音, 2026-06-11)

Professor's critique quantified: GAME emits **14.7%** notes <150ms vs GT's
6.2% (2.4×); 23% of predicted notes have no GT onset within ±50ms, of which
28% are <150ms (median 210ms) — short hallucinated fragments are real.

Post-processing sweep (zero re-inference, on the prediction JSON):

| variant | COn | COnP | COnPOff |
|---|---|---|---|
| raw | 0.794 | 0.579 | 0.438 |
| **min-dur absorb** (<100ms folds into prev tail) | **0.798** | **0.589** | **0.443** |
| merge same-pitch + min-dur | 0.710 | 0.514 | 0.321 |

- **Same-pitch merging is poison** (−8pp COn): consecutive same-pitch notes on
  different syllables are real in singing — the chidori Bb4×3 / --keep-repeats
  lesson, now benchmark-proven.
- **Min-dur fragment absorption is a uniform small win** (all three metrics up,
  notes −5.7%) — adopted as the recommended cheap cleanup.
- Remaining spurious notes are ≥150ms — not fragments but ornament/bleed-class;
  needs GAME's boundary/presence threshold sweep or learned (CTC/CE) boundaries.
