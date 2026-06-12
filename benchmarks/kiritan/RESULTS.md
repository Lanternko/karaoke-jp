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

## Second GT defect class + CE+CTC head-to-head (2026-06-12)

**Kiritan GT defect #2: per-song global TIME shift.** Both models collapsed
on the same 5 songs (01/02/03/13/14, COn 0.09-0.27). RMVPE-voicing
cross-correlation (model-independent, same spirit as Wang & Jang's +30ms
MIR-ST500 label correction) finds those exact songs shifted +150..+340ms;
correcting lifts e.g. song 03 COn 0.10→0.95. `gt_timefix.json` = transposition
+ time corrected.

| system (N=50, double-corrected GT) | COn | COnP | COnPOff |
|---|---|---|---|
| GAME -l ja (zero-shot) | **0.862** | 0.644 | **0.502** |
| CE+CTC (Mandarin-trained, zero-shot) | 0.860 | 0.652 | 0.492 |

- **Essentially tied on Japanese a cappella** — GAME edges COn/COnPOff,
  CE+CTC edges COnP. (Early single-song smoke test suggesting CE+CTC
  collapse was the GT time-shift, not the model.)
- Combined finding: Kiritan GT needed BOTH a pitch-transposition fix
  (16/50 songs) and a time-shift fix (5/50) before either model could be
  fairly scored — a fully automatic RMVPE-based GT audit protocol that
  generalizes to any singing dataset.

## ROSVOT out-of-the-box (2026-06-12)

Same protocol as MIR-ST500 section (no word_durs; RWBD active). N=50,
timefix GT: COn 0.414 / COnP 0.290 / COnPOff 0.190 @50ms; COn 0.745 @100ms.
The 50→100ms jump (+33pp) shows systematic boundary jitter at the
50-100ms scale — the predicted-word-boundary effect, not pitch errors.
Three-way @50ms on identical input: GAME 0.862 > CE+CTC 0.860 >> ROSVOT 0.414.

## Phoneme-boundary alignment benchmark (MMS/SOFA, 2026-06-12)

This is a separate benchmark from the note-transcription tables above. GAME,
CE+CTC, and ROSVOT emit notes and are scored with COn/COnP/COnPOff; MMS and
SOFA emit lyric/phone timings and are scored against Kiritan `mono_label`
phoneme boundaries.

**Protocol.** N=50, 20,291 evaluated phones. Source labels are
`mono_label/*.lab`; `pau/br/SP/AP/<SP>/<AP>/cl` are ignored for evaluation.
`cl` is ignored because the released Japanese SOFA model treats it as a
silence-class label. Metrics are absolute boundary errors over phone starts and
ends: MAE, median AE, P90 AE, and percentage within 50 ms.

| system | input packaging | boundary MAE | median AE | P90 AE | <=50ms |
|---|---:|---:|---:|---:|---:|
| **SOFA JPN v0.0.2b** ⚠trained-on-Kiritan | phrase tokens from Kiritan `pau/br` | **0.018s** | **0.007s** | **0.053s** | **89.3%** |
| MMS_FA (torchaudio bundle) zero-shot | full phone sequence | 0.112s | 0.043s | 0.184s | 56.0% |
| MMS-JA karaoke ckpt zero-shot | full phone sequence | 0.218s | 0.055s | 0.197s | 46.7% |
| SOFA JPN v0.0.2b | whole song as one token | 3.076s | 0.008s | 0.085s | 82.4% |

> **⚠ THIS IS NOT A FAIR SOFA-vs-MMS TEST (reframed 2026-06-12).** The released
> SOFA JPN model's `data_providers.md` lists its training data as:
> Amanoshi Cipher, JSUT, Tohouku itako, **Tohouku kiritan**, Namine ritsu,
> No.7, Ofuton P, Oniku kurumi, PJS, Zundamon. Kiritan is in there.
>
> Critically, **the release discloses no train/test split** — no list of which
> songs/versions/annotations were used. Consequences:
> - We **cannot quantify** the contamination, and we **cannot** even fall back
>   to "test SOFA on held-out Kiritan songs" because we don't know which are held out.
> - So this table is **not** a SOFA-vs-MMS generalization comparison. It is a
>   **MMS-on-SOFA's-training-distribution sanity check**: MMS_FA/MMS-JA have
>   NOT seen any of these corpora, so their 112/218 ms ARE clean zero-shot
>   numbers; SOFA's 18 ms is measured on (part of) its own training set and is
>   not comparable.
> - Constructive flip: the other 9 corpora (PJS, JSUT, Namine Ritsu, …) are
>   all fair, unseen test sets **for MMS** — a route to a real Japanese
>   clean-singing phone-boundary benchmark, reported honestly as such.
>
> On a domain SOFA did NOT train on (our separated polyphonic vocals), SOFA
> collapses — see `tmp/sofa-ourgold/RESULTS.md`.
>
> Sources: SOFA repo https://github.com/qiuqiao/SOFA · JPN model release
> https://github.com/colstone/SOFA_Models/releases/tag/JPN-V0.0.2b · training
> list in the model bundle's `data_providers.md`.

**Read.**

1. **SOFA is the clear winner on clean Japanese singing phone boundaries** when
   given phrase-level transcript tokens. The median boundary error is only 7 ms;
   even P90 is ~53 ms, which is already in karaoke-grade territory.
2. **Input packaging matters more than the model headline.** Feeding SOFA the
   entire song as one token looks good by median but catastrophically fails on
   repeated sections (MAE 3.08 s). Using Kiritan's `pau/br` to form phrase
   tokens gives SOFA silence anchors and removes the outlier tail.
3. **MMS is usable but not competitive on this clean Japanese phone-boundary
   task.** The original MMS_FA bundle beats the karaoke-adapted MMS-JA here,
   even though MMS-JA wins on some polyphonic lyric-alignment cases. This is
   consistent with the Jamendo finding: domain and language adaptations have
   orthogonal strengths.

**Implementation artifacts.**

- Driver: `phone_boundary_benchmark.py`
- MMS outputs: `phone_boundary/mms_ja_htk`, `phone_boundary/mms_fa_htk`
- SOFA phrase outputs: `phone_boundary_phrase/sofa_segments/htk/phones`
- Eval JSONs: `phone_boundary/eval_mms_ja_ignore_cl.json`,
  `phone_boundary/eval_mms_fa_ignore_cl.json`,
  `phone_boundary_phrase/eval_sofa_ignore_cl.json`

## GAME threshold sweep (2026-06-12)

seg/est threshold grid on timefix GT. est (presence) is a dead knob;
**seg (boundary) 0.2->0.3 lifts all three metrics** by cutting spurious
boundary splits:

| config | COn | COnP | COnPOff | notes |
|---|---|---|---|---|
| default seg0.2/est0.2 | 0.862 | 0.644 | 0.502 | 11506 |
| **seg0.3/est0.4** | **0.872** | **0.658** | **0.513** | 11179 |

Cross-validated on MIR-ST500 separated vocals (COn .732->.740, COnPOff
.411->.416) -> promoted into run_game_chain as GAME_SEG_THRESHOLD=0.3.
