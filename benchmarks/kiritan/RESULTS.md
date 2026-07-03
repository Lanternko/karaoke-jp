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

## COnPOff+L — lyric-conditioned joint metric, first numbers (2026-07-02)

`conpoff_l.py`. A note is correct only if onset(50ms) + pitch(50c) + offset +
**romaji mora label** all pass. Literature-verified novel (no published
lyric-conditioned COnPOff exists; template = mir_eval.transcription_velocity,
4th-condition precedent = T3MS note-value, arXiv:2502.12438). Mora attribution:
ownership interval [mora_onset, next_mora_onset); `cl` merges into the next
mora; MMS labs are forced alignment of the GT phone sequence, so L isolates
TIMING attribution (lyrics known — the karaoke setting). Sanity: CE+CTC ladder
reproduces official evaluate.py exactly (.860/.652/.492); GAME within 0.003.

| note model | aligner | COn | COnP | COnPOff | **COnPOff+L** | P(L\|match) |
|---|---|---|---|---|---|---|
| GAME | oracle (GT morae) | .860 | .643 | .499 | .442 | 88.6% |
| GAME | MMS_FA | .860 | .643 | .499 | **.408** | 81.6% |
| GAME | MMS-JA | .860 | .643 | .499 | .372 | 74.4% |
| CE+CTC | oracle | .860 | .652 | .492 | .438 | 89.3% |
| CE+CTC | MMS_FA | .860 | .652 | .492 | .406 | 82.5% |
| CE+CTC | MMS-JA | .860 | .652 | .492 | .376 | 76.5% |

**Discrimination (paired bootstrap over 50 songs, 95% CI):**

- MMS_FA vs MMS-JA (same GAME notes): **+3.6pp [+2.8, +4.4] SIGNIFICANT** —
  the metric cleanly ranks aligners; CI is ~4x narrower than the effect.
- Aligner tax (oracle − MMS_FA): +3.4pp [+2.6, +4.2] significant.
- GAME vs CE+CTC (same aligner): +0.1pp n.s. — the note-model tie carries over.

**Reads.**

1. **The "all low scores, rotten-apples-fighting" worry is empirically refuted
   on the aligner axis**: the floor is set by COnPOff (~.50), the L tax is
   6–13pp (not a collapse), and aligner differences are 4x the bootstrap CI.
2. **The tax is NOT uniform across aligners** (oracle −5.7pp / MMS_FA −9.1 /
   MMS-JA −12.7) — that non-uniformity IS the discrimination. Across note
   models with the same aligner it is near-uniform (−9.1 vs −8.6), as
   predicted: shared aligner ⇒ shared tax ⇒ note-model ranking unchanged.
3. **MMS-JA < MMS_FA on clean a cappella re-confirmed through a completely
   independent metric path** (phone-boundary benchmark said the same; this is
   triangulation, not reuse).
4. **Oracle tax (−5.5pp, 11% of matched notes) = the metric's intrinsic
   attribution-noise floor**: matched note pairs whose onsets straddle a mora
   boundary within the 50ms tolerance. Future refinement target, honest caveat.
5. Product KPI reading: GAME x MMS_FA = **.408 ⇒ ~41% of karaoke bars fully
   correct** (right time, pitch, length, AND word) on the Kiritan protocol.

Artifacts: `conpoff_l.py`, `conpoff_l_results.json`.

## UltraSinger — first-ever COnPOff+L eval of a full-stack karaoke tool (2026-07-02)

`ultrasinger_eval.py` (reuses `conpoff_l.py` verbatim). UltraSinger is the only
open-source tool that produces a *complete* karaoke artifact (notes + pitch +
per-syllable lyrics + timing) end-to-end, and it had **zero published
quantitative evaluation**. This is its first.

**Setup asymmetry (must read).** Every other row in this file is **lyrics-known**:
MMS/oracle align the *given* GT phone sequence, so their L tax isolates *timing
attribution*. UltraSinger is **lyrics-UNKNOWN** — whisperx transcribes the
lyrics from audio itself — so its L tax folds *mis-heard characters* **and**
*time attribution* together. It is a strictly harder setting, and an honest
reading of a full-stack system, not an unfair one. That is why we report the
fully-fair note axes (COn/COnP/COnPOff — pitch-and-timing only, lyrics-agnostic)
next to COnPOff+L.

| system | setting | COn | COnP | COnPOff | **COnPOff+L** | P(L\|match) |
|---|---|---|---|---|---|---|
| GAME × MMS_FA | lyrics-known | .860 | .643 | .499 | **.408** | 81.6% |
| **UltraSinger** | lyrics-unknown | **.304** | **.120** | **.039** | **.002** | **6.0%** |

Kiritan N=50, gt_timefix, macro per-song F1. UltraSinger: **0 failed songs**.

**Discrimination (paired bootstrap, same 50 songs, 2000 iters, seed 7):**
UltraSinger − GAME×MMS_FA is significantly negative on **every** axis —
COn −.556 [−.588, −.516], COnP −.523 [−.569, −.475], COnPOff −.460
[−.501, −.417], **COnPOff+L −.406 [−.439, −.368] SIGNIFICANT**. The gap is
~13× the bootstrap CI; nothing marginal here.

**Reads.**

1. **UltraSinger struggles across the whole ladder on a-cappella Kiritan** — the
   collapse is not localized to the (harder) lyric axis; even the fully-fair
   **COn = .304** (bare onset+pitch note match) is a third of GAME's .860.
   **Caveat — a LANGUAGE confound touches every axis, not just L.** Source read
   (`midi_creator.py:153` → `create_midi_notes_from_pitched_data`): UltraSinger
   has **no pitch-onset note detector**. Every note's start/end is a *whisperx
   syllable boundary*; swift-f0 only fills the pitch value inside each
   whisper-defined window. So weak Japanese ASR timestamping/segmentation
   propagates into COn/COnP/COnPOff onsets, not only into the lyric label. The
   pitch *value* (the P) is language-agnostic (swift-f0); the *timing/segmentation*
   is whisper-driven and therefore language-sensitive. We CANNOT yet separate
   "clean a-cappella note transcription is intrinsically hard for this tool" from
   "non-English ASR-driven segmentation is weak." **Decisive control = run the
   same pipeline on an English a-cappella note-GT set (vocadito, 40 clips).** If
   COn jumps on English, the story is "ASR-segmentation bottleneck on JA"; if it
   stays low, "no pitch-onset detector ⇒ weak regardless of language."
   **CONTROL RAN (`../vocadito/RESULTS.md`, 2026-07-02): confound confirmed
   real.** English COn = .492 [CI .438–.549] vs JA .304 (~1.6×, CI-separated),
   and within one dataset the CJK language (Mandarin .330) lands right at Kiritan
   JA (.304) while Latin-script languages sit at .40–.49. So a large part of this
   COn collapse is whisper-segmentation-on-CJK, NOT intrinsic. **Two caveats
   keep it honest:** (a) even English COnPOff = .095 collapses via the
   language-agnostic pitch/offset conditions (swift-f0), so Kiritan's near-zero
   COnPOff+L was never mainly a lyric penalty — COnPOff is on the floor in every
   language; (b) vocadito (~14 s clips) vs Kiritan (~4 min) is cross-dataset, so
   the within-vocadito Mandarin≈Kiritan gradient (not the raw EN-vs-JA gap)
   carries the language claim.
2. **Two mechanisms, both diagnosed on the raw output** (see `PROGRESS.md`):
   (a) **over-segmentation** — UltraSinger emits **13,361 notes vs GT's 10,370**
   (median est/ref ratio 1.2, up to 2.57×), because it splits held notes into a
   note + a run of `~` continuation segments; the extra notes dilute precision
   and drag F1 down. (b) **syllable over-holding** — whisperx sometimes lumps a
   long stretch under one syllable (song 01: a single `き` at 19.1s then 16+ `~`
   held notes to 32s, while GT has se/re/be/… there), so the est syllable
   timeline is badly misaligned with GT → the L axis reads near-zero honestly.
3. **P(L | match) = 6.0%** (vs GAME×MMS_FA 81.6%): of the few notes that do pass
   COnPOff, almost none also carry the right mora — combined mis-hearing +
   attribution error, exactly the lyrics-unknown double tax.
4. The lyric content itself is *plausible* (song 01 syllables read
   `き っ と 飛 べ ば 空 ま で 届 く…`), so this is not a total transcription
   failure — it is a **timing/segmentation** failure that COnPOff+L surfaces
   quantitatively where a lyrics-only WER would miss it.

**Octave-convention gotcha (recorded for reuse):** this UltraSinger build writes
notes as `midi = ultrastar_note + 48` (its `ultrastar_converter.py` comments
"C4 == 48"), NOT the +60 the generic UltraStar spec / `parse_ultrastar.py`
assumes. The mandated octave check caught it: +60 put the EST−GT pitch delta on
+12 (63 exact-+12 notes on song 01); +48 recentres it on 0. The eval uses +48.

**Alignment audit (is .304 a timebase bug? No).** `../vocadito/alignment_audit.py`
checks both DBs: harness identity est=GT ⇒ COn 1.000; beat→sec formula proven
equal to UltraSinger's own `ultrastar_converter`; drift 0.00 ms/s (no scale
error); COn peaks at Δ=−0.04 s (real ~40 ms latency) lifting .304→.339 only.
The score is real — Kiritan onset precision .27 / recall .35: UltraSinger misses
~⅔ of onsets and most of what it emits is spurious, not a global misalignment.

**Protocol.** UltraSinger commit `e94d942` (v0.0.13.dev16) · whisperx 3.8.1
(model `large-v3`, `--language ja`) · pitch = swift-f0 0.1.2 (not CREPE) ·
separation = demucs 4.0.1 htdemucs (default; a-cappella input) · torch
2.8.0+cu128 on RTX 5090 (sm_120) · inputs = Kiritan wav resampled to
44.1kHz/16-bit/mono · **~34 s/song** (29–43 s), ~17 min for the batch · L chain:
UltraStar syllable → fugashi+UniDic reading → kiritan `japanese.table` phones →
`group_morae` first mora (`~` inherits the prior mora; 723/13361 = 5.4% of notes
un-convertible → "?" = honest L fail). Env note: torchcodec failed to load
(`libavutil.so` missing) on the later songs; UltraSinger fell back to another
decoder and completed — the `rc=1` exits are from the trailing MuseScore
sheet-render step (MuseScore not installed), *after* notes were written, so they
do not affect the transcription. All 50 note files verified non-truncated
(170–480 notes each).

Artifacts: `ultrasinger_eval.py`, `ultrasinger/{PLAN,PROGRESS}.md`,
`ultrasinger/{syllable_to_mora,build_pred,run_batch.sh}`,
`ultrasinger/{ultrasinger_pred,ultrasinger_morae,ultrasinger_results}.json`.
(Raw UltraStar `out/` and the resampled audio are gitignored.)

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
