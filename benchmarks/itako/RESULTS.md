# Itako Benchmark — note transcription + phone boundary (2026-06-20)

Sister benchmark to `../kiritan`. Tohoku **Itako** singing DB (東北イタコ歌唱
データベース): 50 Japanese pop songs, professional solo voice (木戸衣吹),
studio **a cappella** (96 kHz/24-bit, Neumann u87ai, ~51 min voiced) — same
family / creator (Morise) as Kiritan, so the protocol mirrors Kiritan (clean solo
vocal ⇒ **no source separation**, unlike polyphonic mir-st500). Every number below
was independently re-derived by a 7-agent adversarial verification pass (see
*Verification*).

## Result (N=50, zero-shot, a cappella, official MIR-ST500 `evaluate.py`)

| system | GT | COn | COnP | COnPOff |
|---|---|---|---|---|
| **GAME-1.0-large** `-l ja` | raw | 0.824 | 0.494 | 0.400 |
| GAME-1.0-large `-l ja` | defect-fixed | **0.832** | 0.510 | **0.414** |
| **CE+CTC** (`ctc_ce#3_98`, Mandarin-trained) | raw | 0.828 | 0.509 | 0.374 |
| CE+CTC | defect-fixed | **0.839** | **0.525** | 0.386 |

raw GT = 9,906 sung notes; onset 50 ms, pitch 50 cents, offset max(50 ms, 0.2·dur).
"defect-fixed" = itako50 octave (−12) + itako01 onset lag (+50 ms); see below.

**Reads.**

1. **Clean-Japanese-a-cappella onset reproduces across the sister DB, and the
   GAME≈CE+CTC tie holds.** COn 0.824 (GAME) ≈ 0.828 (CE+CTC) raw, 0.832 ≈ 0.839
   defect-fixed — the Kiritan head-to-head (0.862 ≈ 0.860), CE+CTC edging COnP,
   reproduces on a second independent Japanese DB.
2. **The COn→COnP gap is note-level score-vs-sung-pitch — a LABEL issue — not
   per-song key errors.** Per-song transposition correction is inert (COnP
   0.494→0.492). Instead, on the ~1,920 onset-matched notes where GAME = GT−1
   semitone, the independent RMVPE F0 (DeepUNet continuous-pitch; GAME is a CQT
   note-grid) sides with **GAME's lower (sung) pitch 95.0% raw / 80.3%
   bias-corrected**. So the score-based GT is ~1 semitone **sharp** there ⇒
   **COnP understates the models' true sung-pitch accuracy.** (`adjudicate_pitch.py`.)
3. **A different GT-defect profile from Kiritan** (see *Cross-dataset*): Kiritan's
   defects were correctable whole-song transpositions; Itako's are an
   un-correctable note-level pitch gap plus two isolated songs.

## The GT audit (model-independent RMVPE, `audit_gt_rmvpe.py` / `audit_timeshift.py`)

- **No Kiritan-style per-song transpositions.** 45/50 songs offset 0; the 4 flags
  at +1 are borderline round-ups (medians ~0.5, residuals −31..−49¢) and
  correcting them moves COnP **<0.002** (slightly *down*). Global tuning residual
  +21.1¢ (the GT runs mildly sharp of the sung F0 — consistent with read #2).
- **itako50 = octave-high GT (a clean, dramatic single-song defect).** Onsets are
  perfect (COn 0.925) but raw **COnP = 0.000** — every note an octave off.
  Shifting −12 (both models agree exactly −12; RMVPE noisier at +12.6 on this
  highest song, pitch 68–85) restores COnP to 0.46 and lifts corpus COnP +0.010.
- **itako01 = ~50 ms onset lag.** Both models collapse (COn 0.43/0.31); a +50 ms
  GT shift agreed by **both** models restores COn 0.37→0.84 (+0.008 corpus COn).
- **No global time-shifts otherwise** (RMVPE-voicing xcorr finds none decisive).

## ⚠ Methodological note: itako03 / itako47 were OUR bug, not a DB defect

An earlier pass reported itako03/itako47 as "structural GT defects" (COn 0.19/0.26,
unrecoverable under any global shift+transpose). The adversarial review **refuted**
this: `build_itako_gt.py` originally read only the *first* `set_tempo` event. These
are the **only 2/50 songs with a mid-song tempo change** (itako03 100→170 BPM,
itako47 80→128 BPM), so their GT timeline stretched 1.5–1.7× after the change.
Honoring the full tempo map (`mido.merge_tracks`, fixed) recovers them to **COn
0.892 / 0.833** (above corpus average), confirmed model-independently (RMVPE
voicing corr 0.55→0.78; corrected GT span matches the `mono_label` voiced span to
<0.4 s). The lesson: a single-tempo MIDI parser silently corrupts exactly the
songs with tempo changes, and "no shift recovers it" is the signature of a
*linear* (tempo) divergence, not a constant offset. The fix lifted corpus COn
0.798→0.824 (GAME) / 0.801→0.828 (CE+CTC).

## Cross-dataset reads (Itako vs Kiritan)

Same RMVPE audit code/thresholds on both. **Itako is not "cleaner" than Kiritan —
it carries a *different* defect class:**

| | Kiritan | Itako |
|---|---|---|
| dominant GT defect | 16/50 clean per-song transpositions (+1 semitone) | note-level score-vs-sung-pitch (GT ~1 semitone sharp on ~19% of matched notes) |
| correctable? | yes — COnP 0.531→**0.644** after transpose+time fix | no — transpose correction inert; COnP ceiling stays ~0.51–0.53 |
| isolated defects | 5/50 time-shifted | 1 octave (itako50), 1 onset lag (itako01) |
| raw onset COn | 0.862 / 0.860 | 0.824 / 0.828 |

So Kiritan's GT errors are *fixable* (and COnP recovers); Itako's are mostly the
*irreducible* score-vs-performance pitch gap that COnP cannot see past — Itako
demonstrates the annotation-vs-audio thesis in its purest form (the models are
right ~80–95% of the time the GT says they're wrong on pitch).

## Note post-processing ablation (`game_postproc_ablation.py`, no GPU)

| variant | COn | COnP | COnPOff | notes |
|---|---|---|---|---|
| GAME raw | 0.824 | 0.494 | 0.400 | 10,668 |
| **min-dur absorb** (<100 ms → prev tail) | **0.828** | **0.507** | **0.411** | 9,603 |
| merge same-pitch + min-dur | 0.761 | 0.447 | 0.312 | 8,057 |

Reproduces Kiritan: min-dur fragment absorption is a uniform small win;
same-pitch merging is poison (−6.7pp COn — consecutive same-pitch notes on
different syllables are real in singing).

## Phone-boundary benchmark (MMS, `phone_boundary_itako.py`)

Separate axis: GAME/CE+CTC emit notes; MMS emits phone timing, scored vs
`mono_label` (17,752 phones, `pau`/`br`/`sil` ignored). HTK 100-ns labels +
leading `sil` handled in the Itako-adapted reader.

| system | boundary MAE | median | BER@50ms |
|---|---:|---:|---:|
| **MMS_FA** (torchaudio bundle) zero-shot | **0.078 s** | 0.048 s | 48.0% |
| MMS-JA (karaoke ckpt) zero-shot | 0.113 s | 0.055 s | 53.3% |

MMS_FA edges MMS-JA on clean Japanese phone boundaries (mirrors Kiritan).
**SOFA not run: its released JPN model was trained on Itako** (`data_providers.md`
lists "Tohouku itako") ⇒ contaminated, so only MMS is a clean zero-shot number.

## ROSVOT (M4Singer ckpt, RWBD active) — partial, negative result

N=27 (OOM under a concurrent GPU job; resumable): COn **0.297 @50 ms → 0.723
@100 ms**. The same large 50→100 ms jump (+42pp) as Kiritan/mir-st500 — boundary
jitter without word-boundary conditioning; not competitive with GAME/CE+CTC. Full
N=50 can be finished when the GPU is free.

## Data provenance

- **Labels:** GitHub [`mmorise/itako_singing`](https://github.com/mmorise/itako_singing)
  (commit pinned in `../../../../itako/itako_singing/PROVENANCE.txt`), verified
  **byte-identical** to the labels bundled in the gated DB zip.
- **Audio:** gated, https://zunko.jp/itadev/login.php (non-commercial research,
  改正著作権法 30-4; not redistributable). `wav/itako01..50.wav`, 96 kHz mono,
  already match the GT keys.

## Note GT construction

`build_itako_gt.py`. Itako encodes **breaths as out-of-range note events** (the
musicXML `/br/` lyric needs a duration); sentinel pitch varies per song
(81/47/89/48…), so we drop MIDI notes overlapping a `br`/`pau`/`sil` `mono_label`
span by >50% (near-1:1 match). 10,886 raw → **9,906 sung notes**. Uses the full
tempo map (see the methodological note above).

## Artifacts

- GT: `itako_gt.json` (clean), `itako_gt_raw.json` (unfiltered), `gt_defectfix.json`
- transcription: `run_pipeline.sh`; `eval_{game,cectc}_{raw,defectfix}.txt`
- audit: `extract_f0.sh`, `audit_gt_rmvpe.py`, `audit_timeshift.py`,
  `adjudicate_pitch.py` (+ `.json` outputs), `diagnose_persong.py`
- ablation: `game_postproc_ablation.py`; phone-boundary: `phone_boundary_itako.py`
- verification: `synth_verify.workflow.js`
- source: `/home/kojiek/side_projects/itako/itako_singing/`

## Verification

A 7-agent workflow (`synth_verify.workflow.js`) independently re-derived every
headline number (all matched to ≤0.0005) and sent 3 skeptics at the central
claims. It **caught the itako03/47 tempo-map bug** (re-reported above as fixed),
**refuted a shared-bias explanation** of the −1-semitone finding (control bias
only −15.7¢ vs the −100¢ a 1-semitone shared bias would need; on GAME=GT+1 notes
RMVPE follows GAME *upward* 70% — impossible for a flat bias), and corrected
"Itako is cleaner" → "different defect type." Numbers here reflect those fixes.

## COnPOff+L (2026-07-02)

**Verdict — the aligner discriminability REPRODUCES on the second DB.** MMS_FA
beats MMS-JA on COnPOff+L by **+2.2pp CI[+1.2, +3.2]** (defectfix) / **+2.0pp
CI[+1.1, +3.0]** (raw), both **significant**, and this holds *independently* of
Itako's score-vs-sung-pitch GT defect (the defect lives in COnP, which the +L rung
sits on top of unchanged — mora attribution is a pure timing question). The
oracle→MMS_FA tax is **+3.4pp CI[+2.6, +4.2]** (defectfix), matching Kiritan's
+3.4pp almost exactly. The COnPOff→+L drop is **near-constant across note models**
at a fixed aligner (oracle −2.9/−2.2pp, MMS_FA −6.2/−5.4pp, MMS-JA −8.4/−6.9pp for
GAME/CE+CTC) — the tax is an *aligner* property, not a note-model one, as on Kiritan.

Script: `conpoff_l.py` (adapted from `../kiritan/conpoff_l.py`: mono_label ÷1e7,
raw+defectfix GT loop, itako01 GT-morae +50 ms under defectfix). Same four-condition
matching (onset 50 ms / pitch 50 cents / offset max(50 ms, 0.2·dur) / lyric mora
label), same paired bootstrap. Full JSON: `conpoff_l_results.json` (both variants).

### Ladder (N=50, macro per-song F1) — **primary = defectfix**

| note model | aligner | COn | COnP | COnPOff | COnPOff+L | P(L\|match) |
|---|---|---|---|---|---|---|
| GAME | oracle | 0.828 | 0.507 | 0.411 | 0.382 | 92.8% |
| **GAME** | **MMS_FA** | 0.828 | 0.507 | 0.411 | **0.348** | 84.6% |
| GAME | MMS-JA | 0.828 | 0.507 | 0.411 | 0.326 | 79.2% |
| CE+CTC | oracle | 0.838 | 0.525 | 0.386 | 0.364 | 94.3% |
| CE+CTC | MMS_FA | 0.838 | 0.525 | 0.386 | 0.332 | 85.7% |
| CE+CTC | MMS-JA | 0.838 | 0.525 | 0.386 | 0.316 | 81.6% |

raw variant (secondary robustness): GAME COnPOff+L oracle/MMS_FA/MMS-JA =
0.370/0.338/0.318; CE+CTC = 0.355/0.323/0.310. COnPOff = GAME 0.397 / CE+CTC 0.374.

### Discrimination (paired bootstrap, 95% CI, primary defectfix / secondary raw)

| contrast | defectfix | raw |
|---|---|---|
| aligner A/B (GAME: MMS_FA − MMS-JA) | **+0.0219 [+0.0122, +0.0317]** sig | **+0.0202 [+0.0106, +0.0299]** sig |
| aligner tax (GAME: oracle − MMS_FA) | +0.0335 [+0.0258, +0.0416] sig | +0.0324 [+0.0247, +0.0408] sig |
| note model A/B (MMS_FA: GAME − CE+CTC) | +0.0160 [+0.0005, +0.0313] sig | +0.0144 [−0.0004, +0.0292] **n.s.** |

### Anchor validation (COnPOff, no-L rung vs RESULTS.md `evaluate.py`)

| | script | RESULTS.md | Δ |
|---|---|---|---|
| GAME raw | 0.397 | 0.400 | 0.003 |
| GAME defectfix | 0.411 | 0.414 | 0.003 |
| CE+CTC raw | 0.374 | 0.374 | 0.000 |
| CE+CTC defectfix | 0.386 | 0.386 | 0.000 |

All ≤0.003 (well under the ±0.015 stop-threshold). The tiny GAME gap is the known
bipartite-match vs official-`evaluate.py` difference (same as Kiritan); CE+CTC is
exact. COn also matches (GAME raw 0.819 vs 0.824, CE+CTC raw 0.828 = exact).

### Cross-dataset (Itako vs Kiritan, same metric code)

| | Kiritan (gt_timefix) | Itako (gt_defectfix) |
|---|---|---|
| GAME×MMS_FA COnPOff+L | 0.408 | 0.348 |
| aligner A/B (MMS_FA − MMS-JA) | +0.036 [+2.8, +4.4] | +0.022 [+1.2, +3.2] |
| oracle tax (oracle − MMS_FA) | +0.034 | +0.034 |
| tax cross-note-model | near-constant | near-constant |

Itako's absolute COnPOff+L is ~6pp lower than Kiritan — expected, since it inherits
Itako's suppressed COnP (score-vs-sung-pitch GT defect, ~19% matched notes GT sharp
1 semitone; see *Read #2*). The **relative** aligner ordering (oracle > MMS_FA >
MMS-JA) and its significance reproduce; the aligner gap is smaller (+2.2 vs +3.6pp)
but firmly non-zero. Terminology note — two distinct "taxes", both real: (1)
**aligner tax** = oracle − MMS_FA on COnPOff+L = +3.4pp on BOTH DBs (exact
cross-DB match); (2) **attribution noise floor** = the COnPOff→+L drop under the
oracle aligner (perfect phone times, so the residual is pure note-onset-vs-mora-
onset attribution ambiguity) = Kiritan −5.7/−5.4pp (GAME/CE+CTC, ≈5.5pp) vs Itako
−2.9/−2.2pp. The earlier "oracle tax ≈5.5pp" quote was framing (2), not an error.
Itako's attribution floor is roughly HALF Kiritan's — attribution is cleaner here
(0/50 mora mismatches, see below).

### Mora mismatch & smoke checks

- **Mora sequence mismatch: 0/50 songs for BOTH aligners** (MMS_FA and MMS-JA emit
  the identical mora label sequence as mono_label on every song). So the +L rung
  isolates *timing attribution* cleanly here — no label-sequence confound, unlike a
  DB where aligners drop/insert morae. This strengthens the "L = right bar, right
  word" reading versus Kiritan.
- Smoke (a) itako01 3-way anchor: mono_label `da` @5.38s ↔ mms_fa `da` @5.36s ↔
  GT note @5.40s — aligned. Smoke (b) itako05: group_morae=195 == vowel+N in
  mono_label=195. Smoke (c) itako50 raw COnP=0.000 (octave-defect signature),
  COn=0.925 — as expected.

### Note (surprise, reported straight)

The **note-model** contrast (GAME − CE+CTC on MMS_FA) flips from **significant on
defectfix** [+0.0005, +0.0313] to **n.s. on raw** [−0.0004, +0.0292] — the CI
lower bound sits right on zero. This is a knife-edge, not a reversal: GAME never
loses. The defect fix (itako50 octave + itako01 lag) marginally favours GAME's
onset placement, tipping the CI positive. The headline aligner claim is robust to
this; the note-model claim on Itako is best stated as "GAME ≥ CE+CTC, marginal."
No unexpected sign flips elsewhere (MMS_FA never lost to MMS-JA).
