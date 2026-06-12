# MIR-ST500 GAME Raw Subset Result

Date: 2026-06-11

## Scope

This is a provisional MIR-ST500 test-subset run, not the complete official 100-song test score.

- Test IDs attempted: `401-500`
- Downloaded audio: `82`
- Failed downloads: `18`
- Failed IDs: `402, 406, 407, 417, 424, 425, 433, 437, 443, 445, 446, 449, 478, 479, 482, 484, 491, 495`
- Separator/pipeline: existing MIR-ST500 pipeline in `benchmarks/mir-st500/run_pipeline.sh`
- AST model: GAME large, `-l zh`, raw extract output, zero-shot
- Evaluation: official `benchmarks/singing_transcription_ICASSP2021/evaluate/evaluate.py`

## Result

Raw GAME large `-l zh`, zero-shot:

| metric | Precision | Recall | F1 |
|---|---:|---:|---:|
| COnPOff | 0.391935 | 0.433803 | 0.411018 |
| COnP | 0.622012 | 0.693557 | 0.654525 |
| COn | 0.695277 | 0.776927 | 0.732318 |

Additional counts:

- Ground-truth notes: `26115`
- Transcribed notes: `29135`
- Evaluated songs: `82`

## Artifacts

- Evaluation output: `benchmarks/mir-st500/eval_game_raw_zh.txt`
- GAME JSON: `benchmarks/mir-st500/game_raw_zh.json`
- Pipeline log: `benchmarks/mir-st500/run_pipeline.live.log`
- Download log: `benchmarks/mir-st500/download.log`

## Interpretation

The useful comparison point is that GAME raw zero-shot is already in the public benchmark coordinate system: COn `0.732`, COnP `0.655`, COnPOff `0.411` on the 82-song subset.

The main weakness is offset/coverage, not onset/pitch. This matches the Chidori diagnosis: GAME-like note-level models get onset/pitch into a plausible range, while sustained-note tails and coverage still need a union/extension strategy or benchmark-specific training.

## Extend-Sustains Variant

`score_note_postfix.py --extend-sustains` was tested as an offset-only variant after GAME raw. It did not help the benchmark metric:

| variant | COn | COnP | COnPOff |
|---|---:|---:|---:|
| GAME raw | 0.732318 | 0.654525 | 0.411018 |
| GAME + extend-sustains | 0.732318 | 0.654525 | 0.408618 |

Interpretation: the heuristic is useful for karaoke display coverage, but it is not a precise AST offset model. MIR-ST500's offset condition, `max(50ms, 0.2*duration)`, penalizes over-extended tails. This supports using learned boundary calibration rather than display-oriented sustain extension for benchmark AST.

## Note-cleanup ablation (2026-06-11)

Same sweep as Kiritan (see that RESULTS.md for full table): min-dur absorb
(<100ms) lifts all three metrics slightly (COn .732→.733, COnP .655→.657,
COnPOff .411→.413, notes −1.2%); same-pitch merge is catastrophic
(COn →.650) because real repeated same-pitch syllables get destroyed.
MIR-ST500's spurious notes are mostly ≥150ms (median 330ms — separation
bleed / ad-lib class), so min-dur helps less here than on Kiritan.

## CE+CTC head-to-head (2026-06-12, same stems, N=82)

Wang & Jang's pretrained `ctc_ce#3_98` (TASLP 2022) run on OUR Mel-Band-
RoFormer stems via `benchmarks/ctc_ce_infer.py` (their Spleeter bypassed;
(vocal, mixture) channels rebuilt from our vocals+instrumental — true
same-front-end comparison). Their published number: COnPOff 0.574
(N=100, Spleeter, +30ms-shifted GT).

| system | GT | COn | COnP | COnPOff |
|---|---|---|---|---|
| **CE+CTC** | corrected | 0.779 | 0.728 | **0.554** |
| CE+CTC | +30ms (their protocol) | 0.763 | 0.714 | 0.519 |
| GAME raw -l zh | corrected | 0.732 | 0.655 | 0.411 |
| GAME raw -l zh | +30ms | 0.748 | 0.671 | 0.453 |

- **In-domain supervised wins decisively: +14pp COnPOff** — the learned
  onset/offset supervision (the professor's point) is exactly where GAME
  zero-shot loses; onsets are much closer (0.78 vs 0.73).
- Our 0.554 vs their published 0.574 (different separation, N=82 vs 100)
  — consistent; validates our harness and their result.
- GT-convention sensitivity: the ±30ms label shift alone moves COnPOff by
  ±4pp in OPPOSITE directions for the two systems (GAME agrees better with
  +30ms, CE+CTC with unshifted) — at 50ms tolerance, annotation conventions
  are a first-order term.
- Dead-link retry (android client): 0/18 recovered — N=82 is final unless
  the authors' cached audio is obtained by email.

## ROSVOT out-of-the-box (2026-06-12) — large negative result

ROSVOT (ACL 2024, official checkpoints, M4Singer-trained) run on the same
82 stems WITHOUT word-boundary annotations (RWBD predictor active, which is
the honest "no annotations available" condition for in-the-wild use):

| tolerance | COn | COnP | COnPOff |
|---|---|---|---|
| 50ms | 0.219 | 0.169 | 0.108 |
| 100ms | 0.432 | — | — |

Far below its published zero-shot 0.474 COnPOff. Diagnosis (see Kiritan
section): without real word_durs conditioning its note boundaries are
coarse — Kiritan COn jumps 0.41→0.75 going 50→100ms tolerance. ROSVOT is
an annotation tool that assumes word boundaries exist; out-of-the-box on
unannotated audio it is not competitive with GAME or CE+CTC. (A fair
best-case variant — feeding our MMS mora boundaries as word_durs — is
possible future work; our pipeline produces exactly that input.)

## The chidori-gold reference question (york135's challenge, 2026-06-12)

york135 (CE+CTC author) flagged: "COn 0.55 @100ms should have raised alarm;
the reference is almost certainly wrong; this song should be >=0.8 @100ms."
Verified empirically:

| reference | CE+CTC COn@100ms | GAME union COn@100ms |
|---|---|---|
| our chidori humangold (as-is) | 0.513 | 0.528 |
| + global shift sweep (best) | 0.583 | — |
| onsets re-anchored to MMS mora onsets | 0.580 | 0.568 |

BOTH models cap at ~0.55-0.58 against our gold under every anchoring —
while the same two models score 0.78-0.86 COn on MIR-ST500/Kiritan.
Quantified: 54% of gold onsets differ >50ms from acoustic (MMS) onsets;
re-anchoring fixes the easy part but the **note inventory itself**
(mora-grid segmentation, karaoke-bar merge conventions, melisma handling)
is not MIREX-convention singing transcription. Conclusion: our gold is a
PITCH gold (note-level majority KPI), exactly as docs/pitch-benchmark.md
always stated — absolute COn against it is meaningless for any model, and
cross-model rankings on it do not transfer. The proper absolute instrument
is MIR-ST500/Kiritan, where york135's >=0.8 intuition holds for his model.

## GAME seg-threshold 0.3 cross-validation (2026-06-12)

Kiritan sweep found seg 0.2->0.3 helps; confirmed here on separated vocals
(N=82): COn .732->.740, COnP .655->.669, COnPOff .411->.416. Both domains
agree -> GAME_SEG_THRESHOLD=0.3 promoted into run_game_chain (pinned).
