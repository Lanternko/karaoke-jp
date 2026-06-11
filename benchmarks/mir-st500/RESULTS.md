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
