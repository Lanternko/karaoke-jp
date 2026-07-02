# vocadito — English-control for the UltraSinger language confound (2026-07-02)

**Why this exists.** On Kiritan (Japanese a cappella) UltraSinger scored COn .304
(§UltraSinger in `../kiritan/RESULTS.md`). Because UltraSinger has **no
pitch-onset note detector** — every note boundary is a whisperx syllable
boundary (`third_party/UltraSinger/src/modules/Midi/midi_creator.py:153`
`create_midi_notes_from_pitched_data`; swift-f0 only fills the pitch value inside
each whisper-defined window) — that collapse could be *Japanese-ASR-driven*
rather than intrinsic. This is the control: the same pipeline on English (and 6
other languages) with note-level GT.

**Data.** vocadito (Bittner et al., ISMIR-LBD 2021; Zenodo 5578807): 40 short
(~14 s) solo-vocal clips, 44.1 kHz mono, 7 languages, note GT from 2 annotators
(A1/A2 — vocadito reports low inter-annotator agreement, so both are evaluated).
GT note CSV = (onset_s, pitch_Hz, duration_s); Hz→MIDI = 69+12·log₂(f/440).
Same matching harness and tolerances as every Kiritan row (`conpoff_l.match_notes`:
onset 50 ms, pitch 50 cents, offset max(50 ms, 0.2·dur)), same +48 octave
convention. No L axis (no aligner for English; not the question).

## Result (macro per-clip F1)

| group | n | COn | COnP | COnPOff | est/ref |
|---|---|---|---|---|---|
| **English** | 17 | **.492** | .248 | .095 | 0.82 |
| French | 9 | .399 | .202 | .074 | 0.80 |
| Tagalog | 6 | .438 | .233 | .098 | 0.84 |
| Catalan | 2 | .403 | .238 | .072 | 0.54 |
| **Mandarin** | 2 | **.330** | .049 | .021 | 1.03 |
| ALL 40 | 40 | .431 | .217 | .083 | 0.78 |
| — *Kiritan JA (reference)* | 50 | *.304* | *.643†* | *.499†* | *1.2* |

(annotator A1; A2 near-identical — English COn .497, ALL .445. Full JSON:
`vocadito_results.json`. †Kiritan COnP/COnPOff are GAME's own note model, not
comparable to UltraSinger's swift-f0 pitch; only **COn** is the apples-to-apples
onset axis, since COn ignores pitch entirely.)

**English COn = .492, 95% CI [.438, .549]** (n=17, 2000-iter bootstrap). The CI
lower bound (.438) is well above Kiritan's .304.

## Verdict — the language confound is REAL (your hypothesis holds), with two caveats

1. **UltraSinger does markedly better on English than on Japanese onsets.**
   .492 vs .304 ≈ **1.6×**, CI-separated. And the *within-vocadito* gradient (one
   dataset, one recording protocol — the clean internal control) tracks script /
   ASR-segmentation difficulty: Latin-script, space-delimited languages
   (English .49 / Tagalog .44 / Catalan/French .40) sit high, and the **one CJK
   language, Mandarin, drops to .330 — right at Kiritan Japanese's .304.** So a
   large part of the Kiritan COn collapse is *whisper-segmentation-on-CJK*, not
   "clean a cappella note transcription is impossible." The earlier read #1 that
   framed it as intrinsic was over-claimed; corrected.

2. **But it is not *only* language, and English is not actually good.** Even at
   its English best, COn .492 is still ~⅗ of GAME's .860, and the ladder still
   **collapses to COnPOff .095 on English** — the pitch (50 cents) and offset
   conditions, which are language-agnostic (swift-f0), gut it independently of
   ASR. So the near-zero Kiritan COnPOff+L (.002) was never mainly a lyric-axis
   penalty: COnPOff is already on the floor even in English, so +L had nothing to
   subtract. UltraSinger's weakness is **two stacked failures** — (a) CJK
   ASR-segmentation drives bad onsets, (b) swift-f0-in-whisper-windows gives poor
   note pitch/boundaries in *every* language.

3. **Cross-dataset caveat (honest).** vocadito (~14 s clips) vs Kiritan (~4 min
   studio songs) differ in length and recording condition, and the est/ref
   direction flips (vocadito under-produces 0.78, Kiritan over-produces 1.2 —
   likely a clip-length × held-note-subdivision effect). So the raw English-vs-JA
   number conflates language with dataset. The **within-vocadito Mandarin ≈
   Kiritan** gradient is what carries the language claim, because it holds dataset
   and protocol fixed.

**Bottom line for the paper.** UltraSinger's Kiritan number is depressed by a
genuine non-English (CJK) ASR-segmentation penalty — so the honest framing is
"UltraSinger is bottlenecked by whisper-anchored segmentation, severely so on
CJK," **not** "it cannot transcribe clean a cappella." But COnPOff+L's core
reading survives either way: the joint metric surfaces a real, quantifiable
failure that lyrics-only WER or pitch-only COnPOff would each miss.

## Protocol

UltraSinger commit `e94d942` · whisper large-v3 · per-clip `--language` from
vocadito metadata (English→en, French→fr, Tagalog→tl, Spanish→es, Catalan→ca,
Mandarin→zh; mixed/none→autodetect) · swift-f0 pitch · mean ~15 s/clip, 40/40
succeeded, 0 failures. Scripts: `vocadito_batch.sh`, `build_pred_vocadito.py`,
`vocadito_eval.py` (reuses `conpoff_l`). Audio + `out/` are gitignored
(regenerable); tracked = GT JSON, pred JSON, results JSON, scripts.
