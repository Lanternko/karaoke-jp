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

## Pitch — why is COnP only *half* of COn? (Not an offset bug; two real causes)

The reasonable objection: "even if onsets are rough, a dedicated f0 tracker
should get *pitch* right, so COnP shouldn't halve COn." Investigated:

- **No pitch-offset bug.** Over onset-matched pairs, median (est−gt) = **0.00
  semitone / 0 cents** (both DBs) — no octave/transpose/reference error; the +48
  convention is correct. The failures are *scattered* ±1-semitone WRONG-NOTE
  errors (vocadito Δ-histogram: 0→200, ±1→163, ±2→43…), not a shift. (A perfect
  semitone transcriber would score ~100 % within 50 c, since every value is ≤50 c
  from its own nearest semitone — so semitone quantization is NOT the ceiling;
  wrong-note picks are.)
- **Cause 1 — `quantize_to_key=True` default (Settings.py:29, no CLI flag) — but
  its sign is DATASET-DEPENDENT.** It snaps every note to a *detected* musical
  key. **Ablation (re-run with `quantize_to_key=False`, both DBs, N=full):**

  | | COn | COnP | COnPOff | P(pitch OK \| onset) |
  |---|---|---|---|---|
  | vocadito English — key-quant ON (default) | .492 | .248 | .095 | 46.1 % |
  | vocadito English — key-quant OFF | .499 | **.287** | .104 | **54.7 %** |
  | Kiritan JA — key-quant ON (default) | .304 | **.120** | **.039** | **37.2 %** |
  | Kiritan JA — key-quant OFF | .300 | .108 | .034 | 33.9 % |

  On **clean-pitch English the key-snap HURTS** (correct near-miss notes pushed
  to a wrong scale degree; OFF is +4 pp COnP). On **noisy-pitch Kiritan it HELPS**
  — the snap *regularizes* the wild-error tail (below) back toward scale tones, so
  ON (the default) is already the better setting and OFF is −1.2 pp. **So the
  main (Kiritan) COnP is NOT a fixable-default artifact** — the default is already
  favorable there; key-quant is not what's depressing it.
- **Cause 2 — mode-over-a-misaligned-window (inherent, dominant on Kiritan).**
  Per-note pitch = the *mode* of per-frame `librosa.hz_to_note` labels
  (midi_creator.py:71,148) across the **whisper-defined** note window. When that
  window is wrong/over-held (the same segmentation defect that tanks COn), the
  mode is taken over the wrong audio → wrong note. This is why even key-quant-OFF
  English stalls at 55 %, and why Kiritan (worse windows) has a genuine wild tail.
  (swift-f0 per se is fine; UltraSinger never exposes its sub-semitone value — it
  snaps to a note name per frame, then modes, then key-snaps.)

### How wrong is the pitch — near-miss or wild? (onset-matched notes)

| |Δ| | vocadito English | Kiritan JA |
|---|---|---|
| ≤0.5 st (≈right) | 46 % | 37 % |
| 0.5–1.5 st (**off by ~1 semitone, near-miss**) | 38 % | 25 % |
| 1.5–2.5 st | 10 % | 18 % |
| 2.5–6.5 st (moderate) | 6 % | 11 % |
| ≥6.5 st incl. octave (**wild**) | **0.6 %** | **~9 %** |
| median \|Δ\| | 56 cents | 100 cents |

**English pitch is overwhelmingly near-misses** (84 % within one semitone, wild
<1 %): "roughly right, tipped over the 50 c line by semitone-quantization +
key-snap" — which is exactly why turning key-quant off recovers a chunk. **Kiritan
is mostly near-miss too but carries a real ~9 % wild tail** (half-octave to
octave-plus errors) from segmentation windows landing on the wrong audio — which
is why key-snap *helps* it. Octave errors specifically are only 0.4 % on both
(confirms the +48 convention: a wrong octave convention would spike here).

**Net:** COnP halving COn is, on English, ~⅓ self-inflicted key-quant
(recoverable) + ~⅔ segmentation; on **Kiritan it is essentially all segmentation**
(key-quant already helps). Neither is a harness bug. Artifacts:
`*_pred_nokey.json`.

### Is the wrong pitch "right note, wrong position" or "genuinely wrong"?

Decomposition of onset-matched Kiritan notes (stable across 0–50 ms overlap
margins): does the est pitch exist in a GT note that temporally overlaps the est
note's own window?

| onset-matched note | share | meaning |
|---|---|---|
| pitch RIGHT (COnP pass) | 37 % | — |
| right pitch, **wrong slot** | ~15 % | est pitch IS a real GT pitch overlapping its window → position/attribution error, *fixable by fixing timing* (hypothesis A) |
| **fabricated** | ~47 % | est pitch matches NO overlapping GT note → a pitch never sung there; repositioning cannot fix it |

So of the ~63 % that fail pitch, only ~¼ are "right pitch, wrong position"; ~¾
are **genuinely wrong values**. **Causal test of "wrong window ⇒ wrong pitch"
(hypothesis B):** stratify pitch-accuracy by how many GT notes the est window
overlaps — if wide windows caused it, narrow ones would be accurate. They are
not: span=1 (isolated) 36 % vs span≥5 (dense) 24 % — a real but *secondary*
effect. Even the cleanest isolated est note is only ~36 % pitch-correct. So the
dominant cause is neither pure mis-position (A, ~15 %) nor purely the window
feeding a bad mode (B, secondary): it is **swift-f0's per-frame estimate snapped
to a semitone then mode-pooled producing a value that was never sung**, which
happens even for narrow isolated notes. (Structure: est 13 361 vs GT 10 370
notes, 1.29×; 53 % of est notes < 60 ms = the `~`-continuation fragmentation.)

### Control — GAME on the SAME Kiritan audio & GT (is the pitch even recoverable?)

Yes. A dedicated note-transcription model on the identical clips (given onset):

| model (Kiritan, onset-matched) | P(pitch OK) | median \|Δ\| | ≤0.5 st | ~1 st | wild ≥6.5 st |
|---|---|---|---|---|---|
| **GAME** (joint note+pitch AST) | **74.1 %** | **0 cents** | 74 % | 22 % | **0.2 %** |
| UltraSinger (lyric-window + swift-f0 mode) | 37.2 % | 100 cents | 37 % | 25 % | **8.8 %** |

GAME gets **median 0 cents** and ~0 wild errors on the very same audio ⇒ the pitch
is fully recoverable; UltraSinger's collapse is **architectural, not an audio
difficulty**. The gap's root cause: GAME estimates note boundaries and pitch
*jointly from the acoustics* (each note is a pitch-coherent region), whereas
UltraSinger reads a single pitch off a **lyrics-defined** window as the mode of
semitone-snapped swift-f0 frames — so wrong/over-held whisper windows (and swift-f0
octave slips) produce the 8.8 % wild tail GAME never has. UltraSinger never does
note-level pitch estimation; it does lyric segmentation and then reads off a pitch.

## UltraSinger architecture — exact pipeline (from source)

Every read above follows from *how* UltraSinger builds a note. It never detects a
note acoustically; it segments **lyrics**, chops them on a **metrical grid**, and
reads a pitch off each cell. The chain (file:line):

```
whisperx ─▶ word timestamps ─▶ hyphenate to SYLLABLES        (UltraSinger.py:84 add_hyphen_to_data)
                                    │  ← the onset/offset whisper gives is a LYRIC unit, not a note
                                    ▼
 split_syllables_into_segments      (UltraSinger.py:248) — any syllable longer than a
   16th note is cut into [first cell] + a train of "~" cells, each ONE 16TH NOTE long
   (get_sixteenth_note_second(bpm)).  ← a fixed METRICAL grid, not acoustic. This is where
                                         the "~" continuation notes and the 40 ms fragments come from.
                                    ▼
 per cell [start,end] → create_midi_note_from_pitched_data   (midi_creator.py:117)
   frames in window → high-confidence only → librosa.hz_to_note() snaps EACH frame to the
   nearest semitone (cents discarded) → most_frequent() = MODE → quantize_note_to_key()
                                    ▼
 merge_syllable_segments (UltraSinger.py:319) — merge adjacent cells with the SAME pitch
```

**Two consequences that drive every number in this file:**

1. **Note onset/offset are lyric+metrical, never acoustic.** No boundary comes
   from the pitch changing; it comes from a syllable edge or a 16th-note gridline.
   Hence the ~40 ms latency (§Alignment audit), the 40 ms median note length, and
   the 53 % `<60 ms` fragments — and, on CJK, the collapse when whisper syllable
   timing is weak (§language confound).
2. **Pitch is a by-product of that grid**, not an estimate: the *mode* of
   semitone-snapped swift-f0 frames over a window not aligned to the note. When
   the window straddles a real note change, the mode returns a value that was
   never sung (§the ~47 % "fabricated" pitches). Contrast GAME, which estimates
   note boundaries and pitch *jointly from the acoustics* (median 0 cents,
   §GAME control): each GAME note is a pitch-coherent region; each UltraSinger
   note is a lyric/metrical cell with a pitch stamped on afterward.

Crux for the paper: a **pitch-only** benchmark (e.g. MIR-ST500 COnP) or a
**lyrics-only** WER each hides this; only a joint note×lyric metric (COnPOff+L)
surfaces that UltraSinger's pitch fails *because* it is bolted onto lyric
segmentation.

## Alignment audit — is the low score a harness/parse bug? (No.)

Prompted by the reasonable worry that COn ~.5 is "too low to be real, must be a
timebase/parse misalignment." Four checks, all pass:

1. **Harness identity.** Feed GT as its own prediction ⇒ COn/COnP/COnPOff =
   **1.000 / 1.000 / 1.000** (both vocadito and Kiritan). GT shifted +40 ms ⇒
   COn still 1.000 (inside the 50 ms window, by design); +60 ms ⇒ 0.151 (drops,
   as it must). The matcher is not deflating anything.
2. **Beat→second conversion is provably UltraSinger's own.** `parse_ultrastar`
   uses `spb = 60/(#BPM·4)`, `t = GAP/1000 + beat·spb`; UltraSinger's
   `ultrastar_converter.py` computes `beat_to_second(beat, #BPM·4) + GAP/1000` —
   identical. No BPM/scale error.
3. **No time drift.** Matched-pair regression of (est_onset − gt_onset) vs time:
   slope **+0.04 ms/s** (≈0 over a 14 s clip) ⇒ pure constant offset, not a
   stretching/scale bug.
4. **Global-offset sweep.** COn peaks at **Δ = −0.04 s** on BOTH datasets
   (vocadito English .492→.532, Kiritan .304→.339) with a sharp, symmetric
   falloff — i.e. a real but small **~40 ms systematic latency** (UltraSinger
   onsets land slightly late; mean matched Δ +27 ms). Correcting it lifts the
   score modestly and changes no conclusion (best-aligned English .53 is still
   far below GAME's .86).

**So the score is real, not a misalignment.** Even perfectly offset-corrected,
onset **precision .60 / recall .49** (English): UltraSinger misses ~half the GT
onsets (it under-segments/merges) and ~40 % of what it emits is spurious. The
40 ms latency is within the 50 ms tolerance's purpose (onset-convention noise)
and is reported as a robustness figure, not applied to the headline. Diagnostic:
`alignment_audit.py`.

## Protocol

UltraSinger commit `e94d942` · whisper large-v3 · per-clip `--language` from
vocadito metadata (English→en, French→fr, Tagalog→tl, Spanish→es, Catalan→ca,
Mandarin→zh; mixed/none→autodetect) · swift-f0 pitch · mean ~15 s/clip, 40/40
succeeded, 0 failures. Scripts: `vocadito_batch.sh`, `build_pred_vocadito.py`,
`vocadito_eval.py` (reuses `conpoff_l`). Audio + `out/` are gitignored
(regenerable); tracked = GT JSON, pred JSON, results JSON, scripts.
