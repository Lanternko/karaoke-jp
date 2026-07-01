export const meta = {
  name: 'itako-synth-verify',
  description: 'Independently re-derive Itako numbers, send 3 skeptics to refute the central GT-defect claims, then synthesize the verified Itako↔Kiritan story',
  phases: [
    { title: 'Reproduce', detail: 'independent re-derivation of note metrics, ablation, phone-boundary, and the RMVPE -1-mode adjudication' },
    { title: 'Challenge', detail: '3 skeptics attack the central thesis from distinct lenses' },
    { title: 'Synthesize', detail: 'final RESULTS section grounded only in surviving claims' },
  ],
}

const BM = '/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks/itako'
const EVAL = '/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks/singing_transcription_ICASSP2021/evaluate'
const PY = '/home/kojiek/venvs/karaoke-jp/bin/python'
const KIRITAN = '/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks/kiritan/RESULTS.md'

const CLAIMS = `Itako benchmark claims to verify/refute (all from N=50, official MIR-ST500 evaluate.py, onset 50ms):
C1 raw GT: GAME COn 0.798/COnP 0.476/COnPOff 0.384 (pred 10668); CE+CTC 0.801/0.492/0.362 (pred 8379).
C2 defect-fixed GT (gt_defectfix.json = itako50 octave −13 + itako01 +50ms consensus timing + borderline transposes): GAME 0.806/0.482; CE+CTC 0.812/0.501. (transposition-only correction gt_transcorr lifted COnP <0.002 = negligible.)
C3 RMVPE transposition audit: 45/50 songs offset 0; the 5 flags have residuals −31..−49¢ (medians ~0.5 = borderline round-ups, NOT clean ±1) except itako50 = +13 (octave). ⇒ Itako has NO Kiritan-style clean per-song transpositions.
C4 time-shift: itako01 recovers COn 0.37→0.84 under a +50ms shift agreed by BOTH models (real ~50ms GT onset lag); itako03 & itako47 do NOT recover under any shift+transpose (cap COn ~0.22 / ~0.30) ⇒ structural GT/audio mismatch.
C5 CENTRAL CLAIM: on the 1831 notes where GAME=GT−1 semitone, RMVPE (independent F0) sides with GAME (lower) 95.0% vs GT 5.0% ⇒ the score-based GT is ~1 semitone SHARP on ~22% of notes (label issue, not model flat-bias). COnP therefore UNDERSTATES the models' true sung-pitch accuracy.
C6 GAME post-proc ablation: min-dur absorb 0.798→0.802 (uniform small win); same-pitch merge → 0.737 (poison). Reproduces Kiritan.
C7 phone-boundary (MMS, vs mono_label, 17752 phones): MMS_FA boundary MAE 0.078s/median 0.048s/BER50 48.0%; MMS_JA 0.113s/0.055s/53.3%. MMS_FA edges JA. SOFA not run (trained on Itako = contaminated).`

phase('Reproduce')

const VSCHEMA = { type: 'object', required: ['rows', 'all_match', 'notes'], properties: {
  rows: { type: 'array', items: { type: 'object', required: ['label', 'recomputed', 'claimed', 'match'],
    properties: { label: { type: 'string' }, recomputed: { type: 'string' }, claimed: { type: 'string' }, match: { type: 'boolean' } } } },
  all_match: { type: 'boolean' }, notes: { type: 'string' } } }

const reNote = agent(
  `Independently verify C1, C2, C6. Run the official evaluator (cd ${EVAL} && ${PY} evaluate.py <GT> <PRED> 0.05). PREDS: GAME=${BM}/game_raw_ja.json, CE+CTC=${BM}/ctcce_pred.json. GTs: ${BM}/itako_gt.json (raw), ${BM}/gt_defectfix.json. For the ablation, run ${PY} ${BM}/game_postproc_ablation.py --pred ${BM}/game_raw_ja.json --gt ${BM}/itako_gt.json --tag /tmp/chk_game (read its printed table). Report each metric recomputed vs claimed; match=true iff within 0.003.\n\n${CLAIMS}`,
  { label: 'reproduce:note+ablation', phase: 'Reproduce', schema: VSCHEMA })

const rePhone = agent(
  `Independently verify C7. Re-run: cd ${BM} && ${PY} phone_boundary_itako.py eval --pred phone_boundary/mms_fa_htk --target phone_boundary/target_htk and same for mms_ja_htk. Report boundary MAE / median / BER50 recomputed vs claimed (match within 0.005s / 1.5pp). Note phone count + mismatches.\n\n${CLAIMS}`,
  { label: 'reproduce:phone-boundary', phase: 'Reproduce', schema: VSCHEMA })

const reAdjudicate = agent(
  `Independently RE-IMPLEMENT the C5 adjudication from scratch — do NOT reuse any existing script; write your own. For every onset-matched note (GAME pred onset within 50ms of a GT onset in ${BM}/game_raw_ja.json vs ${BM}/itako_gt.json) where round(GAME_pitch − GT_pitch) == −1, load that song's RMVPE F0 from ${BM}/f0/<sid>.npz (keys 'f0' Hz array, 'hop_seconds'), take voiced (f0>0) frames inside the note span, median→MIDI (69+12·log2(f0/440)), and decide whether RMVPE is closer to GAME_pitch (lower) or GT_pitch (higher). Report the % siding with GAME over all such notes and the n. Use ${PY} (numpy available). State whether your independent number corroborates the claimed 95.0%.\n\n${CLAIMS}`,
  { label: 'reproduce:rmvpe-adjudication', phase: 'Reproduce', schema: VSCHEMA })

const [vNote, vPhone, vAdj] = await Promise.all([reNote, rePhone, reAdjudicate])

phase('Challenge')

const CSCHEMA = { type: 'object', required: ['claim', 'verdict', 'reasoning', 'evidence'], properties: {
  claim: { type: 'string' }, verdict: { type: 'string', enum: ['holds', 'weakened', 'refuted'] },
  reasoning: { type: 'string' }, evidence: { type: 'string' } } }

const skBias = agent(
  `Skeptic lens = SHARED BIAS. The central claim C5 says RMVPE backs GAME (the lower pitch) 95% on GAME=GT−1 notes ⇒ GT is sharp. Attack it: could GAME and RMVPE share a systematic ~1-semitone FLAT bias that manufactures this 95% without the GT being wrong? Consider: are GAME and RMVPE truly independent (GAME = CQT note transcription; RMVPE = a separate DeepUNet F0 model — different architectures/training)? Would a shared flat bias be exactly 1 semitone? Check a control: on notes where GAME==GT (|err|<50c), does RMVPE also sit ~0 vs GT (no global flat bias)? You may run ${PY} over ${BM}/f0/*.npz, ${BM}/game_raw_ja.json, ${BM}/itako_gt.json. Give a verdict on whether C5 survives.\n\n${CLAIMS}`,
  { label: 'challenge:shared-bias', phase: 'Challenge', schema: CSCHEMA })

const skStructural = agent(
  `Skeptic lens = OUR-PIPELINE-ARTIFACT. C4 calls itako03 & itako47 "structural GT defects". Attack it: could their low COn instead be an artifact of OUR GT build (build_itako_gt.py breath/sil removal over-deleting real notes, or itako50-style octave, or a time origin issue) rather than a real DB defect? Inspect itako03/itako47: compare GT note count + pitch range to neighbours, check for large gaps, check whether the raw (unfiltered) GT ${BM}/itako_gt_raw.json differs a lot, and whether a per-section (not global) shift would recover them. Run ${PY} as needed. Verdict on whether "structural GT defect" is the right label or an overclaim.\n\n${CLAIMS}`,
  { label: 'challenge:structural-artifact', phase: 'Challenge', schema: CSCHEMA })

const skCross = agent(
  `Skeptic lens = CROSS-DATASET OVERCLAIM. We want to say "Itako GT is cleaner per-song than Kiritan (no transpositions) but note-level score-vs-sung-pitch is the dominant COnP limiter." Read ${KIRITAN} for Kiritan's numbers (16/50 transposed, 5/50 time-shift, COnP 0.531→0.644 after correction). Attack: is the comparison fair given the SAME audit code/thresholds were used? Is concluding "Itako cleaner" justified when itako03/47 are broken and itako50 is octave-off (3-4 bad songs)? Is the real difference just "different defect TYPE" not "cleaner"? Propose the most defensible phrasing. Verdict.\n\n${CLAIMS}`,
  { label: 'challenge:cross-dataset', phase: 'Challenge', schema: CSCHEMA })

const [cBias, cStruct, cCross] = await Promise.all([skBias, skStructural, skCross])

phase('Synthesize')

const synthesis = await agent(
  `Write the final "Cross-dataset reads (Itako vs Kiritan)" section for ${BM}/RESULTS.md, in the terse evidence-first voice of ${KIRITAN} (read it). Ground every number ONLY in the verified results; incorporate the skeptics' surviving caveats and DROP or soften any claim they refuted/weakened.

REPRODUCED: notes=${JSON.stringify(vNote)} ; phone=${JSON.stringify(vPhone)} ; adjudication=${JSON.stringify(vAdj)}
SKEPTICS: shared-bias=${JSON.stringify(cBias)} ; structural=${JSON.stringify(cStruct)} ; cross-dataset=${JSON.stringify(cCross)}

Cover concisely, with tables where apt:
1. Headline note table (raw → defect-fixed) GAME vs CE+CTC; the COn tie; that defect-fixing barely moves COnP.
2. The GT-audit result, framed by what the skeptics let stand: Itako has no per-song transpositions; the dominant COnP limiter is note-level score-vs-sung-pitch (RMVPE sides with the models ~95% on GAME=GT−1 notes ⇒ GT sharp ⇒ COnP understates true accuracy). Compare/contrast with Kiritan's correctable whole-song defects. Use the skeptics' recommended phrasing on "cleaner vs different defect type."
3. itako01 (recoverable ~50ms lag), itako03/47 (per the structural skeptic's verdict), itako50 (octave).
4. GAME ablation reproduced; phone-boundary MMS_FA edges MMS_JA (numbers), SOFA contaminated/not run.
Output GitHub-flavored markdown only, no preamble.`,
  { label: 'synthesize:cross-dataset', phase: 'Synthesize' })

return { vNote, vPhone, vAdj, cBias, cStruct, cCross, synthesis }
