#!/bin/bash
# Itako note-transcription benchmark: a cappella, zero-shot, mirrors the Kiritan
# protocol (NOT mir-st500: Itako is clean solo vocal, so NO source separation).
#
# GT  : benchmarks/itako/itako_gt.json  (built by build_itako_gt.py from the
#       public mmorise/itako_singing labels; breath/out-of-range notes removed
#       via mono_label br/pau/sil overlap).
# Eval: official MIR-ST500 evaluate.py (mir_eval; onset 50 ms, pitch 50 cents,
#       offset max(50ms, 0.2*dur)).
#
# AUDIO IS GATED. The 50 wav files are NOT public; download them (research /
# non-commercial, 著作権法30-4) from https://zunko.jp/itadev/login.php and place
# them as itako01.wav .. itako50.wav under:
#     /home/kojiek/side_projects/itako/itako_singing/wav/
# Then run this script. Every stage skips work whose output already exists.
set -u

BASE=/home/kojiek/side_projects/music-ai/karaoke-jp
BM=$BASE/benchmarks/itako
SRC=/home/kojiek/side_projects/itako/itako_singing
WAV=$SRC/wav
GT=$BM/itako_gt.json

GAME_DIR=$BASE/third_party/GAME
GAME_PY=/home/kojiek/venvs/karaoke-jp-game/bin/python
PY=/home/kojiek/venvs/karaoke-jp/bin/python
EVAL=$BASE/benchmarks/singing_transcription_ICASSP2021/evaluate

# ---- audio guard -----------------------------------------------------------
n_wav=$(ls "$WAV"/itako*.wav 2>/dev/null | wc -l)
if [ "$n_wav" -eq 0 ]; then
  cat <<EOF
[itako] No wav found in $WAV
        The Itako audio is gated. Download the singing DB (non-commercial
        research) from https://zunko.jp/itadev/login.php and place the files as
        itako01.wav .. itako50.wav in that directory, then re-run.
        Labels + GT are already prepared; this is the only missing input.
EOF
  exit 0
fi
echo "[itako] $n_wav wav files found"

# ---- 1) GAME extract, Japanese conditioning, batch over the wav dir ---------
# Itako is a cappella -> feed the raw vocal directly (same as Kiritan).
if [ ! -f "$BM/game_raw_ja.json" ]; then
  echo "[game] extract -l ja over $WAV"
  cd "$GAME_DIR"
  $GAME_PY infer.py extract "$WAV" -m "$GAME_DIR/pretrained/GAME-1.0-large/model.pt" \
    -l ja --glob 'itako*.wav' --output-formats mid
  $PY "$BASE/benchmarks/mir-st500/midi_to_json.py" "$WAV" "$BM/game_raw_ja.json"
fi
cd "$EVAL"
$PY evaluate.py "$GT" "$BM/game_raw_ja.json" 0.05 | tee "$BM/eval_game_raw_ja.txt"

# ---- 2) CE+CTC (Wang & Jang, TASLP 2022), a cappella so no accompaniment ----
CECTC_CKPT=$BASE/benchmarks/CTC_CE_for_AST/pretrained/ctc_ce#3_98/ctc_ce#3_98
if [ ! -f "$BM/ctcce_pred.json" ]; then
  echo "[ctc-ce] inference over $WAV (a cappella: acc = silence)"
  $GAME_PY "$BASE/benchmarks/ctc_ce_infer.py" \
    --repo "$BASE/benchmarks/CTC_CE_for_AST" \
    --ckpt "$CECTC_CKPT" \
    --vocals "$WAV" \
    --out "$BM/ctcce_pred.json"
fi
cd "$EVAL"
$PY evaluate.py "$GT" "$BM/ctcce_pred.json" 0.05 | tee "$BM/eval_ctcce.txt"

# ---- 3) ROSVOT (ACL 2024, M4Singer ckpt), RWBD active (no word_durs) --------
# Documented as a large negative result on unannotated audio (Kiritan/mir-st500);
# included for parity. Batched inference -> MIDI -> JSON -> eval.
ROSVOT_DIR=$BASE/benchmarks/ROSVOT
ROSVOT_PY=/home/kojiek/venvs/karaoke-jp-melody/bin/python
if [ ! -f "$BM/rosvot_pred.json" ]; then
  echo "[rosvot] building manifest + batched inference"
  $PY - "$WAV" "$BM/rosvot_manifest.json" <<'PYEOF'
import json, sys
from pathlib import Path
wav, out = Path(sys.argv[1]), sys.argv[2]
items = [{"item_name": p.stem, "wav_fn": str(p)} for p in sorted(wav.glob("itako*.wav"))]
Path(out).write_text(json.dumps(items))
print(f"[rosvot] manifest: {len(items)} items -> {out}")
PYEOF
  cd "$ROSVOT_DIR"
  # ROSVOT's inference script imports its top-level `utils` package -> repo root
  # must be on PYTHONPATH (else ModuleNotFoundError: No module named 'utils').
  PYTHONPATH="$ROSVOT_DIR" $ROSVOT_PY inference/rosvot.py -o "$BM/rosvot_out" \
    --metadata "$BM/rosvot_manifest.json" --save_midi || \
    echo "[rosvot] check inference/rosvot.py --help for the exact batched flags in this checkout"
  $PY "$BASE/benchmarks/mir-st500/midi_to_json.py" "$BM/rosvot_out/midi" "$BM/rosvot_pred.json" || true
fi
if [ -f "$BM/rosvot_pred.json" ]; then
  cd "$EVAL"
  $PY evaluate.py "$GT" "$BM/rosvot_pred.json" 0.05 | tee "$BM/eval_rosvot.txt"
fi

echo PIPELINE_DONE
