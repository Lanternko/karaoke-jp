#!/bin/bash
# Variant: GAME raw + extend-sustains (RMVPE F0). Onset/pitch unchanged;
# only note tails (offset) are pushed forward while F0 holds the pitch.
# Quantifies how much of the COnPOff gap is pure sustain-truncation.
# Resumable.
set -u
BASE=/home/kojiek/side_projects/music-ai/karaoke-jp
BM=$BASE/benchmarks/mir-st500
MELODY_PY=/home/kojiek/venvs/karaoke-jp-melody/bin/python
PY=/home/kojiek/venvs/karaoke-jp/bin/python
RMVPE=$BASE/third_party/SOME/pretrained/rmvpe/model.pt

mkdir -p "$BM/f0" "$BM/ext_mid"

for vwav in "$BM"/vocals_all/*.wav; do
  id=$(basename "$vwav" .wav)
  game_mid="$BM/vocals_all/$id.mid"
  [ -f "$game_mid" ] || { echo "[skip] $id no GAME mid"; continue; }

  # 1) RMVPE F0 (GPU)
  f0="$BM/f0/$id.npz"
  if [ ! -f "$f0" ]; then
    $MELODY_PY "$BASE/scripts/extract_rmvpe_f0.py" \
      --wav "$vwav" --model "$RMVPE" --out "$f0" >/dev/null 2>&1 \
      || { echo "[f0] $id FAILED"; continue; }
  fi

  # 2) extend-sustains only (no aligned needed)
  ext="$BM/ext_mid/$id.mid"
  if [ ! -f "$ext" ]; then
    $PY "$BASE/scripts/score_note_postfix.py" \
      --midi "$game_mid" --f0 "$f0" --out "$ext" \
      --extend-sustains --keep-repeats >/dev/null 2>&1 \
      || { echo "[postfix] $id FAILED"; continue; }
  fi
  echo "[ok] $id"
done

# 3) predictions JSON + official eval
$PY "$BM/midi_to_json.py" "$BM/ext_mid" "$BM/game_ext_zh.json"
cd "$BASE/benchmarks/singing_transcription_ICASSP2021/evaluate"
$PY evaluate.py "$BASE/benchmarks/singing_transcription_ICASSP2021/MIR-ST500_20210206/MIR-ST500_corrected.json" \
  "$BM/game_ext_zh.json" 0.05 | tee "$BM/eval_game_ext_zh.txt"
echo VARIANT_DONE
