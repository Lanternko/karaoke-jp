#!/bin/bash
# MIR-ST500 test-set benchmark: separation -> GAME (-l zh) -> JSON -> official eval
# Resumable: every stage skips work whose output already exists.
set -u
BASE=/home/kojiek/side_projects/music-ai/karaoke-jp
BM=$BASE/benchmarks/mir-st500
GAME_DIR=$BASE/third_party/GAME
GAME_PY=/home/kojiek/venvs/karaoke-jp-game/bin/python
KJ=/home/kojiek/venvs/karaoke-jp/bin/karaoke-jp
PY=/home/kojiek/venvs/karaoke-jp/bin/python

mkdir -p "$BM/sep" "$BM/vocals_all"

# 1) separation (Mel-Band-RoFormer, same as production chain)
for wav in "$BM"/audio/*.wav; do
  id=$(basename "$wav" .wav)
  out="$BM/sep/$id"
  if [ ! -f "$out/vocals.wav" ]; then
    echo "[sep] $id"
    $KJ separate "$wav" -o "$out" || { echo "[sep] $id FAILED"; continue; }
  fi
  ln -sf "$out/vocals.wav" "$BM/vocals_all/$id.wav"
done

# 2) GAME extract, Mandarin language conditioning, batch over the directory
echo "[game] running extract over vocals_all"
cd "$GAME_DIR"
$GAME_PY infer.py extract "$BM/vocals_all" -m "$GAME_DIR/pretrained/GAME-1.0-large/model.pt" \
  -l zh --glob '*.wav' --output-formats mid

# 3) MIDI -> prediction JSON
$PY "$BM/midi_to_json.py" "$BM/vocals_all" "$BM/game_raw_zh.json"

# 4) official evaluation (onset tol 0.05, pitch 50 cents)
cd "$BASE/benchmarks/singing_transcription_ICASSP2021/evaluate"
$PY evaluate.py "$BASE/benchmarks/singing_transcription_ICASSP2021/MIR-ST500_20210206/MIR-ST500_corrected.json" \
  "$BM/game_raw_zh.json" 0.05 | tee "$BM/eval_game_raw_zh.txt"
echo PIPELINE_DONE
