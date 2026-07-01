#!/bin/bash
set -u
BM=/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks/itako
ROSVOT_DIR=/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks/ROSVOT
EVAL=/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks/singing_transcription_ICASSP2021/evaluate
PY=/home/kojiek/venvs/karaoke-jp/bin/python
RPY=/home/kojiek/venvs/karaoke-jp-melody/bin/python
export CUDA_VISIBLE_DEVICES=0
for _ in $(seq 1 240); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1)
  [ "${free:-0}" -ge 6000 ] && break || sleep 60
done
echo "[rosvot] GPU free ${free}MB, running"
cd "$ROSVOT_DIR"
PYTHONPATH="$ROSVOT_DIR" $RPY inference/rosvot.py -o "$BM/rosvot_out" \
  --metadata "$BM/rosvot_manifest.json" --apply_rwbd > "$BM/rosvot_retry.log" 2>&1
$PY "$BM/../mir-st500/midi_to_json.py" "$BM/rosvot_out/midi" "$BM/rosvot_pred.json"
if [ -s "$BM/rosvot_pred.json" ]; then
  cd "$EVAL"
  $PY evaluate.py "$BM/itako_gt.json" "$BM/rosvot_pred.json" 0.05 > "$BM/eval_rosvot_50ms.txt" 2>/dev/null
  $PY evaluate.py "$BM/itako_gt.json" "$BM/rosvot_pred.json" 0.10 > "$BM/eval_rosvot_100ms.txt" 2>/dev/null
fi
echo "[rosvot] ROSVOT_RETRY_DONE midi=$(ls $BM/rosvot_out/midi/*.mid 2>/dev/null|wc -l)"
