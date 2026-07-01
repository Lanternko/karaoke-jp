#!/bin/bash
# RMVPE F0 for the Itako vocals (GT-audit input). Needs a mostly-free GPU.
set -u
BASE=/home/kojiek/side_projects/music-ai/karaoke-jp
WAV=/home/kojiek/side_projects/itako/itako_singing/wav
OUT=$BASE/benchmarks/itako/f0
MELODY_PY=/home/kojiek/venvs/karaoke-jp-melody/bin/python
RMVPE=$BASE/third_party/SOME/pretrained/rmvpe/model.pt
mkdir -p "$OUT"
# guard: RMVPE needs ~2-3GB; bail early if the GPU is nearly full (someone training)
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${free_mb:-0}" -lt 4000 ]; then
  echo "[f0] GPU only ${free_mb}MB free — aborting (RMVPE needs ~3GB). Re-run when free."
  exit 3
fi
for w in "$WAV"/itako*.wav; do
  id=$(basename "$w" .wav); f0="$OUT/$id.npz"
  [ -f "$f0" ] && { echo "[f0] $id exists"; continue; }
  if $MELODY_PY "$BASE/scripts/extract_rmvpe_f0.py" --wav "$w" --model "$RMVPE" --out "$f0" 2>"$OUT/$id.err"; then
    echo "[f0] $id done"; rm -f "$OUT/$id.err"
  else
    echo "[f0] $id FAILED (see $OUT/$id.err)"; fi
done
echo F0_ALL_DONE
