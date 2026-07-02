#!/bin/bash
# English-control batch: run UltraSinger over all 40 vocadito clips (solo vocals,
# note-level GT). Audio is already 44.1kHz mono -> fed directly, no resampling.
# Per-clip --language from metadata (clip_lang.json); AUTO -> whisperx autodetect.
# Mirrors kiritan/ultrasinger/run_batch.sh (same whisper large-v3, +48 build).
set -u
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=0

US=/home/kojiek/side_projects/music-ai/karaoke-jp/third_party/UltraSinger
AUDIO="$1"                   # vocadito Audio dir (scratchpad)
HERE=/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks/vocadito
OUT="$HERE/out"
LOG="$HERE/batch.log"
PY="$US/.venv/bin/python"
LANGMAP="$HERE/clip_lang.json"

mkdir -p "$OUT"
echo "=== VOCADITO BATCH START $(date -u +%FT%TZ) ===" | tee -a "$LOG"
for tid in $(python3 -c "import json;print(' '.join(sorted(json.load(open('$LANGMAP')),key=int)))"); do
  wav="$AUDIO/vocadito_${tid}.wav"
  outdir="$OUT/$tid"
  code=$(python3 -c "import json;print(json.load(open('$LANGMAP'))['$tid']['code'])")
  if find "$outdir" -name "*.txt" 2>/dev/null | grep -q .; then
    echo "[$tid] SKIP (txt exists)" | tee -a "$LOG"; continue
  fi
  if [ ! -f "$wav" ]; then echo "[$tid] MISSING $wav" | tee -a "$LOG"; continue; fi
  langflag=""
  [ "$code" != "AUTO" ] && langflag="--language $code"
  t0=$(date +%s)
  echo "[$tid] START lang=$code $(date -u +%FT%TZ)" | tee -a "$LOG"
  ( cd "$US/src" && "$PY" UltraSinger.py -i "$wav" -o "$outdir" \
      $langflag --whisper large-v3 ) >> "$LOG" 2>&1
  rc=$?
  t1=$(date +%s)
  if find "$outdir" -name "*.txt" 2>/dev/null | grep -q .; then
    echo "[$tid] DONE rc=$rc $((t1-t0))s" | tee -a "$LOG"
  else
    echo "[$tid] FAIL rc=$rc $((t1-t0))s (no txt)" | tee -a "$LOG"
  fi
done
echo "=== VOCADITO BATCH END $(date -u +%FT%TZ) ===" | tee -a "$LOG"
