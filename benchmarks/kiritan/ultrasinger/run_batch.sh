#!/bin/bash
# Phase 3: run UltraSinger over all 50 Kiritan songs (a cappella, ja, large-v3).
# Inputs = 44.1kHz/16-bit/mono conversions in scratchpad (Phase 1).
# song 01 already done in the smoke test; skipped if output .txt exists.
set -u
export UV_LINK_MODE=copy
export CUDA_VISIBLE_DEVICES=0

US=/home/kojiek/side_projects/music-ai/karaoke-jp/third_party/UltraSinger
SCRATCH="$1"                 # scratchpad path (wav441 lives here)
OUT=/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks/kiritan/ultrasinger/out
LOG=/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks/kiritan/ultrasinger/batch.log
PY="$US/.venv/bin/python"

mkdir -p "$OUT"
echo "=== BATCH START $(date -u +%FT%TZ) ===" | tee -a "$LOG"
for i in $(seq -w 1 50); do
  wav="$SCRATCH/wav441/$i.wav"
  outdir="$OUT/$i"
  # already have a parseable txt?
  if find "$outdir" -name "*.txt" 2>/dev/null | grep -q .; then
    echo "[$i] SKIP (txt exists)" | tee -a "$LOG"; continue
  fi
  if [ ! -f "$wav" ]; then echo "[$i] MISSING input $wav" | tee -a "$LOG"; continue; fi
  t0=$(date +%s)
  echo "[$i] START $(date -u +%FT%TZ)" | tee -a "$LOG"
  ( cd "$US/src" && "$PY" UltraSinger.py -i "$wav" -o "$outdir" \
      --language ja --whisper large-v3 ) >> "$LOG" 2>&1
  rc=$?
  t1=$(date +%s)
  if find "$outdir" -name "*.txt" 2>/dev/null | grep -q .; then
    echo "[$i] DONE rc=$rc $((t1-t0))s" | tee -a "$LOG"
  else
    echo "[$i] FAIL rc=$rc $((t1-t0))s (no txt)" | tee -a "$LOG"
  fi
done
echo "=== BATCH END $(date -u +%FT%TZ) ===" | tee -a "$LOG"
