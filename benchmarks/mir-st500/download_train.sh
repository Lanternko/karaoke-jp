#!/bin/bash
# MIR-ST500 train split (1-400) for CE+CTC retraining
LINKS=/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks/singing_transcription_ICASSP2021/MIR-ST500_20210206/MIR-ST500_link.json
OUT=/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks/mir-st500/train_audio
mkdir -p "$OUT"
ok=0; fail=0
for i in $(seq 1 400); do
  [ -f "$OUT/$i.wav" ] && { ok=$((ok+1)); continue; }
  url=$(python3 -c "import json;print(json.load(open('$LINKS'))['$i'])")
  if /home/kojiek/venvs/dac/bin/yt-dlp -x --audio-format wav --audio-quality 0 \
       -o "$OUT/$i.%(ext)s" --no-playlist --quiet --no-warnings "$url" 2>/dev/null; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "[$i] FAILED ($ok ok, $fail fail)"
  fi
  sleep 1.5
done
echo "TRAIN DOWNLOAD DONE: $ok ok, $fail failed"
