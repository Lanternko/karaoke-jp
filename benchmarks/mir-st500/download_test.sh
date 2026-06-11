#!/bin/bash
# Download MIR-ST500 test set (songs 401-500) audio as wav
LINKS=/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks/singing_transcription_ICASSP2021/MIR-ST500_20210206/MIR-ST500_link.json
OUT=/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks/mir-st500/audio
ok=0; fail=0
for i in $(seq 401 500); do
  [ -f "$OUT/$i.wav" ] && { ok=$((ok+1)); continue; }
  url=$(python3 -c "import json;print(json.load(open('$LINKS'))['$i'])")
  if /home/kojiek/venvs/dac/bin/yt-dlp -x --audio-format wav --audio-quality 0 \
       -o "$OUT/$i.%(ext)s" --no-playlist --quiet --no-warnings "$url" 2>/dev/null; then
    ok=$((ok+1)); echo "[$i] OK ($ok ok, $fail fail)"
  else
    fail=$((fail+1)); echo "[$i] FAILED ($ok ok, $fail fail)"
  fi
  sleep 2
done
echo "DONE: $ok downloaded, $fail failed"
