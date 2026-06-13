#!/bin/bash
# GAME boundary/presence threshold sweep on Kiritan (timefix GT).
# Outputs land next to input wavs -> one symlink farm per config.
set -u
BM=/home/kojiek/side_projects/music-ai/karaoke-jp/benchmarks
GAME=/home/kojiek/side_projects/music-ai/karaoke-jp/third_party/GAME
GPY=/home/kojiek/venvs/karaoke-jp-game/bin/python
PY=/home/kojiek/venvs/karaoke-jp/bin/python
KIR=/home/kojiek/side_projects/kiritan/kiritan_singing/wav

for cfg in "0.2 0.3" "0.2 0.4" "0.3 0.2" "0.3 0.3" "0.3 0.4"; do
  set -- $cfg; seg=$1; est=$2
  tag="seg${seg}_est${est}"
  dir=$BM/kiritan/sweep/$tag
  mkdir -p "$dir"
  for w in "$KIR"/*.wav; do ln -sf "$w" "$dir/$(basename "$w")"; done
  if ! ls "$dir"/*.mid >/dev/null 2>&1; then
    (cd "$GAME" && $GPY infer.py extract "$dir" -m pretrained/GAME-1.0-large/model.pt \
      -l ja --glob '*.wav' --output-formats mid \
      --seg-threshold "$seg" --est-threshold "$est" >/dev/null 2>&1)
  fi
  $PY $BM/mir-st500/midi_to_json.py "$dir" "$BM/kiritan/sweep_${tag}.json" >/dev/null
  line=$(cd $BM/singing_transcription_ICASSP2021/evaluate && \
    $PY evaluate.py "$BM/kiritan/gt_timefix.json" "$BM/kiritan/sweep_${tag}.json" 0.05 2>/dev/null | \
    grep -E 'COnPOff|COnP |COn ' | awk '{printf "%s %.3f  ", $1, $4}')
  notes=$($PY -c "import json;print(sum(len(v) for v in json.load(open('$BM/kiritan/sweep_${tag}.json')).values()))")
  echo "$tag: $line notes=$notes"
done
echo "baseline seg0.2_est0.2: COnPOff 0.502 COnP 0.644 COn 0.862 notes=11506"
echo SWEEP_DONE
