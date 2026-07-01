#!/bin/bash
# Itako: complete every GPU-gated stage, then the corrected evals. Waits for a
# stable free-GPU window first so it NEVER competes with another job's training.
# Resumable: each stage skips work whose output already exists.
#
# Stages:
#   1 RMVPE F0 (50 songs)                         [GPU, melody venv]
#   2 transposition audit -> gt_transcorr.json    [CPU]
#   3 time-shift audit    -> gt_timefix.json      [CPU]
#   4 re-eval GAME + CE+CTC on raw/transcorr/timefix   [CPU]
#   5 ROSVOT (PYTHONPATH-fixed) @ 50 & 100 ms     [GPU, melody venv]
#   6 phone-boundary MMS_FA + MMS_JA -> eval      [GPU, align venv]
set -u
BASE=/home/kojiek/side_projects/music-ai/karaoke-jp
BM=$BASE/benchmarks/itako
SRC=/home/kojiek/side_projects/itako/itako_singing
WAV=$SRC/wav
GT=$BM/itako_gt.json
EVAL=$BASE/benchmarks/singing_transcription_ICASSP2021/evaluate
PY=/home/kojiek/venvs/karaoke-jp/bin/python
MELODY_PY=/home/kojiek/venvs/karaoke-jp-melody/bin/python
ALIGN_PY=/home/kojiek/venvs/karaoke-jp-align/bin/python
GAME_PY=/home/kojiek/venvs/karaoke-jp-game/bin/python
RMVPE=$BASE/third_party/SOME/pretrained/rmvpe/model.pt
NEED_MB=5000
export CUDA_VISIBLE_DEVICES=0
cd "$BM"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# ---- wait for a stable free-GPU window (2 consecutive polls, max ~6h) -------
log "waiting for >=${NEED_MB}MB free GPU (will not compete with running jobs)..."
ok=0
for _ in $(seq 1 360); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "${free:-0}" -ge "$NEED_MB" ]; then ok=$((ok+1)); else ok=0; fi
  if [ "$ok" -ge 2 ]; then log "GPU free (${free}MB) — proceeding"; break; fi
  sleep 60
done
if [ "$ok" -lt 2 ]; then log "GPU never freed within budget — abort"; exit 3; fi

# ---- 1) RMVPE F0 -----------------------------------------------------------
mkdir -p f0
for w in "$WAV"/itako*.wav; do
  id=$(basename "$w" .wav)
  [ -f "f0/$id.npz" ] && continue
  $MELODY_PY "$BASE/scripts/extract_rmvpe_f0.py" --wav "$w" --model "$RMVPE" --out "f0/$id.npz" \
    >/dev/null 2>"f0/$id.err" && { log "f0 $id"; rm -f "f0/$id.err"; } || log "f0 $id FAILED"
done

# ---- 2) transposition audit ------------------------------------------------
[ -f gt_transcorr.json ] || $PY audit_gt_rmvpe.py --gt "$GT" --f0-dir f0 --out gt_transcorr.json | tee audit_transcorr.txt
# ---- 3) time-shift audit ---------------------------------------------------
[ -f gt_timefix.json ] || $PY audit_timeshift.py --gt gt_transcorr.json --f0-dir f0 --out gt_timefix.json | tee audit_timefix.txt

# ---- 4) re-eval on the three GTs ------------------------------------------
cd "$EVAL"
for sys in game:game_raw_ja.json cectc:ctcce_pred.json; do
  name=${sys%%:*}; pred=$BM/${sys##*:}
  for gtv in raw:$GT transcorr:$BM/gt_transcorr.json timefix:$BM/gt_timefix.json; do
    tag=${gtv%%:*}; gtf=${gtv##*:}
    [ -f "$gtf" ] || continue
    log "eval $name vs $tag"
    $PY evaluate.py "$gtf" "$pred" 0.05 > "$BM/eval_${name}_${tag}.txt" 2>/dev/null
  done
done
cd "$BM"

# ---- 5) ROSVOT (fixed) @ 50 & 100 ms --------------------------------------
ROSVOT_DIR=$BASE/benchmarks/ROSVOT
ROSVOT_PY=/home/kojiek/venvs/karaoke-jp-melody/bin/python
if [ ! -f rosvot_pred.json ]; then
  $PY - "$WAV" rosvot_manifest.json <<'PYEOF'
import json,sys; from pathlib import Path
w,o=Path(sys.argv[1]),sys.argv[2]
Path(o).write_text(json.dumps([{"item_name":p.stem,"wav_fn":str(p)} for p in sorted(w.glob("itako*.wav"))]))
PYEOF
  ( cd "$ROSVOT_DIR" && PYTHONPATH="$ROSVOT_DIR" $ROSVOT_PY inference/rosvot.py \
      -o "$BM/rosvot_out" --metadata "$BM/rosvot_manifest.json" --save_midi ) 2>"$BM/rosvot.err" \
    && log "rosvot inference done" || log "rosvot inference FAILED (see rosvot.err)"
  $PY "$BASE/benchmarks/mir-st500/midi_to_json.py" "$BM/rosvot_out/midi" rosvot_pred.json 2>/dev/null || true
fi
if [ -s rosvot_pred.json ]; then
  cd "$EVAL"
  $PY evaluate.py "$GT" "$BM/rosvot_pred.json" 0.05 > "$BM/eval_rosvot_50ms.txt" 2>/dev/null
  $PY evaluate.py "$GT" "$BM/rosvot_pred.json" 0.10 > "$BM/eval_rosvot_100ms.txt" 2>/dev/null
  cd "$BM"; log "rosvot eval done"
fi

# ---- 6) phone-boundary MMS (MMS_FA clean zero-shot; MMS_JA karaoke ckpt) ----
# SOFA's JPN model trained on Itako => contaminated; only MMS is reported clean.
for m in mms_fa mms_ja; do
  od="phone_boundary/${m}_htk"
  if [ ! -d "$od" ] || [ -z "$(ls -A "$od" 2>/dev/null)" ]; then
    log "phone-boundary run-mms $m"
    $ALIGN_PY phone_boundary_itako.py run-mms --method "$m" --out "$od" --device cuda \
      2>"phone_boundary/${m}.err" && log "$m aligned" || log "$m FAILED (see phone_boundary/${m}.err)"
  fi
  if [ -d "$od" ] && [ -n "$(ls -A "$od" 2>/dev/null)" ]; then
    $PY phone_boundary_itako.py eval --pred "$od" --target phone_boundary/target_htk \
      --json-out "phone_boundary/eval_${m}.json" > "phone_boundary/eval_${m}.txt" 2>/dev/null && log "$m eval done"
  fi
done

log "GPU_STAGES_DONE"
