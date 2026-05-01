#!/usr/bin/env bash
# karaoke-jp — one-shot install
#
# 建 4 個 venv、clone 3 個 third-party repo、下載 3 個 checkpoint。
# Idempotent：重跑會跳過已完成的步驟。
#
# 使用：
#   bash scripts/setup.sh
#
# 需要：python 3.10–3.12, git, wget, unzip, ffmpeg, curl
# GPU：NVIDIA CUDA 12+ 或 13（torch 2.11 + cu130 wheel）

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${HOME}/venvs"
TP="${REPO_ROOT}/third_party"

TORCH_INDEX_CU130="https://download.pytorch.org/whl/cu130"
PYPI_EXTRA="https://pypi.org/simple"

step()  { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m✔\033[0m %s\n' "$*"; }
skip()  { printf '\033[0;33m↪ skip\033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31m✘\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 0. 系統前置 ----------
step "檢查系統依賴"
for cmd in python3 git wget unzip ffmpeg curl; do
  command -v "$cmd" >/dev/null 2>&1 || fail "找不到指令：$cmd"
done
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PY_VER" in
  3.10|3.11|3.12) ok "python $PY_VER" ;;
  *) fail "需要 Python 3.10–3.12，目前是 $PY_VER" ;;
esac
mkdir -p "$VENV_DIR" "$TP"

# ---------- 1. 主 venv ----------
step "venv: ~/venvs/karaoke-jp（M0 下載 / M1 separate / M4 prep / CLI）"
if [ ! -x "${VENV_DIR}/karaoke-jp/bin/python" ]; then
  python3 -m venv "${VENV_DIR}/karaoke-jp"
fi
"${VENV_DIR}/karaoke-jp/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/karaoke-jp/bin/pip" install \
  torch==2.11.0 torchaudio==2.11.0 \
  --index-url "$TORCH_INDEX_CU130"
( cd "$REPO_ROOT" && "${VENV_DIR}/karaoke-jp/bin/pip" install -e '.[separation,render,batch]' )
"${VENV_DIR}/karaoke-jp/bin/pip" install 'librosa>=0.10'
ok "主 venv 完成"

# ---------- 2. melody venv ----------
step "venv: ~/venvs/karaoke-jp-melody（M2 SOME / CTC+CE 推理）"
if [ ! -x "${VENV_DIR}/karaoke-jp-melody/bin/python" ]; then
  python3 -m venv "${VENV_DIR}/karaoke-jp-melody"
fi
"${VENV_DIR}/karaoke-jp-melody/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/karaoke-jp-melody/bin/pip" install \
  torch==2.11.0 torchaudio==2.11.0 \
  'numpy<2' 'librosa<0.10' einops==0.6.1 \
  praat-parselmouth==0.4.3 'lightning>=2.0.0' \
  mido click PyYAML scipy h5py matplotlib torchmetrics tqdm gdown \
  --index-url "$TORCH_INDEX_CU130" \
  --extra-index-url "$PYPI_EXTRA"
ok "melody venv 完成"

# ---------- 3. lyrics venv ----------
step "venv: ~/venvs/karaoke-jp-lyrics（M3 ASR + tokenize + align）"
if [ ! -x "${VENV_DIR}/karaoke-jp-lyrics/bin/python" ]; then
  python3 -m venv "${VENV_DIR}/karaoke-jp-lyrics"
fi
"${VENV_DIR}/karaoke-jp-lyrics/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/karaoke-jp-lyrics/bin/pip" install \
  fugashi unidic-lite pyopenjtalk faster-whisper \
  nvidia-cublas-cu12 nvidia-cudnn-cu12 \
  numpy soundfile click
ok "lyrics venv 完成"

# ---------- 4. render venv ----------
step "venv: ~/venvs/karaoke-jp-render（M4 MID2BAR-Player）"
if [ ! -x "${VENV_DIR}/karaoke-jp-render/bin/python" ]; then
  python3 -m venv "${VENV_DIR}/karaoke-jp-render"
fi
"${VENV_DIR}/karaoke-jp-render/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/karaoke-jp-render/bin/pip" install \
  'pygame>2.6' Pillow numpy pandas py_midicsv mido \
  opencv-python chardet tqdm
ok "render venv 完成"

# ---------- 5. clone third-party ----------
step "Clone third_party/ repos"
clone_if_missing() {
  local url="$1" dest="$2"
  if [ -d "$dest/.git" ]; then
    skip "$(basename "$dest") 已 clone"
  else
    git clone --depth 1 "$url" "$dest"
  fi
}
clone_if_missing "https://github.com/openvpi/SOME.git"               "$TP/SOME"
clone_if_missing "https://github.com/york135/CTC_CE_for_AST.git"     "$TP/CTC_CE_for_AST"
clone_if_missing "https://github.com/keisuke-okb/MID2BAR-Player.git" "$TP/MID2BAR-Player"

# ---------- 6. SOME baseline checkpoint ----------
step "下載 SOME baseline checkpoint（~435 MB）"
SOME_PT="$TP/SOME/pretrained"
mkdir -p "$SOME_PT"
if [ -d "$SOME_PT/256_5spk" ]; then
  skip "SOME baseline checkpoint 已存在"
else
  ( cd "$SOME_PT" \
    && wget -c "https://github.com/openvpi/SOME/releases/download/v1.0.0-baseline/0119_continuous128_5spk.zip" \
    && unzip -o 0119_continuous128_5spk.zip \
    && rm 0119_continuous128_5spk.zip )
fi

# ---------- 7. RMVPE checkpoint ----------
step "下載 RMVPE checkpoint（~352 MB）"
if [ -f "$SOME_PT/rmvpe/model.pt" ]; then
  skip "RMVPE checkpoint 已存在"
else
  ( cd "$SOME_PT" \
    && wget -c "https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip" \
    && rm -rf rmvpe rmvpe_unpack \
    && unzip -o rmvpe.zip -d rmvpe_unpack \
    && mkdir -p rmvpe \
    && find rmvpe_unpack -name model.pt -exec mv {} rmvpe/model.pt \; \
    && rm -rf rmvpe_unpack rmvpe.zip )
fi

# ---------- 8. CTC+CE checkpoint（gdown）----------
step "下載 CTC+CE checkpoint via gdown（~3.9 MB）"
CECTC_PT="$TP/CTC_CE_for_AST/pretrained"
if compgen -G "$CECTC_PT/ctc_ce*" > /dev/null; then
  skip "CTC+CE checkpoint 已存在"
else
  mkdir -p "$CECTC_PT"
  ( cd "$CECTC_PT" \
    && "${VENV_DIR}/karaoke-jp-melody/bin/gdown" --folder \
       "https://drive.google.com/drive/folders/1lxq-IF83cEXE8XsTFywNJhwtDSRXWqRx" )
fi

# ---------- 9. smoke test ----------
step "Smoke test"
"${VENV_DIR}/karaoke-jp/bin/karaoke-jp" --help >/dev/null && ok "karaoke-jp CLI 可用"
"${VENV_DIR}/karaoke-jp-melody/bin/python" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
"${VENV_DIR}/karaoke-jp-lyrics/bin/python" -c 'import faster_whisper, fugashi; print("ASR + 形態素 OK")'
"${VENV_DIR}/karaoke-jp-render/bin/python" -c 'import pygame, cv2; print("Pygame", pygame.__version__, "OpenCV", cv2.__version__)'

cat <<EOF

$(printf '\033[1;32m安裝完成！\033[0m')

跑第一首歌：
  ${VENV_DIR}/karaoke-jp/bin/python scripts/download_song.py \\
    'https://youtu.be/<id>' -o songs/<song-id>/
  # 把歌詞貼進 songs/<song-id>/lyrics.txt
  ${VENV_DIR}/karaoke-jp/bin/snakemake --rerun-triggers mtime -j 1 \\
    outputs/<song-id>/karaoke.mp4

四個 venv 在 ${VENV_DIR}/，third_party repo + checkpoint 在 ${TP}/。
EOF
