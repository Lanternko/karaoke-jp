#!/usr/bin/env bash
# karaoke-jp — one-shot install
#
# Creates 6 venvs, clones 4 third-party repos, downloads the checkpoints and
# builds the lyric font. Idempotent: re-running skips finished steps.
#
# Usage:
#   bash scripts/setup.sh
#
# Needs: python 3.10–3.12, git, wget, unzip, ffmpeg, curl
# GPU: NVIDIA, CUDA 12.9+ driver (torch 2.11 cu130 / cu129 wheels)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${HOME}/venvs"
TP="${REPO_ROOT}/third_party"

TORCH_INDEX_CU130="https://download.pytorch.org/whl/cu130"
# cu129 wheels: required for GAME / the aligner on RTX 50xx (sm_120)
TORCH_INDEX_CU129="https://download.pytorch.org/whl/cu129"
FONT_DIR="${HOME}/.local/share/fonts/karaoke-jp"
PYPI_EXTRA="https://pypi.org/simple"

step()  { printf '\n\033[1;36m=== %s ===\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m✔\033[0m %s\n' "$*"; }
skip()  { printf '\033[0;33m↪ skip\033[0m %s\n' "$*"; }
fail()  { printf '\033[1;31m✘\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 0. prerequisites ----------
step "System dependencies"
for cmd in python3 git wget unzip ffmpeg curl; do
  command -v "$cmd" >/dev/null 2>&1 || fail "missing command: $cmd"
done
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PY_VER" in
  3.10|3.11|3.12) ok "python $PY_VER" ;;
  *) fail "need Python 3.10–3.12, found $PY_VER" ;;
esac
mkdir -p "$VENV_DIR" "$TP"

# ---------- 1. main venv ----------
step "venv: ~/venvs/karaoke-jp (download / separate / LRC prep / CLI)"
if [ ! -x "${VENV_DIR}/karaoke-jp/bin/python" ]; then
  python3 -m venv "${VENV_DIR}/karaoke-jp"
fi
"${VENV_DIR}/karaoke-jp/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/karaoke-jp/bin/pip" install \
  torch==2.11.0 torchaudio==2.11.0 \
  --index-url "$TORCH_INDEX_CU130"
( cd "$REPO_ROOT" && "${VENV_DIR}/karaoke-jp/bin/pip" install -e '.[separation,render,batch]' )
"${VENV_DIR}/karaoke-jp/bin/pip" install 'librosa>=0.10'
ok "main venv done"

# ---------- 2. melody venv ----------
step "venv: ~/venvs/karaoke-jp-melody (SOME / RMVPE / CTC+CE)"
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
ok "melody venv done"

# ---------- 3. lyrics venv ----------
step "venv: ~/venvs/karaoke-jp-lyrics (tokenize + furigana + ASR)"
if [ ! -x "${VENV_DIR}/karaoke-jp-lyrics/bin/python" ]; then
  python3 -m venv "${VENV_DIR}/karaoke-jp-lyrics"
fi
"${VENV_DIR}/karaoke-jp-lyrics/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/karaoke-jp-lyrics/bin/pip" install \
  fugashi unidic-lite pyopenjtalk faster-whisper \
  nvidia-cublas-cu12 nvidia-cudnn-cu12 \
  numpy soundfile click
ok "lyrics venv done"

# ---------- 4. render venv ----------
step "venv: ~/venvs/karaoke-jp-render (MID2BAR-Player)"
if [ ! -x "${VENV_DIR}/karaoke-jp-render/bin/python" ]; then
  python3 -m venv "${VENV_DIR}/karaoke-jp-render"
fi
"${VENV_DIR}/karaoke-jp-render/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/karaoke-jp-render/bin/pip" install \
  'pygame>2.6' Pillow numpy pandas py_midicsv mido \
  opencv-python chardet tqdm click freetype-py essentia
ok "render venv done"

# ---------- 4b. GAME venv ----------
step "venv: ~/venvs/karaoke-jp-game (GAME note transcription)"
if [ ! -x "${VENV_DIR}/karaoke-jp-game/bin/python" ]; then
  python3 -m venv "${VENV_DIR}/karaoke-jp-game"
fi
"${VENV_DIR}/karaoke-jp-game/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/karaoke-jp-game/bin/pip" install \
  torch==2.11.0 torchaudio==2.11.0 --index-url "$TORCH_INDEX_CU129"
"${VENV_DIR}/karaoke-jp-game/bin/pip" install \
  click colorednoise 'einops>=0.7.0' h5py librosa lightning loguru matplotlib \
  mido 'numpy<2.0.0' omegaconf 'pydantic>=2.11,<3.0' resampy 'scipy>=1.10.0' \
  sympy torchmetrics tqdm PyYAML
ok "GAME venv done"

# ---------- 4c. align venv ----------
step "venv: ~/venvs/karaoke-jp-align (MMS CTC forced alignment)"
if [ ! -x "${VENV_DIR}/karaoke-jp-align/bin/python" ]; then
  python3 -m venv "${VENV_DIR}/karaoke-jp-align"
fi
"${VENV_DIR}/karaoke-jp-align/bin/pip" install --upgrade pip --quiet
"${VENV_DIR}/karaoke-jp-align/bin/pip" install \
  torch==2.11.0 torchaudio==2.11.0 --index-url "$TORCH_INDEX_CU129"
"${VENV_DIR}/karaoke-jp-align/bin/pip" install \
  transformers 'numpy<2' soundfile click mido
ok "align venv done"

# ---------- 5. clone third-party ----------
step "Clone third_party/ repos"
clone_if_missing() {
  local url="$1" dest="$2"
  if [ -d "$dest/.git" ]; then
    skip "$(basename "$dest") already cloned"
  else
    git clone --depth 1 "$url" "$dest"
  fi
}
clone_if_missing "https://github.com/openvpi/SOME.git"               "$TP/SOME"
clone_if_missing "https://github.com/york135/CTC_CE_for_AST.git"     "$TP/CTC_CE_for_AST"
clone_if_missing "https://github.com/keisuke-okb/MID2BAR-Player.git" "$TP/MID2BAR-Player"
clone_if_missing "https://github.com/openvpi/GAME.git"               "$TP/GAME"

# ---------- 6. SOME baseline checkpoint ----------
step "SOME baseline checkpoint (~435 MB)"
SOME_PT="$TP/SOME/pretrained"
mkdir -p "$SOME_PT"
if [ -d "$SOME_PT/256_5spk" ]; then
  skip "SOME baseline checkpoint present"
else
  ( cd "$SOME_PT" \
    && wget -c "https://github.com/openvpi/SOME/releases/download/v1.0.0-baseline/0119_continuous128_5spk.zip" \
    && unzip -o 0119_continuous128_5spk.zip \
    && rm 0119_continuous128_5spk.zip )
fi

# ---------- 7. RMVPE checkpoint ----------
step "RMVPE checkpoint (~352 MB)"
if [ -f "$SOME_PT/rmvpe/model.pt" ]; then
  skip "RMVPE checkpoint present"
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
step "CTC+CE checkpoint via gdown (~3.9 MB)"
CECTC_PT="$TP/CTC_CE_for_AST/pretrained"
if compgen -G "$CECTC_PT/ctc_ce*" > /dev/null; then
  skip "CTC+CE checkpoint present"
else
  mkdir -p "$CECTC_PT"
  ( cd "$CECTC_PT" \
    && "${VENV_DIR}/karaoke-jp-melody/bin/gdown" --folder \
       "https://drive.google.com/drive/folders/1lxq-IF83cEXE8XsTFywNJhwtDSRXWqRx" )
fi

# ---------- 8b. GAME checkpoint ----------
step "GAME 1.0 large checkpoint (~400 MB)"
GAME_PT="$TP/GAME/pretrained"
if [ -f "$GAME_PT/GAME-1.0-large/model.pt" ]; then
  skip "GAME checkpoint present"
else
  mkdir -p "$GAME_PT"
  ( cd "$GAME_PT" \
    && wget -c "https://github.com/openvpi/GAME/releases/download/v1.0.0/GAME-1.0-large.zip" \
    && unzip -o GAME-1.0-large.zip \
    && rm GAME-1.0-large.zip )
fi

# ---------- 8c. flat bar skin ----------
step "Flat bar skin (MID2BAR images/bar/8_flat)"
( cd "$REPO_ROOT" && "${VENV_DIR}/karaoke-jp-render/bin/python" scripts/make_flat_bar_skin.py )
ok "flat bar skin generated"

# ---------- 8d. lyric font ----------
# Noto Serif JP (SIL OFL 1.1). Pygame cannot select a variable-font weight, so
# a static Black (wght=900) instance is cut from the variable font.
step "Lyric font: Noto Serif JP Black"
if [ -f "$FONT_DIR/NotoSerifJP-Black.ttf" ]; then
  skip "NotoSerifJP-Black.ttf present"
else
  mkdir -p "$FONT_DIR"
  wget -c -O "$FONT_DIR/NotoSerifJP-VF.ttf" \
    "https://github.com/notofonts/noto-cjk/raw/main/Serif/Variable/TTF/Subset/NotoSerifJP-VF.ttf"
  "${VENV_DIR}/karaoke-jp/bin/pip" install --quiet fonttools
  "${VENV_DIR}/karaoke-jp/bin/python" -m fontTools.varLib.instancer \
    "$FONT_DIR/NotoSerifJP-VF.ttf" wght=900 --static \
    -o "$FONT_DIR/NotoSerifJP-Black.ttf"
fi

# ---------- 9. smoke test ----------
step "Smoke test"
"${VENV_DIR}/karaoke-jp/bin/karaoke-jp" --help >/dev/null && ok "karaoke-jp CLI OK"
"${VENV_DIR}/karaoke-jp-melody/bin/python" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
"${VENV_DIR}/karaoke-jp-lyrics/bin/python" -c 'import faster_whisper, fugashi; print("ASR + morphology OK")'
"${VENV_DIR}/karaoke-jp-render/bin/python" -c 'import pygame, cv2; print("Pygame", pygame.__version__, "OpenCV", cv2.__version__)'
"${VENV_DIR}/karaoke-jp-game/bin/python" -c 'import torch; print("GAME torch", torch.__version__, "cuda", torch.cuda.is_available())'
"${VENV_DIR}/karaoke-jp-align/bin/python" -c 'import transformers; print("aligner transformers", transformers.__version__)'

cat <<EOF

$(printf '\033[1;32mSetup complete.\033[0m')

First song:
  ${VENV_DIR}/karaoke-jp/bin/python scripts/download_song.py \\
    'https://youtu.be/<id>' -o songs/<song-id>/
  # paste the lyrics into songs/<song-id>/lyrics.txt
  ${VENV_DIR}/karaoke-jp/bin/snakemake --rerun-triggers mtime -j 1 \\
    outputs/<song-id>/karaoke.mp4

venvs live in ${VENV_DIR}/, third-party repos + checkpoints in ${TP}/.
EOF
