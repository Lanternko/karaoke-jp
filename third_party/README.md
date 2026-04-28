# third_party/

External repos used by the pipeline. Cloned by setup, **not vendored** —
gitignored to keep this repo small. Re-create with the commands below.

## openvpi/SOME — singing voice → MIDI (M2)

```bash
cd third_party
git clone --depth 1 https://github.com/openvpi/SOME.git

# Pretrained checkpoint (~435 MB; folder is named '256_5spk' inside the zip
# despite the file being '128_5spk' — upstream typo, see MEMORY.md).
cd SOME && mkdir -p pretrained && cd pretrained
wget https://github.com/openvpi/SOME/releases/download/v1.0.0-baseline/0119_continuous128_5spk.zip
unzip 0119_continuous128_5spk.zip

# RMVPE checkpoint for the default `karaoke-jp melody --backend rmvpe` path.
wget https://github.com/yxlllc/RMVPE/releases/download/230917/rmvpe.zip
rm -rf rmvpe rmvpe_unpack
unzip -o rmvpe.zip -d rmvpe_unpack
mkdir -p rmvpe
find rmvpe_unpack -name model.pt -exec mv {} rmvpe/model.pt \;
rm -rf rmvpe_unpack rmvpe.zip
```

Runtime venv (avoids fairseq + librosa<0.10 collision with main env):

```bash
python3 -m venv ~/venvs/karaoke-jp-melody
~/venvs/karaoke-jp-melody/bin/pip install \
  torch==2.11.0 torchaudio numpy<2 librosa<0.10.0 einops==0.6.1 \
  praat-parselmouth==0.4.3 lightning>=2.0.0 mido click PyYAML scipy \
  h5py matplotlib torchmetrics tqdm \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple
```

## keisuke-okb/MID2BAR-Player — JOYSOUND-style renderer (M4)

```bash
cd third_party
git clone --depth 1 https://github.com/keisuke-okb/MID2BAR-Player.git
```

Render venv (Pygame + opencv + the rest of MID2BAR's runtime; mic/GUI deps
are stubbed at call time by `scripts/render_mp4.py`, so we skip
`customtkinter` and the system-level PortAudio that `sounddevice` would
want):

```bash
python3 -m venv ~/venvs/karaoke-jp-render
~/venvs/karaoke-jp-render/bin/pip install \
  'pygame>2.6' Pillow numpy pandas py_midicsv mido \
  opencv-python chardet tqdm
```

Headless invocation: `SDL_VIDEODRIVER=dummy` (no X11/Wayland needed). LRC
format reverse-engineering + render notes are in `../MEMORY.md`.

## Lyrics venv (M3) — independent of third_party but worth pinning here

`faster-whisper` (CTranslate2) needs CUDA 12 cuBLAS / cuDNN at runtime, even
when other torch in the system is on CUDA 13:

```bash
python3 -m venv ~/venvs/karaoke-jp-lyrics
~/venvs/karaoke-jp-lyrics/bin/pip install \
  fugashi unidic-lite pyopenjtalk faster-whisper \
  nvidia-cublas-cu12 nvidia-cudnn-cu12 \
  numpy soundfile click

# At call time, prepend these to LD_LIBRARY_PATH so ctranslate2 finds cuBLAS:
export LD_LIBRARY_PATH="$HOME/venvs/karaoke-jp-lyrics/lib/python3.12/site-packages/nvidia/cublas/lib:$HOME/venvs/karaoke-jp-lyrics/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```
