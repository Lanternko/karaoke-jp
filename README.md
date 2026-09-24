# karaoke-jp

**Japanese karaoke video generator.** Give it a song and its lyrics and it
renders a 1080p60 sing-along video in the style of Japanese karaoke machines.
The video has discrete pitch bars, a per-mora lyric wipe, automatic furigana
(ruby), a technique HUD (しゃくり / こぶし / フォール / ビブラート) and a
background of your choice.

<img src="figures/main.png" width="100%" alt="Pitch bars (yellow = sung, white = upcoming), song-info HUD, per-mora lyric wipe with furigana" />

<img src="figures/demo.gif" width="100%" alt="Animated demo: scrolling pitch bars and lyric wipe (muted)" />

<table>
  <tr>
    <td width="33%"><img src="figures/wipe.png" alt="Per-mora wipe" /></td>
    <td width="33%"><img src="figures/furigana.png" alt="Furigana on kanji runs only" /></td>
    <td width="33%"><img src="figures/dualline.png" alt="Two-line lyrics + range/key gauge" /></td>
  </tr>
  <tr>
    <td align="center"><sub>Per-mora wipe</sub></td>
    <td align="center"><sub>Furigana on kanji runs only</sub></td>
    <td align="center"><sub>Two-line lyrics + range / key gauge</sub></td>
  </tr>
  <tr>
    <td><img src="figures/portrait_a.png" alt="Portrait 9:16 layout" /></td>
    <td><img src="figures/portrait_b.png" alt="Portrait 9:16 layout" /></td>
    <td></td>
  </tr>
  <tr>
    <td align="center" colspan="2"><sub>Portrait 9:16 layout: score, video, lyrics</sub></td>
    <td></td>
  </tr>
</table>

> ⚠️ **For personal singing practice only.** An accompaniment separated from
> a commercial recording is still a derivative of that recording. Do not
> upload the generated videos to YouTube or any other public platform.

## How it works

```
source audio (+ optional video)
  └─ separate        vocals / instrumental          (Mel-Band RoFormer)
       ├─ melody     note transcription             (GAME, seeded; SOME/RMVPE fallback)
       │    └─ display grid: fixed-width note values, breath gaps, one bar per mora
       └─ lyrics     tokenize + readings             (fugashi + UniDic)
            └─ timing  CTC forced alignment of the mora sequence
                       against the separated vocals (MMS-300m karaoke aligner)
  └─ render          MID2BAR-Player, headless → ffmpeg (h264 + aac)
```

- You supply the lyrics, and they are used as ground truth. The pipeline
  does not transcribe lyrics with ASR.
- All parameters live in one versioned profile, [config/versions.json](config/versions.json).
- Each stage runs in its own virtualenv, because the model stacks have
  conflicting pins.

## Requirements

- Linux with an NVIDIA GPU. Developed on an RTX 5090 with CUDA 13. RTX 50xx
  cards need the cu129/cu130 torch wheels that setup installs.
- Python 3.10–3.12, `git`, `wget`, `unzip`, `ffmpeg` and `curl`.
- About 10 GB of disk for the virtualenvs and checkpoints.

## Install

```bash
git clone https://github.com/Lanternko/karaoke-jp.git
cd karaoke-jp
bash scripts/setup.sh
```

`setup.sh` does the following, and is idempotent:
- Creates six venvs under `~/venvs/`.
- Clones the third-party repos into `third_party/`.
- Downloads the checkpoints.
- Generates the flat bar skin.
- Builds the lyric font (Noto Serif JP Black) into `~/.local/share/fonts/karaoke-jp/`.

For the manual steps, see [third_party/README.md](third_party/README.md).

## Usage

### GUI

```bash
~/venvs/karaoke-jp/bin/karaoke-jp gui        # http://127.0.0.1:7860
```

1. Paste a YouTube URL or upload an MP4.
2. Paste the lyrics.
3. Pick the vocal level and a background.
4. Render.

The GUI binds to localhost only.

### Command line

```bash
# 1. fetch audio (+ background video) into a song directory
~/venvs/karaoke-jp/bin/python scripts/download_song.py 'https://youtu.be/<id>' -o songs/<song-id>/

# 2. put the lyrics in songs/<song-id>/lyrics.txt (one sung line per line)

# 3. render
~/venvs/karaoke-jp/bin/snakemake --rerun-triggers mtime -j 1 outputs/<song-id>/karaoke.mp4
```

The song directory holds:
- `source.{wav,mp3,m4a,flac}`: the audio.
- `lyrics.txt`: the lyrics.
- `background.{mp4,webm,mov,mkv,png,jpg}` (optional): the background. Without
  one, the default gradient is used.

Environment knobs:
- `VOCAL_RATIO=0.0–1.0`: guide-vocal level in the mix.
- `KARAOKE_PROFILE=<id>`: render with a non-default profile.

### Per-song fixes (optional)

Furigana and alignment can never be 100% right, so small JSON files in
`overrides/` let you correct them without touching code:

| file | purpose |
|---|---|
| `overrides/<song>.json` | reading overrides, flat `{"漢字": "よみ"}`. Fixes the displayed furigana *and* the alignment. |
| `overrides/<song>_pitch_patch.json` | display-only pitch fixes, melisma splits and note drops (see `scripts/make_display_grid.py`) |
| `overrides/<song>_render.json` | `bg_mask` (hide lyrics burned into a video), `lrc_breaks` (manual line breaks), `key` (fixed key for the HUD) |

`scripts/romaji_overrides.py` suggests reading fixes when you have a romaji
version of the lyrics. Review each suggestion by hand before you accept it.

### Portrait (9:16)

`scripts/make_portrait_grid.py` builds the portrait layout from the same chain
outputs. `scripts/render_portrait.py --help` shows the rendering options.

## Third-party components

These are fetched by setup and are not redistributed here:

| component | use | license |
|---|---|---|
| [openvpi/GAME](https://github.com/openvpi/GAME) | note transcription | MIT |
| [openvpi/SOME](https://github.com/openvpi/SOME) | melody fallback | MIT |
| [RMVPE](https://github.com/yxlllc/RMVPE) checkpoint | f0 tracking | see upstream |
| [york135/CTC_CE_for_AST](https://github.com/york135/CTC_CE_for_AST) | alternative transcription backend | see upstream |
| [keisuke-okb/MID2BAR-Player](https://github.com/keisuke-okb/MID2BAR-Player) | renderer | Apache-2.0 |
| [NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn](https://huggingface.co/NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn) | CTC forced alignment | see model card |
| Mel-Band RoFormer, faster-whisper, fugashi / UniDic, Essentia, Noto Serif JP | separation, tokenization, key detection, font | respective licenses |

## License

[PolyForm Noncommercial 1.0.0](LICENSE). You may use, modify and share this
software for any **noncommercial** purpose, including personal use, study,
research, hobby projects, and work at charities and educational institutions.
**Commercial use is not permitted**, including selling the software or its
output, building a paid product or service on it, or using it inside a
company. For a commercial license, open an issue.

Copyright (c) 2026 Lanternko.

---

## 中文簡介

日式卡拉 OK 影片自動生成器。給它一首歌加上歌詞，就會輸出 1080p60 的練唱影片，
內含離散音高方塊、逐 mora 歌詞 wipe、自動振假名、原唱技巧 HUD，以及自選背景。

- 安裝：`bash scripts/setup.sh`
- 使用：先下載歌曲，把歌詞寫進 `songs/<id>/lyrics.txt`，再執行
  `snakemake --rerun-triggers mtime -j 1 outputs/<id>/karaoke.mp4`。
  也可以改用 `karaoke-jp gui`。
- **僅限個人練唱，請勿上傳公開平台。** 分離出的伴奏仍是原盤的衍生物。
- 授權為 PolyForm Noncommercial 1.0.0：可自由用於非商業用途，**禁止商業使用**。
