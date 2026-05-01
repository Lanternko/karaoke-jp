<img width="1421" height="811" alt="image" src="https://github.com/user-attachments/assets/c4d2b381-109c-47ec-a2e2-a99b13abc8ec" /># karaoke-jp

JOYSOUND 風日式卡拉 OK 影片自動生成器。**個人練唱用，不要上傳。**

輸入一個 YouTube URL 或本地音訊，輸出 1080p60 的 MP4，包含離散音高方塊、逐字歌詞 wipe、振假名（ruby）、可選背景圖／影片。

完整設計、決策歷史、踩坑記錄分別在 [spec.md](spec.md) / [MEMORY.md](MEMORY.md) / [CLAUDE.md](CLAUDE.md)。

<img width="1421" height="811" alt="image" src="https://github.com/user-attachments/assets/46217b8e-0d26-4b62-8c7b-fb5c241485ce" />




## 重要：版權與使用範圍

- 僅限**個人練唱自用**。日本著作權法第 30 條私的使用允許自用 reproduction
- **不要上傳產出**到 YouTube / 任何公開平台。Demucs / RoFormer 分離出的伴奏仍是原盤的衍生作品，JASRAC blanket license 不 cover；唱片公司會 Content ID claim
- 公開分享請改走完整商用授權路徑（與本專案無關）

## Pipeline

```
YouTube URL / 本地音訊
  └─ download   ─► source.wav, background.mp4         (M0, yt-dlp)
       └─ separate  ─► vocals.wav, instrumental.wav   (M1, Mel-Band-RoFormer)
            └─ melody  ─► melody.mid                  (M2, RMVPE / CTC+CE)
            └─ tokenize + asr + align ─► aligned.json (M3, fugashi + faster-whisper)
                 ├─ midi_timing  ─► aligned_midi.json (mora→note 對齊)
                 ├─ midi_markers ─► melody_markers.mid
                 ├─ export_lrc   ─► karaoke.lrc       (MID2BAR 格式 + @RubyN)
                 ├─ mix          ─► mixed.wav         (instrumental + 20% guide vocal)
                 └─ render       ─► karaoke.mp4       (M4, headless MID2BAR-Player)
```

每個階段是一條 Snakemake rule，cache 友善：改了哪一支腳本只會重跑下游。

## 使用的技術

| 階段 | 技術 | 用途 |
|---|---|---|
| **M0 下載** | yt-dlp | YouTube 拉音訊 + 背景影片，避開 AV1（OpenCV 軟解不了），優先 h264 720p |
| **M1 人聲分離** | [openmirlab/melband-roformer-infer](https://github.com/openmirlab/melband-roformer-infer)（KimberleyJensen Mel-Band-RoFormer Kim FT2 Bleedless）| 把 source 拆成 vocals + instrumental |
| **M2 melody 抽取** | RMVPE（預設）／ [york135/CTC_CE_for_AST](https://github.com/york135/CTC_CE_for_AST)（`MELODY_BACKEND=cectc`）／ [openvpi/SOME](https://github.com/openvpi/SOME)（legacy） | vocals → MIDI 音符序列 |
| **M2b quantize** | beat-grid snap | note duration 對齊到 8th / quarter / half，BPM 寫進 sidecar |
| **M3a 形態素解析** | [fugashi](https://github.com/polm/fugashi) + UniDic | 切詞 + 取讀音（reading）|
| **M3b ASR** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (large-v3) | 唱詞 → kana stream，初頭用歌詞前 150 字當 `initial_prompt` 偏置 |
| **M3c 對齊** | Needleman-Wunsch（kana-aware） | ASR kana stream ↔ 歌詞 token 比對，產生 char-level timing skeleton |
| **M3d MIDI timing** | mora→note bounded greedy match | 把 Whisper char timing 換成 MIDI note onset，「再会」4 morae 吃 4 個 notes |
| **M4a LRC export** | 自製 [@RubyN= 拆分](src/karaoke_jp/lrc_export.py) | aligned.json → MID2BAR `[mm:ss:cs]` LRC，per-kanji-run 發 ruby（hiragana 不發）|
| **M4b 頁面 markers** | beat-based | 每 8 quarter notes 一頁，固定 pixels-per-quarter，bar 顯示穩定 |
| **M5 mix** | ffmpeg `amix` | instrumental + 20% guide vocal → mixed.wav |
| **M4c render** | [keisuke-okb/MID2BAR-Player](https://github.com/keisuke-okb/MID2BAR-Player) (forked headless) | MIDI + LRC + audio → 1080p60 h264 MP4，bg 自動 codec normalize |

## 一鍵跑完（已 setup 後）

```bash
# 1. 下載歌
~/venvs/karaoke-jp/bin/python scripts/download_song.py \
  'https://youtu.be/<id>' -o songs/<song-id>/

# 2. 把歌詞貼進 songs/<song-id>/lyrics.txt（純文字，每行一句）

# 3. 跑整條 pipeline
~/venvs/karaoke-jp/bin/snakemake --rerun-triggers mtime -j 1 \
  outputs/<song-id>/karaoke.mp4
```

產出：`outputs/<song-id>/karaoke.mp4`。

### Lyric Video 型背景的處理
如果 YouTube 來源是已經燒入歌詞的 Lyric Video（如 Aniplex 動畫官方上傳），加 `--no-video` 跳過 video 下載，自己放一張 `songs/<song-id>/background.png` 當背景，避免歌詞層撞圖。

### 切換 melody backend

```bash
MELODY_BACKEND=cectc snakemake --rerun-triggers mtime -j 1 \
  outputs/<song-id>/karaoke.mp4
```

`cectc` 是 Wang & Jang TASLP 2023 的 CRNN+CTC+CE，直接輸出 onset/offset/pitch 一次到位，低音域比 RMVPE 穩（RMVPE 會掉八度）。

### Score-first（已有樂譜 MIDI）

如果是鋼琴 cover、有可信賴的樂譜 MIDI，**不要**讓 RMVPE / BasicPitch 從多聲部鋼琴音檔猜旋律。把樂譜當 pitch ground truth，audio 只決定 timing：

```bash
~/venvs/karaoke-jp/bin/pip install -e '.[score]'

~/venvs/karaoke-jp/bin/karaoke-jp score-melody \
  songs/<song-id>/source.wav \
  --score-midi songs/<song-id>/score.mid \
  -o outputs/<song-id>/melody.mid
```

旗標：
- `--top-voice`（預設）：每個 onset 取最高音，雙手譜也能用
- `--all-notes`：DTW 對齊後保留全譜
- `--tempo 93`：強制寫進 melody.mid 的 tempo metadata

## 安裝

四個 venv，刻意分開（套件版本互衝）：

| venv | 負責階段 | 為什麼分開 |
|---|---|---|
| `~/venvs/karaoke-jp/`         | M0 下載、M1 separate、M4 prep、CLI | 主環境，torch 2.11+cu13 |
| `~/venvs/karaoke-jp-melody/`  | M2 SOME / CTC+CE 推理 | librosa<0.10 + numpy<2 跟主 venv 撞 |
| `~/venvs/karaoke-jp-lyrics/`  | M3 ASR + tokenize + align | faster-whisper 要 CUDA 12 cuBLAS shim |
| `~/venvs/karaoke-jp-render/`  | M4 render | Pygame + opencv-python，跟主 venv 的 click 衝 |

四個 venv 完整 setup（含 third_party clone、checkpoint 下載）見 [`third_party/README.md`](third_party/README.md)。

## 目錄結構

```
karaoke-jp/
├── src/karaoke_jp/         # 主 package（separate, melody, ruby, align,
│                             lrc_export, midi_markers, cli）
├── scripts/                # 各 stage subprocess 跑的 CLI
├── Snakefile               # 11 條 rule：separate → melody → quantize →
│                             tokenize / asr → align → midi_timing →
│                             {export_lrc, midi_markers} → mix → render
├── config/mid2bar_settings.json   # MID2BAR 渲染設定
├── songs/<song-id>/        # source.wav, lyrics.txt, source.md,
│                             background.{mp4,png}（音檔/影片 gitignored）
├── outputs/<song-id>/      # 所有中間檔（gitignored）
├── overrides/              # 每首歌的 ruby override JSON（gikun 用）
├── third_party/            # 外部 repo（gitignored，setup script clone）
├── spec.md  CLAUDE.md  MEMORY.md
└── pyproject.toml  README.md  Snakefile
```

## 設計取捨（不要再爭論）

- **Fork MID2BAR-Player，不從零寫 renderer**：MID2BAR 已實作 JOYSOUND 風格的 ruby + 音高方塊 + ffmpeg encode；UltraStar 系列無 ruby 語法且不支援 MP4 export
- **Timing 用 mora→note alignment**：Whisper word timestamp 對唱歌不準，「再会」會被切成 2 char 而不是 4 mora；改用 MIDI note onset per-mora 對齊，sustain 尾段抓得到
- **不做連續 f0 曲線**：要的是 JOYSOUND 風格離散音高方塊，Vocal Pitch Monitor 風格已否決
- **不用 pykakasi 做振假名**：沒形態素解析，常選錯讀音；改用 fugashi + UniDic
- **不用 CREPE 做 pitch**：2026 已被 RMVPE / SOME 取代

更多踩坑見 [MEMORY.md](MEMORY.md)。
