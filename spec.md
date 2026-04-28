# karaoke-jp — 日式卡拉 OK 影片自動生成器 規格書

**版本**：v1.1（2026-04-28，M0-M4 wired）
**狀態**：M0-M4 端到端跑通（兩首歌驗證），polish 階段（M5/M6/M7）pending

---

## 1. 目標

從**任意一首歌**（YouTube 連結 / mp3 / wav）自動產生 **JOYSOUND / DAM 風格**的卡拉 OK 影片，**個人練唱用**。

### 1.1 視覺規格

參考 JOYSOUND 機台 + 部分 YouTube 卡拉 OK 頻道（例：用 AI 生成背景的個人創作者）：

- **上方**：離散音高方塊（一音節一塊，垂直位置 = 音高，左側紅色豎線當播放游標）
- **下方**：日文歌詞，**目前唱到的字逐字 wipe / 反白**
- **振假名**：漢字上方標小假名（信→しん、気→き、君→きみ）
- **背景**：自選圖（AI 生成 / 風景照 / 動漫截圖皆可）
- **次行歌詞淡入預告**（右下或下行）

**不做**：Vocal Pitch Monitor 風格的連續 f0 曲線（已否決，要的是離散方塊）。

### 1.2 功能需求

- 自動分離人聲 / 伴奏
- 自動抽人聲 melody → 量化成音高方塊
- 自動歌詞辨識 + 假名 mora 級時間對齊
- 自動振假名標註（漢字 → ruby）
- 人聲音量可調混（30%、50%、100% 多版輸出當練習音源）
- 輸出 1080p MP4

### 1.3 非功能需求

- 開發機：Mac Apple Silicon
- 重運算（人聲分離）可用 SSH GPU 跑
- 整曲 4 分鐘處理時間：本機 < 15 分鐘可接受，GPU 上 < 3 分鐘
- 可批次（一次跑 50 首）

---

## 2. 法律 stance（重要）

### 2.1 用途限定：**個人使用，不上傳，不分享**

依據日本著作権法**第 30 条「私的使用のための複製」**：個人或家庭範圍內的重製合法。

具體允許：
- 本機跑 Demucs / Mel-Band-RoFormer 分離商業歌
- 本機產 MP4 自己耳機聽 / 自己手機同步
- 不破解 DRM 的前提下取得來源（自己買的 mp3、YouTube 一般下載 OK）

具體**禁止**：
- 上傳 YouTube / niconico / Bilibili
- Google Drive 公開連結 / 雲端「分享」給朋友
- 任何形式的散布

### 2.2 為什麼不能上傳

Demucs / Mel-Band-RoFormer 分離出的伴奏**仍然是原盤的衍生**（衍生著作權 / 著作隣接権 / 原盤権）：

- JASRAC blanket license 只 cover 著作権（作詞 / 作曲），**不 cover 原盤権**
- 原盤権由唱片公司持有（Sony Music Japan、Avex、Universal、Pony Canyon）
- 第一興商一年發 12 萬封 takedown
- 重複觸發 Content ID 三次 / 90 天會被刪頻道

→ 上傳路線需改用：自製伴奏 / Vocaloid (Piapro 釋出 off-vocal) / 公版。**本專案不走上傳路線**。

### 2.3 Model weight 授權

- Mel-Band-RoFormer Kim weight 商用授權不乾淨 → 自用無影響
- SOME weight CC-BY-NC-SA 4.0（非商用）→ 自用無影響
- 程式碼全部 MIT / Apache → 隨便用

---

## 3. 最終 stack（已定）

### 3.1 元件總表

| 階段 | 工具 | 備註 |
|---|---|---|
| 下載 | `yt-dlp` | Unlicense |
| 人聲分離 | **`openmirlab/melband-roformer-infer`** | pip 裝；default model = KimberleyJensen Mel-Band-RoFormer Kim；70+ fine-tune 可換 |
| Pitch 抽取 | **RMVPE** | 2026 SOTA on vocals；frame-level F0 |
| Note 量化（onset/offset） | **`openvpi/SOME`** | RMVPE backbone；輸出 MIDI with note-on/off；無需另跑 madmom |
| 歌詞辨識 | **mlx-whisper**（large-v3-turbo） | Mac 上比 whisper.cpp 快 2× |
| Word-level 對齊 | WhisperX 的 wav2vec2 alignment | MPS 可跑（WhisperX 主程式不行） |
| Mora-level 對齊 | **SOFA**（`qiuqiao/SOFA`） | Singing-Oriented Forced Aligner，用 `lottev1991/opencpop-cjke-multidict` 加日文 |
| Tokenize / 讀音 | **fugashi + UniDic** | 形態素解析；不要用 pykakasi |
| 異讀詞修正 | **Yomikata BERT**（130 個 heteronym） | 運命 / 本気 等 |
| 罕用讀音 / gikun | **per-song JSON override** | 運命=さだめ、宇宙=そら 必須手動寫 |
| 字幕渲染 | **Aegisub karaskel + KaraTemplater**（The0x's fork） | 產 ASS，用 libass render；ASS 沒原生 ruby，靠雙行 trick |
| 影片合成主層 | **MID2BAR-Player**（fork） | 已實作 JOYSOUND 風格音高方塊 + ruby LRC + 背景；輸入 MIDI + ruby-LRC，輸出 MP4 |
| 編碼 | `ffmpeg-python` pipe raw RGB | > 100 fps |
| 批次 orchestration | **Snakemake** | 50 首 fan-out 一行；conda env per rule |

### 3.2 Pipeline 架構

```
[input]
  YouTube URL / mp3
        │
        ├─ yt-dlp ──────────────────────► raw audio (wav)
        │
        ▼
[separation] (SSH GPU 或本機)
  melband-roformer-infer (Kim model)
        │
        ├──► vocals.wav
        └──► instrumental.wav
                                          │
[melody]                                   │
  vocals.wav ─► RMVPE ─► f0 contour       │
              ─► SOME  ─► melody MIDI     │
                          (note onsets)    │
                                          │
[lyrics]                                   │
  vocals.wav ─► mlx-whisper ─► raw text   │
              ─► WhisperX align ─► word ts │
              ─► SOFA ─► mora ts          │
              ─► fugashi+UniDic ─► tokens  │
              ─► Yomikata + override JSON ─► ruby
              ─► ruby-LRC                  │
                                          │
[render]                                   │
  MID2BAR-Player(fork)                    │
   ← melody MIDI                          │
   ← ruby-LRC                             │
   ← background.png                       │
   ← instrumental.wav (mix volume)         │
        │                                  │
        ▼                                  │
  ffmpeg encode ─► out.mp4 ◄──────────────┘
```

### 3.3 Mac Apple Silicon 注意事項

| 工具 | MPS 狀態 | 對策 |
|---|---|---|
| Mel-Band-RoFormer | PyTorch MPS 部分支援 | **建議 SSH 到 GPU 機跑**（這步本來就最重） |
| RMVPE | 可 MPS | `device='mps'`，加 `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| SOME | 可 MPS | 同上 |
| WhisperX | **MPS 壞掉** | 換 `mlx-whisper`，alignment 那段才用 WhisperX 的 wav2vec2 |
| SOFA | PyTorch MPS 可跑 | 同上 |
| pyannote.audio | MPS 偶有 timestamp 亂掉 | 第一首 sanity check vs CPU |

→ 結論：**人聲分離跑 SSH，其他全本機 Mac**。本機 dev / debug 速度快，分離一次跑完拉檔。

---

## 4. 專案結構（實況，2026-04-28）

```
karaoke-jp/
├── CLAUDE.md                 # 給 Claude session 的工作指南
├── MEMORY.md                 # 關鍵決策 / 踩坑記錄
├── spec.md                   # 本檔
├── README.md                 # 給人類看的快速上手
├── pyproject.toml            # uv / pip（optional-dependencies 分階段 extras）
├── Snakefile                 # 8 rule pipeline（separate, melody, tokenize,
│                               asr, align, midi_markers, export_lrc, render）
├── src/karaoke_jp/
│   ├── __init__.py
│   ├── cli.py                # click dispatcher（separate, melody）
│   ├── separate.py           # melband-roformer wrapper（M1）
│   ├── melody.py             # SOME subprocess wrapper（M2）
│   ├── ruby.py               # fugashi tokenizer + reading（M3）
│   ├── align.py              # kana-aware NW + 時間戳 back-map（M3）
│   ├── lrc_export.py         # aligned.json → MID2BAR LRC（M4）
│   ├── midi_markers.py       # mido 注入 markers + ts_signature（M4）
│   ├── download.py           # （M0 stub，實際在 scripts/download_song.py）
│   ├── lyrics.py             # （stub，邏輯散在 scripts/run_asr + tokenize + align_lyrics）
│   ├── render.py             # （stub，實際在 scripts/render_mp4.py）
│   └── mix.py                # （stub，M5 留位）
├── scripts/                  # 各階段 venv-segregated CLI
│   ├── download_song.py      # M0
│   ├── tokenize_lyrics.py    # M3a
│   ├── run_asr.py            # M3b（faster-whisper）
│   ├── align_lyrics.py       # M3c
│   ├── add_midi_markers.py   # M4a
│   ├── export_lrc.py         # M4b
│   ├── render_mp4.py         # M4c（headless MID2BAR-Player）
│   └── plot_pianoroll.py     # debug viz
├── overrides/                # per-song gikun override JSON（pending M7）
│   └── <song-id>.json
├── songs/                    # 輸入（user-curated）
│   └── <song-id>/
│       ├── source.wav        # gitignored
│       ├── lyrics.txt        # tracked
│       ├── source.md         # tracked（artist / URL / gikun notes）
│       └── background.{mp4,png}  # optional, gitignored
├── outputs/                  # all gitignored
│   └── <song-id>/
│       ├── vocals.wav, instrumental.wav     # M1
│       ├── melody.mid, melody_markers.mid   # M2 + M4
│       ├── tokens.json, asr.json, aligned.json  # M3
│       ├── karaoke.lrc                      # M4
│       ├── _background.mp4                  # M4 normalized bg
│       └── karaoke.mp4                      # final
├── figures/                  # piano roll PNGs + frame extracts，sample size
└── third_party/              # NOT vendored — see third_party/README.md
    ├── SOME/                 # cloned, pretrained/0119_continuous256_5spk/
    └── MID2BAR-Player/       # cloned
```

**已撤** `envs/*.yaml`（Snakemake conda 沒 wire）。M6 真要批次跑時可以再加。

---

## 5. 實作里程碑

**M0 — Download** ✅ done
- `scripts/download_song.py <url> -o songs/<id>/` 抓 audio + bg video（一鍵）
- bg 偏好 h264 720p（vcodec*=avc1），跳過 AV1 避開 OpenCV 軟解 gap
- `--no-video` for Lyric Video 上傳

**M1 — 人聲分離** ✅ done（實際半天）
- `melband-roformer-infer` KJ Kim model，CUDA 13 主 venv
- 實測 RTX 5090：4 分鐘 song GPU 7 秒（RTF ≈ 0.03）

**M2 — Melody MIDI** ✅ done（實際半天）
- openvpi/SOME v1.0.0-baseline，獨立 venv 避 librosa/numpy 衝突
- 實測：4 分鐘 song GPU 8 秒，626 notes，piano roll 結構符合 J-pop
- **發現**：me_infer.py 實際走 parselmouth，不是 RMVPE（config 是 stale）→ fairseq 不用裝

**M3 — 歌詞 ruby-LRC** ✅ v1 done（mora-level → v2 留 SOFA）
- ASR：faster-whisper large-v3 + lyrics initial_prompt 偏置開頭
- Tokenize：fugashi + UniDic-lite
- Align：char→kana 雙向 stream + Needleman-Wunsch（codex round 1 翻過：raw char distance 在 kanji/kana mismatch 會崩）
- 實測：tuki-zero 538 lyrics chars vs 503 ASR chars，88% kana 直接配對
- v2 計畫：SOFA forced alignment 替代 Whisper proportional split，melisma / 拖長母音更準

**M4 — MID2BAR-Player fork** ✅ done（實際半天）
- Headless：SDL_VIDEODRIVER=dummy + tkinter/sounddevice stub
- LRC：`[mm:ss:cs]` centiseconds + `@RubyN=base,ruby,start,end` header（自寫 export ~70 行）
- MIDI：mido 注入 page-boundary markers + default 4/4 time_signature
- BG：optional, auto-detect songs/<song>/background.{mp4,png,...}, 一律 ffmpeg normalize 成 h264 1080p

**M5 — 人聲音量調混** pending（半天估算）
- ffmpeg amix 多版輸出（30% / 50% / 100% / off）
- Snakefile rule，跟 M4 平行（不卡 render）

**M6 — Snakemake 批次化** partially done
- 已 wire：DAG 8 個 rule，single song 跑通
- TODO：50 首 fan-out 實測、conda env per rule（取代當前 venv）、`--rerun-triggers params input code` 配置

**M7 — 振假名品質 polish** pending
- Yomikata BERT pass 處理 130 個 heteronym
- 加 `overrides/<song-id>.json` gikun 機制 + LLM correction pass

---

## 6. 開放問題 / 已知風險

1. ~~**MID2BAR-Player 成熟度**~~ ✅ 解決：fork 確認可行，read-code 驗證了 offline frame pipe（2026-04-28）

2. **SOFA 日文 dict 完整度** — `lottev1991/opencpop-cjke-multidict` 是社群 dict，覆蓋度未知。M3 v2 polish 時驗證。
   - 備案 1：ESPnet CTC segmentation（OWSM-CTC v4，CPU OK）
   - 備案 2：Julius segmentation-kit

3. **Gikun（特殊讀音）** — 已驗證在 tuki-zero（標題「零」是 ゼロ 但歌詞 絶対零度 的「零」是 れい，fugashi 自動分對了）+ bocchi-guitar（無 gikun）。M7 加 override JSON 機制給「運命=さだめ」這類完全救不了的。

4. ~~**音高量化粒度**~~ ✅ 解決：SOME v1.0.0-baseline 實測仍是整數 MIDI（非整數要等 GAME / 連續模型）。當前夠用。

5. **背景圖音高方塊壓字** — bocchi-guitar 用原 Lyric Video 當 bg 時，原片的燒入字會跟我們歌詞層撞。VIDEO_ALPHA 可降但全曲統一改。M7 polish 可加 per-song setting override。

6. ~~**批次速度**~~ ✅ 部分解決：4 分鐘 song 從 download 到 mp4 ~3-4 分鐘（含 ASR model load）。M6 50 首 fan-out 還沒實測，但 cache 友善的 DAG 已驗。

7. **M3 alignment 在 instrumental-heavy 段** — Whisper 在 0-30s pure instrumental 會幻覺成「ご視聴ありがとう」/「初音ミク」，要 lyrics initial_prompt 偏置才能搶回 verse 1。當前已實作但歌曲尾段（Whisper missed quiet 收尾）還是要靠 char-level interpolation 填，誤差累積。

8. **Mac MPS path 從未實測** — 原 spec 設計成 Mac 為主、SSH GPU 跑 M1，實際到目前都在 Linux GPU 跑。Mac on-board 跑要重新驗證。

---

## 7. Out of scope（明確不做）

- 上傳 YouTube / 任何公開散布（法律問題，見 §2）
- 連續 f0 曲線視覺（Vocal Pitch Monitor 風格，已否決）
- Real-time 麥克風 pitch matching / 評分（MID2BAR 內建有，但本專案不需要）
- 中文 / 英文歌支援（先做日文，其他語言後續）
- 自製伴奏 / DAW 編曲（用商業歌分離，自用範圍內）
- GUI（CLI + Snakemake 即可）

---

## 8. 參考資料

### 8.1 主要 repo
- `openmirlab/melband-roformer-infer` — 人聲分離 inference wrapper
- `openvpi/SOME` — singing-voice → MIDI（含 RMVPE backbone）
- `qiuqiao/SOFA` — Singing-Oriented Forced Aligner
- `keisuke-okb/mid2bar-player` — JOYSOUND 風格 renderer（待 fork）
- `The0x539/Aegisub-Scripts` — KaraTemplater 現代 fork
- `ml-explore/mlx-examples`（whisper） — Mac 加速
- `polm/fugashi` + UniDic — 日文形態素
- `passaglia/yomikata` — heteronym BERT

### 8.2 不要再走的路
- ❌ pykakasi（讀音爛、無形態素）
- ❌ CREPE（已被 RMVPE / PESTO 取代）
- ❌ WhisperX（Mac MPS 壞掉）
- ❌ libass 原生 ruby（不存在，要靠 ASS 雙行 trick）
- ❌ Fork UltraSinger（沒 ruby 格式、沒 MP4 export，工作量比 fork MID2BAR 大）
- ❌ 從零寫 renderer（MID2BAR 已存在，重造輪子）

### 8.3 已完成研究
- 兩輪 deep research 結論已內化到本 spec
- 詳細討論過程：`MEMORY.md`
- M4 翻案 incident（sub-agent 誤判 MID2BAR 是 screen capture）：`MEMORY.md` "Fork-vs-build 決策 checklist" 段
- Codex remote review 兩輪迭代記錄：`MEMORY.md` "Codex review 記錄" 段
