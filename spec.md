# karaoke-jp — 日式卡拉 OK 影片自動生成器 規格書

**版本**：v1.0（2026-04-28，研究階段結論）
**狀態**：尚未實作；stack 已定，可開工

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

## 4. 專案結構（建議）

```
karaoke-jp/
├── CLAUDE.md                 # 給 Claude session 的工作指南
├── MEMORY.md                 # 關鍵決策 / 踩坑記錄
├── spec.md                   # 本檔
├── README.md                 # 給人類看的快速上手
├── pyproject.toml            # uv / pip
├── Snakefile                 # 批次 pipeline
├── envs/                     # per-rule conda env
│   ├── separation.yaml
│   ├── pitch.yaml
│   ├── lyrics.yaml
│   └── render.yaml
├── src/karaoke_jp/
│   ├── __init__.py
│   ├── download.py           # yt-dlp wrapper
│   ├── separate.py           # melband-roformer wrapper（local + ssh）
│   ├── melody.py             # RMVPE + SOME → MIDI
│   ├── lyrics.py             # whisper + sofa + fugashi → ruby-LRC
│   ├── ruby.py               # 振假名邏輯（Yomikata + override）
│   ├── render.py             # MID2BAR fork 接口
│   └── mix.py                # 人聲音量調混（ffmpeg amix）
├── overrides/                # per-song gikun override JSON
│   └── <song-id>.json
├── songs/                    # 輸入
│   └── <song-id>/
│       ├── source.wav
│       ├── lyrics.txt        # 已知歌詞（從 Genius / 自打）
│       └── background.png
├── outputs/
│   └── <song-id>/
│       ├── vocals.wav
│       ├── instrumental.wav
│       ├── melody.mid
│       ├── ruby.lrc
│       └── karaoke.mp4
└── third_party/
    └── mid2bar-player/       # fork
```

---

## 5. 實作里程碑

**M1 — 人聲分離跑通**（半天）
- 安裝 `melband-roformer-infer`
- SSH 到 GPU 機跑一首歌
- 確認 vocals / instrumental 品質 ✅ 即往下

**M2 — Melody MIDI 跑通**（一天）
- vocals.wav → RMVPE → SOME → MIDI
- 用 MIDI viewer（如 signal）目視驗證 melody 對不對
- 量化的 onset/offset 跟原唱對得上

**M3 — 歌詞 ruby-LRC 跑通**（兩天）
- mlx-whisper 抽歌詞
- SOFA 對齊到 mora
- fugashi + UniDic + Yomikata 加 ruby
- 找一首有 gikun 的歌測試 override 機制

**M4 — MID2BAR-Player fork 接通**（兩天）
- clone + 看 render code
- 把 M2/M3 輸出餵進去
- 出第一個 MP4

**M5 — 人聲音量調混**（半天）
- ffmpeg amix 多版輸出（30% / 50% / 100% / off）

**M6 — Snakemake 批次化**（半天）
- 50 首 fan-out
- 中間 cache（M1 的分離最貴，最該 cache）

**M7 — 振假名品質 polish**（持續）
- 用真歌測試 → 補 override JSON
- 視需要加 LLM correction pass（GPT-4 / Claude 知道熱門歌的 gikun）

---

## 6. 開放問題 / 已知風險

1. **MID2BAR-Player 成熟度** — 報告說它已實作 JOYSOUND 風格，但沒驗證星數 / 活躍度 / API 穩定性。M4 進場前要先看 repo，如果死了要備案。
   - 備案：自己用 Pillow + ffmpeg-python 寫 ~300 行 renderer

2. **SOFA 日文 dict 完整度** — `lottev1991/opencpop-cjke-multidict` 是社群 dict，覆蓋度未知。如果不行：
   - 備案 1：ESPnet CTC segmentation（OWSM-CTC v4，CPU OK）
   - 備案 2：Julius segmentation-kit

3. **Gikun（特殊讀音）** — 任何工具都救不了「運命=さだめ」這種，必須有 override 機制。第一批歌要選讀音正常的（Vocaloid 純假名歌詞最安全），確認 pipeline 通了再挑戰 gikun 多的歌。

4. **音高量化粒度** — SOME 輸出非整數 MIDI（如 60.5）。要決定 render 時是否四捨五入到半音 / 全音；保留浮點可以畫出 vibrato 抖動。

5. **背景圖音高方塊壓字** — 背景太亮會吃掉歌詞 / 方塊。要不要加半透明黑色 overlay？

6. **批次速度** — 沒實測 SSH 來回 + GPU 排隊的延遲，可能比想像慢。M6 前先實測。

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
