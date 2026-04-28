# karaoke-jp

## Why
日式卡拉 OK 影片自動生成器（JOYSOUND 風格：離散音高方塊 + 逐字歌詞 wipe + 振假名 + 自選背景）。**使用者 自用練唱**，不上傳。

## 狀態
**M1-M4 全 wire（2026-04-28）**，第一首歌 tuki. 「零-zero-」end-to-end 跑通：
`outputs/tuki-zero/karaoke.mp4`（1080p60，237.5s 對齊原曲，pitch bars + per-char wipe + furigana 全部達成）。
下一步：polish（M5 amix 多音量版 / M6 Snakemake 批次 / M7 Yomikata + override JSON）。

## 三件套
- [spec.md](spec.md) — 完整技術規格、pipeline、里程碑
- [MEMORY.md](MEMORY.md) — 關鍵決策、踩坑、為什麼選 X 不選 Y
- 本檔（CLAUDE.md）— 工作指南

## 關鍵決策（不要再爭論）
1. **Fork MID2BAR-Player**，不 fork UltraSinger，不從零寫 renderer。Why: MID2BAR 已實作 JOYSOUND 風格 ruby + 音高方塊 + ffmpeg encode。**驗證細節（讀過 code 確認）**：
   - `framerecorder.py` = headless offline frame pipe（`pygame.image.tostring(surface, "RGB")` → ffmpeg stdin），**不是螢幕錄製**。Recording 模式 `current_time = frame_idx / fps` deterministic
   - LRC 格式：`[mm:ss:cs]`（colon 不是 dot，centiseconds 不是 ms），ruby 是 header `@RubyN=base,ruby,start,end`，不是 inline `(かな)`
   - Linux headless：跑前設 `SDL_VIDEODRIVER=dummy`
   - 入口：`main.py` 是 hard-coded paths sample，要包成 CLI
2. **人聲分離跑 SSH GPU**，其他全本機 Mac。Why: Mel-Band-RoFormer 在 MPS 上半殘；其他工具在 Mac 跑得動。
3. **WhisperX 主程式換成 mlx-whisper**，只留 wav2vec2 alignment 用 WhisperX。Why: WhisperX 在 MPS 壞掉（sparse_coo_tensor / repeat_interleave fail）。
4. **個人使用，不上傳**。Why: 著作権法 30 条（私的使用）允許自用；上傳會撞原盤権 + Content ID。詳見 spec §2。
5. **不做連續 f0 曲線**（Vocal Pitch Monitor 風格已否決）。要的是離散音高方塊。

## NEVER
- **不要把專案產出上傳 YouTube / 公開分享。** Why: Demucs / RoFormer 分離出的伴奏仍是原盤的衍生，JASRAC blanket license 不 cover；Avex / SME / Universal 會 Content ID claim 或刪頻道。詳見 [spec.md §2](spec.md)。
- **不要用 pykakasi 做振假名。** Why: 沒形態素解析，常選錯讀音。改用 fugashi + UniDic + Yomikata。
- **不要用 CREPE 做 pitch。** Why: 2026 已被 RMVPE / SOME 取代，accuracy 落後。
- **不要試圖 fork UltraSinger 加 ruby。** Why: UltraStar 1.2 格式無 ruby 語法，且 USDX/Vocaluxe/UltraStar Play 都沒 MP4 export，要從根改，比 fork MID2BAR 更費工。
- **不要在 Mac 本機跑 WhisperX 主程式。** Why: MPS 壞掉，會 silent fallback 到 CPU，但其他 ops 又會 fail。改 mlx-whisper。

## 第一個 test case 怎麼選
- **Vocaloid 純假名歌詞**最安全（沒 gikun，沒漢字異讀問題）
- **使用者 已經很熟、知道原唱怎麼唱的歌**（音高 / 對齊出錯時抓得到）
- 4 分鐘以內（短一點 debug 快）

## 做事方式
- 每個里程碑結束跑一次 end-to-end，確認沒 regression
- 振假名 / mora 對齊這種「永遠不可能 100%」的步驟，準備 override JSON 機制比追求模型完美重要
- 中間檔（vocals.wav、melody.mid、ruby.lrc）每階段都存，cache 友善
- Snakemake `--rerun-triggers params input code` 開著，改 code 不會重跑分離（注意：Snakemake 9 要 space-separated，不是 comma）

## 環境隔離 — single source of truth
**「主 venv + per-stage subprocess venv」是唯一正典**。三個 venv：

| venv | 負責 | 為何分開 |
|---|---|---|
| `~/venvs/karaoke-jp/` | M1 separate, M4 render, click CLI | 主環境，torch 2.11+cu13 |
| `~/venvs/karaoke-jp-melody/` | M2 SOME inference | librosa<0.10 + numpy<2 跟主 venv 撞 |
| `~/venvs/karaoke-jp-lyrics/` | M3 ASR + tokenize + align | faster-whisper 要 CUDA12 cuBLAS shim |

**已撤掉** `envs/*.yaml`（給 Snakemake `--use-conda` 用的 spec），原因：never wired to actual rules，三套 isolation story 並存只會 drift。M6 真要批次跑時可以再加回去（Snakemake conda envs 替 venv），但現在先收斂。

`melody.py` 的 subprocess `env=` 只 passthrough whitelist 的變數（PATH / HOME / LANG / LD_LIBRARY_PATH / CUDA_*），**不抄整個 `os.environ`**，避免父 venv 的 PYTHONPATH/PYTHONHOME/VIRTUAL_ENV 漏進子環境。M3/M4 subprocess 走同 pattern。

## 跨專案 context
- 使用者 的 Mac：Apple Silicon
- SSH GPU 機：MeanAudio 訓練那台（具體機器名要實測時確認）
- 整個 hub：[Documents/CLAUDE.md](../../CLAUDE.md)
