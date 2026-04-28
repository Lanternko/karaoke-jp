# karaoke-jp

## Why
日式卡拉 OK 影片自動生成器（JOYSOUND 風格：離散音高方塊 + 逐字歌詞 wipe + 振假名 + 自選背景）。**使用者 自用練唱**，不上傳。

## 狀態
**M0-M4 全 wire（2026-04-28）**，兩首歌端到端跑通：

| song | bg 來源 | mp4 路徑 |
|---|---|---|
| tuki-zero (`零-zero-`) | YT Official Audio = album art | `outputs/tuki-zero/karaoke.mp4` |
| bocchi-guitar (`ギターと孤独と蒼い惑星`) | YT Lyric Video MV（含燒入字，user accepted） | `outputs/bocchi-guitar/karaoke.mp4` |

兩首都 1080p60，duration 跟原曲對齊到 sample，pitch bars + per-char wipe + furigana 全部達成。

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
2. **整個 pipeline 在 Linux GPU 機跑**（RTX 5090, CUDA 13）。Mac MPS path 在 spec 裡規劃過但**從未實測**；如果 使用者 之後要在本機跑，需要重新驗證 mlx-whisper / demucs-mlx / pyannote MPS 行為。
3. **ASR 用 faster-whisper + lyrics initial_prompt** 偏置開頭幾個字，避免 quiet 段（如 verse 1 開頭）漏抓 + 幻覺成「ご視聴ありがとう」。Mora-level 對齊先用 char-level proportional split (Whisper word_timestamps 對 JP 直接 per-char)，SOFA 升級留 v2。
4. **個人使用，不上傳**。Why: 著作権法 30 条（私的使用）允許自用；上傳會撞原盤権 + Content ID。詳見 spec §2。
5. **不做連續 f0 曲線**（Vocal Pitch Monitor 風格已否決）。要的是離散音高方塊。

## NEVER
- **不要把專案產出上傳 YouTube / 公開分享。** Why: Demucs / RoFormer 分離出的伴奏仍是原盤的衍生，JASRAC blanket license 不 cover；Avex / SME / Universal 會 Content ID claim 或刪頻道。詳見 [spec.md §2](spec.md)。
- **不要用 pykakasi 做振假名。** Why: 沒形態素解析，常選錯讀音。改用 fugashi + UniDic + Yomikata。
- **不要用 CREPE 做 pitch。** Why: 2026 已被 RMVPE / SOME 取代，accuracy 落後。
- **不要試圖 fork UltraSinger 加 ruby。** Why: UltraStar 1.2 格式無 ruby 語法，且 USDX/Vocaluxe/UltraStar Play 都沒 MP4 export，要從根改，比 fork MID2BAR 更費工。
- **不要兩首歌都用 `karaoke.lrc` 當輸出名 — MID2BAR 用 `lyrics_images/<lrc_basename>/` 當 cache，會把 song A 的歌詞圖渲染到 song B 上。** `render_mp4.py` 已經在每次 render 前 wipe 該 cache dir，但若改 LRC 命名邏輯要記得重 walk 這條 cache invalidation。
- **不要直接拿 yt-dlp 預設下的 mp4 當 MID2BAR 背景。** YouTube 預設給 AV1，OpenCV 沒軟解，會 silent black bg。`render_mp4.py` 一律 ffmpeg re-encode 成 h264 yuv420p；`download_song.py` 也偏好 `vcodec*=avc1` 720p 避開 AV1。
- **「Lyric Video」型 YouTube 上傳（已燒入歌詞）不要當 karaoke bg**，會跟我們的歌詞層撞。`download_song.py --no-video` 跳過抓 video，user 自己提供靜態 bg。

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
**「主 venv + per-stage subprocess venv」是唯一正典**。**四個** venv：

| venv | 負責 | 為何分開 |
|---|---|---|
| `~/venvs/karaoke-jp/` | M0 download, M1 separate, M4 prep（lrc/marker），click CLI | 主環境，torch 2.11+cu13 |
| `~/venvs/karaoke-jp-melody/` | M2 SOME inference | librosa<0.10 + numpy<2 跟主 venv 撞 |
| `~/venvs/karaoke-jp-lyrics/` | M3 ASR + tokenize + align | faster-whisper 要 CUDA12 cuBLAS shim（pip 裝 nvidia-cublas-cu12 + LD_LIBRARY_PATH 指過去）|
| `~/venvs/karaoke-jp-render/` | M4 render（MID2BAR-Player）| Pygame + opencv-python，跟主 venv 的 click 衝得到 |

**已撤掉** `envs/*.yaml`（給 Snakemake `--use-conda` 用的 spec），原因：never wired to actual rules，三套 isolation story 並存只會 drift。M6 真要批次跑時可以再加回去（Snakemake conda envs 替 venv），但現在先收斂。

`melody.py` 的 subprocess `env=` 只 passthrough whitelist 的變數（PATH / HOME / LANG / LD_LIBRARY_PATH / CUDA_*），**不抄整個 `os.environ`**，避免父 venv 的 PYTHONPATH/PYTHONHOME/VIRTUAL_ENV 漏進子環境。M3/M4 subprocess 走同 pattern。

## 跨專案 context
- 使用者 的 Mac：Apple Silicon（**目前 pipeline 沒在 Mac 跑過**）
- 主 dev / runtime 機：`linux-gpu`（RTX 5090, 33.67 GB VRAM, MeanAudio 訓練那台）
- 整個 hub：[Documents/CLAUDE.md](../../CLAUDE.md)
- Repo: https://github.com/Lanternko/karaoke-jp
