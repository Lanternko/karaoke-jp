# karaoke-jp

## Why
日式卡拉 OK 影片自動生成器（JOYSOUND 風格：離散音高方塊 + 逐字歌詞 wipe + 振假名 + 自選背景）。**自用練唱**，不上傳。

## 狀態
**M0-M4 全 wire（2026-04-28）**；**timing pivot 到 mora→note alignment（2026-05-01）**；**M8 Gradio GUI 完成（2026-05-02，commit `d1d6157`）**。三首歌端到端跑通：

| song | bg 來源 | mp4 路徑 |
|---|---|---|
| tuki-zero (`零-zero-`) | YT Official Audio = album art | `outputs/tuki-zero/karaoke.mp4` |
| bocchi-guitar (`ギターと孤独と蒼い惑星`) | YT Lyric Video MV（含燒入字，user accepted） | `outputs/bocchi-guitar/karaoke.mp4` |
| chidori (`ヨルシカ — 千鳥`) | 靜態 bg（Lyric Video 有燒入字故不用） | `outputs/chidori/karaoke.mp4` |

三首都 1080p60，duration 跟原曲對齊到 sample，pitch bars + per-char wipe + furigana 全部達成。

下一步：**M6 Snakemake 批次** / **M7 Yomikata + override JSON**（M5 已完成 20% vocal mix，可選多音量版）。

**音高里程碑（2026-06-10）**：樂譜層音高推論整輪完成 — chidori 全驗證 gold（sheet+人耳）、
雙歌 benchmark、canonical 雙鏈（classic scorefix + GAME union）。chidori note-level
52.6%→**72.8%**。完整方法論與計分板見 [docs/pitch-benchmark.md](docs/pitch-benchmark.md)；
不可再生 gold 已存 tracked 的 `gold/`。
**顯示系統 v8（2026-06-11）**：固定網格 + 時間 warp（[docs/display-grid.md](docs/display-grid.md)），
成品 `karaoke...gameunion_v8grid.mp4` 待 Kojek 最終驗收後切 canonical；
display 整輪尚未 commit — 接手先讀 [docs/handoff-2026-06-11.md](docs/handoff-2026-06-11.md)。

## M8 GUI（已實作 2026-05-02）
**目標**：clone repo → 4 venv setup → 一句指令啟 GUI → 4 分鐘歌約 5–10 分鐘出 mp4。本機 only（預設 `127.0.0.1:7860`，**不開 share**；`--host 0.0.0.0` 才接外網）。

**啟動**：`~/venvs/karaoke-jp/bin/karaoke-jp gui [--host 127.0.0.1] [--port 7860]`
> 實作落在 `src/karaoke_jp/gui.py`（package module，**不是** standalone `scripts/gui.py`），透過 Click `gui` 子命令 launch；裝 GUI 依賴：`pip install -e '.[batch,gui]'`（`gradio>=5,<6`）。

**5 個 Gradio 欄位 → song dir**：
1. YouTube URL **或** MP4 上傳（二選一）→ 走 `download_song.py` 或 `ffmpeg -i upload.mp4 -vn songs/<id>/source.wav`（mp4 不在 `Snakefile.source_for()` 接受清單，必須 ffmpeg 抽 wav）
2. Lyrics 純文字 paste → `songs/<id>/lyrics.txt`（pipeline 會用它當 alignment ground truth，**不靠 Whisper transcribe 的文字**）
3. 人聲比例 slider 0–100 → export `VOCAL_RATIO=<x/100>` 給 Snakemake
4. 背景模式 radio：原影片（mp4 copy / yt-dlp 拉 video）/ 純黑（`ffmpeg -f lavfi -i color=c=black:s=1920x1080:r=60 -t <duration>`）/ MID2BAR 預設藍漸層（不放 background.* 即可）

**song-id 規則**：YouTube → yt-dlp title slug；MP4 上傳 → `<filename-slug>-<sha1[:8]>`（避開 `lyrics_images/` cache 撞名雷）

**Snakefile 接點**：`Snakefile:30` `VOCAL_RATIO = float(os.environ.get("VOCAL_RATIO", "0.35"))`，validation 強制落在 `[0, 1]`。其他靠 subprocess，沒有額外 Snakemake 改動。

**不在 GUI scope**：LLM 修歌詞 / LLM 推 ruby / Whisper transcribe-only 模式 / 多音量版輸出 / share=True / 多歌並行。詳見 NEVER。

## 文件套件
- [spec.md](spec.md) — 完整技術規格、pipeline、里程碑
- [MEMORY.md](MEMORY.md) — 關鍵決策、踩坑、為什麼選 X 不選 Y
- [docs/pitch-benchmark.md](docs/pitch-benchmark.md) — 音高 gold 方法論、benchmark、canonical 鏈
- [docs/display-grid.md](docs/display-grid.md) — 標準化 bar 顯示系統（grid + 時間 warp）
- [docs/handoff-2026-06-11.md](docs/handoff-2026-06-11.md) — 最新交接快照（未 commit 清單、待辦）
- 本檔（CLAUDE.md）— 工作指南

## 關鍵決策（不要再爭論）
1. **Fork MID2BAR-Player**，不 fork UltraSinger，不從零寫 renderer。Why: MID2BAR 已實作 JOYSOUND 風格 ruby + 音高方塊 + ffmpeg encode。**驗證細節（讀過 code 確認）**：
   - `framerecorder.py` = headless offline frame pipe（`pygame.image.tostring(surface, "RGB")` → ffmpeg stdin），**不是螢幕錄製**。Recording 模式 `current_time = frame_idx / fps` deterministic
   - LRC 格式：`[mm:ss:cs]`（colon 不是 dot，centiseconds 不是 ms），ruby 是 header `@RubyN=base,ruby,start,end`，不是 inline `(かな)`
   - Linux headless：跑前設 `SDL_VIDEODRIVER=dummy`
   - 入口：`main.py` 是 hard-coded paths sample，要包成 CLI
2. **整個 pipeline 在 Linux GPU 機跑**（RTX 5090, CUDA 13）。Mac MPS path 在 spec 裡規劃過但**從未實測**；如果之後要在本機跑，需要重新驗證 mlx-whisper / demucs-mlx / pyannote MPS 行為。
3. **ASR 用 faster-whisper + lyrics initial_prompt** 偏置開頭幾個字，避免 quiet 段（如 verse 1 開頭）漏抓 + 幻覺成「ご視聴ありがとう」。**Timing 用 mora→note alignment（`scripts/midi_timing.py` `--mode mora` 預設，2026-05-01 起）**：每個 token 的 reading 展成 mora 序列，per-line bounded greedy-monotone 配對 MIDI notes。kanji 詞（如 `再会` 4 morae）會吃到 4 個 notes 而不是 char-mode 的 2 個，sustain 尾段抓得到。Whisper char timing 只當 proximity hint。`--mode char` 是 legacy fallback。SOFA phoneme-level 升級留 v2（mora→note 視覺上已比 char-level 明顯準）。

   **Pitch backend ≠ timing backend**：`midi_timing.py` 吃的 MIDI 用 rmvpe（segment 數接近 mora 數），pitch bar 顯示用 cectc。同首歌兩個 MIDI 共存。
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
- **MID2BAR `BarCountEntry` / `AnimationEntry` 是 frozen dataclass，app.py 用 `["key"]` subscript 存取 → `TypeError: 'BarCountEntry' object is not subscriptable`，每 frame 都觸發，被 `draw()` 的 try/except 吞掉，導致 `draw_lyrics()` 永遠不執行，歌詞全黑。** `settings_schema.py` gitignored 不能直接改，修法是在 `render_mp4.py` import app 之前 monkey-patch `__getitem__` 到兩個 class：`_cls.__getitem__ = lambda self, key: getattr(self, key)`。任何新版 MID2BAR 如果歌詞突然消失，先查這條。
- **不要對整個 token 發 `@Ruby`** — fugashi 把「ぶつけ合っ」當一個動詞 token (reading「ぶつけあっ」)，naive 寫法 `@Ruby=ぶつけ合っ,ぶつけあっ` 會在 hiragana 「ぶつけ」上面也標假名。永遠走 `karaoke_jp.lrc_export.split_furigana()` 拆 kanji-run / kana-run，**只給 kanji-run 發 ruby**。
- **不要用 `app.particles = []` 關 MID2BAR 粒子。** `update_particles()` 每 frame 把三個 list 重新 reassign 成普通 list，patch 只活一個 frame。要 patch 的是 method：`app._update_particle_list = lambda particle_list, screen: []`。
- **不要為了通過 MID2BAR 的 ruby time 檢查設 char.end = next_char.start。** MID2BAR `apply_rubies_to_result` 用 `time_end ≤ ruby.end` 嚴格 contain，`time_end` 是 body 內 lyric 之後的下一個 time tag。@Ruby end 必須涵蓋這個 next-body-tag 時間（lookup 到下一 char.start，不是用 segment 自己的 char.end，因為 midi_timing 把 char.end 砍到 note_off 留了吐氣 gap）。
- **GUI 不要引入任何 LLM 步驟**（修歌詞、推 ruby、補 timing、整理 prompt）。Why: user 反映「成果不穩定，且通常要付費 API 才有像樣效果」。既有 fugashi+UniDic+forced alignment 已夠用；本機模型（demucs/roformer/whisper/cectc）不算 LLM，那些保留無妨。
- **不要對 YT karaoke guide 調參或把它當 gold。** 實測它對人耳/sheet 驗證 gold 只有 76.6% exact（F#4×44、B3×22 等家族性錯誤）。也**不要把「guide 與 F0 tracker 一致」當獨立證據** — 兩者被同一種演唱偏差（唱平/しゃくり）同向帶偏，B3 家族 22 顆因此誤判過。信任層級：鋼琴譜/人耳 >> guide ≈ F0。詳見 docs/pitch-benchmark.md。
- **不要把 GAME 直推原始混音**（chidori 實測 exact .48、八度錯 6.4%，README 的伴奏穩健性宣稱對密伴奏不成立）；**不要對 GAME 輸出套 refine-boundaries / absorb-shakuri**（實測有害 — 它的天然音符邊界比 mora grid 好）；GAME 的 align 模式不要當主旋律來源（強制 mora 切分傷邊界）。
- **不要重寫 user-lyrics → alignment ground truth 的 glue。已經 wired**：`scripts/run_asr.py --lyrics lyrics.txt` 把開頭當 initial_prompt 偏置 ASR；`scripts/align_lyrics.py` 跑 NW kana alignment 後輸出 `aligned.json`，**文字以 `lyrics.txt` 為準、timing 用 Whisper char ts 後續被 `midi_timing.py` 替換**。GUI 只需要把 user paste 寫進 `songs/<id>/lyrics.txt` 即可。

## 第一個 test case 怎麼選
- **Vocaloid 純假名歌詞**最安全（沒 gikun，沒漢字異讀問題）
- **自己已經很熟、知道原唱怎麼唱的歌**（音高 / 對齊出錯時抓得到）
- 4 分鐘以內（短一點 debug 快）

## 做事方式
- 每個里程碑結束跑一次 end-to-end，確認沒 regression
- 振假名 / mora 對齊這種「永遠不可能 100%」的步驟，準備 override JSON 機制比追求模型完美重要
- 中間檔（vocals.wav、melody.mid、ruby.lrc）每階段都存，cache 友善
- Snakemake **永遠帶 `--rerun-triggers mtime`**（單純看時間戳）。Why: 2026-05-01 起 outputs/ 有部分檔案是手跑產生（沒寫 `.snakemake/metadata/` provenance hash），default trigger set 會以「missing metadata」為由全 rebuild。code 改動的偵測自己用 git，不靠 Snakemake hash。注意：Snakemake 9 trigger 名要 space-separated，不是 comma

## 環境隔離 — single source of truth
**「主 venv + per-stage subprocess venv」是唯一正典**。**五個** venv：

| venv | 負責 | 為何分開 |
|---|---|---|
| `~/venvs/karaoke-jp/` | M0 download, M1 separate, M4 prep（lrc/marker），click CLI | 主環境，torch 2.11+cu13 |
| `~/venvs/karaoke-jp-melody/` | M2 SOME inference | librosa<0.10 + numpy<2 跟主 venv 撞 |
| `~/venvs/karaoke-jp-lyrics/` | M3 ASR + tokenize + align | faster-whisper 要 CUDA12 cuBLAS shim（pip 裝 nvidia-cublas-cu12 + LD_LIBRARY_PATH 指過去）|
| `~/venvs/karaoke-jp-render/` | M4 render（MID2BAR-Player）| Pygame + opencv-python，跟主 venv 的 click 衝得到 |
| `~/venvs/karaoke-jp-game/` | GAME note 轉譜（third_party/GAME）| **RTX 5090 (sm_120) 必須 torch cu129 wheel**，cu126 直接 CUDA kernel error |

**已撤掉** `envs/*.yaml`（給 Snakemake `--use-conda` 用的 spec），原因：never wired to actual rules，三套 isolation story 並存只會 drift。M6 真要批次跑時可以再加回去（Snakemake conda envs 替 venv），但現在先收斂。

`melody.py` 的 subprocess `env=` 只 passthrough whitelist 的變數（PATH / HOME / LANG / LD_LIBRARY_PATH / CUDA_*），**不抄整個 `os.environ`**，避免父 venv 的 PYTHONPATH/PYTHONHOME/VIRTUAL_ENV 漏進子環境。M3/M4 subprocess 走同 pattern。

## 跨專案 context
- 本機 Mac：Apple Silicon（**目前 pipeline 沒在 Mac 跑過**）
- 主 dev / runtime 機：Linux GPU（RTX 5090, 33.67 GB VRAM）
- Repo: https://github.com/Lanternko/karaoke-jp
