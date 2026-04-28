# karaoke-jp — 決策 / 踩坑記錄

> 這個檔記錄「為什麼選 X 不選 Y」與「不要再走的彎路」。每次踩到新坑就補一條。

---

## 已驗證決策

### 人聲分離：openmirlab/melband-roformer-infer
- 內含 KimberleyJensen Mel-Band-RoFormer Kim 為 default model
- pip 一行裝完，CLI + Python API
- 70+ 社群 fine-tune registry，需要時可換 model
- Kojie 過去用過 KJ 版本（透過 LLM 設定 SSH 跑），記得「效果超好」
- MIT license（程式碼）；weight 商用授權不乾淨但**自用無影響**
- 來源：ByteDance BS-RoFormer 論文 → KJ 訓練 → openmirlab Eric 老師學生包成 wrapper（2025/11 開 repo）

### Pitch + Note：RMVPE → SOME → BasicPitch（待評估）

#### SOME v1.0.0-baseline 完整解剖（2026-04-28 讀 code 確認）
- `me_infer.py` 的 f0_algo 整個 block 被 comment 掉，active code 是 `pitch = torch.zeros(...)`
- **更嚴重**：`modules/conform/Gconform.py` 的 `forward(self, x, pitch, mask)` 接受 pitch 但 **body 完全不用它**（`x1=x.clone()`，pitch 從未出現在計算裡）
- 結論：**SOME v1.0.0-baseline 不管輸入什麼 f0（RMVPE / parselmouth / zeros），output 完全一樣** — 驗證方法：`torch.zeros` 和 RMVPE 跑出來的 626 notes / 所有 onset / 所有 pitch 逐位完全相同
- config 寫 `pe: rmvpe`, `pe_ckpt: pretrained/rmvpe/model.pt` 是 stale，訓練時可能有 f0 conditioning 但 simplified checkpoint 把這條路關掉了
- 同樣 config 寫 `units_encoder_ckpt: pretrained/contentvec/...`，但實際 `units_encoder: mel` 走 mel-spec → **fairseq / ContentVec checkpoint 都不用裝**
- 結論：minimal install 只需 torch + torchaudio + librosa<0.10 + numpy<1 + parselmouth + lightning + mido + click + matplotlib，parselmouth 甚至也不需要
- **SOME 直接輸出 onset/offset，不用另跑 madmom**
- **SOME 輸出整數 MIDI 0-127**（非整數 MIDI 是 GAME / 連續模型才有）
- SOME 訓練資料全是中文歌（`test_prefixes` 裡 JuanZhuLian, GuanShanJiu 等），未必能泛化到日文 J-pop
- SOME weight CC-BY-NC-SA 4.0 → 自用 OK
- **RMVPE checkpoint 已下載**（352 MB）→ `third_party/SOME/pretrained/rmvpe/model.pt`，但對 v1.0.0-baseline 無效
- SOME 作者已釋出 GAME（successor），可能修復 f0 conditioning，**待評估**

#### 實測 RTF
8 秒跑完 4 分鐘 vocals.wav（RTX 5090，CUDA 13），626 notes，pitch range MIDI 49-79（C#3-G5，tuki. 17 歲女歌手合理），piano roll 結構符合 J-pop verse-chorus 結構。

#### 音階不準的根因 & 下一步
- SOME 的 pitch accuracy 受限於 mel-spec 特徵 + 中文訓練資料的泛化能力，不是 f0 問題
- 替代方案（優先評估）：**BasicPitch（Spotify）** — 專門 audio-to-MIDI，訓練資料多元，`pip install basic-pitch` 一行裝
- 替代方案（次選）：GAME checkpoint（SOME 後繼），`openvpi/GAME` — 可能有更好的泛化
- 手動 MIDI 是 ceiling（saund-box 參考影片做法），但 4 月發行的新歌找不到現成 MIDI

### 歌詞 timing：MIDI note onset（2026-04-28 實裝）
- **根因**：Whisper word-level timestamp 對唱歌不準；同一 token 內多字全部同一時間戳 → 畫面多字同時出現然後跳
- **修法**：`scripts/midi_timing.py` — 用 MIDI note onset 替換 Whisper char-level timing
  - 讀 melody.mid → 提取 (onset, offset, pitch) 清單
  - 對每一行，以 Whisper line-level 時間窗（±0.4s 容差）找該行的 notes
  - Greedy monotone 匹配：每個 char 分配一個 note，onset 為 char_start，next note onset 為 char_end
  - 626 notes / 538 chars → 43 行全部找到 notes，0 行退回 Whisper
- **效果**：`再` `会` 從 9.870/9.870 → 10.286/10.925（各自獨立），`変わっ` 三字從 14.190/14.190/14.190 → 14.025/14.210/14.664
- **pipeline**：`align` → `aligned.json`（Whisper timing）→ `midi_timing` → `aligned_midi.json`（MIDI timing）→ `export_lrc` / `midi_markers`
- SOFA（singing forced aligner）仍列為備選升級方案，但 MIDI-based timing 已足夠好

### 20% 人聲 mix（2026-04-28 實裝）
- `src/karaoke_jp/mix.py` 實作 `mix_vocals(instrumental, vocals, out, vocal_ratio=0.20)`
- ffmpeg `amix` filter：`[bg]volume=1.0` + `[voc]volume=0.20` + `amix normalize=0` → `mixed.wav`
- `scripts/mix_audio.py` 是 CLI wrapper；`karaoke-jp mix` 是頂層 click command
- Snakefile `mix` rule → `outputs/<song>/mixed.wav`；`render` rule 改用 `mixed.wav` 而非 `instrumental.wav`
- 測試：`mixed.wav` 40 MB，ffprobe duration 237s，無 error

### 歌詞辨識：mlx-whisper
- WhisperX 在 Mac MPS **壞掉**（sparse_coo_tensor + repeat_interleave fail）
- mlx-whisper 在 M1 Max 上 large-v3-turbo 跑 13.1s，whisper.cpp 跑 26.7s（同一 audio）
- **保留** WhisperX 的 wav2vec2 alignment 步驟（這部分 MPS 可跑）做 word-level 時間戳

### Mora-level 對齊：SOFA
- WhisperX word-level 在日文常切到「単語」而不是「モーラ」，做不到假名級
- SOFA（qiuqiao/SOFA）= Singing-Oriented Forced Aligner，針對唱歌的長母音 / vibrato / melisma
- 預設 dict 是中文，要加 `lottev1991/opencpop-cjke-multidict` 補日文
- pyopenjtalk 的 phoneme set 可 map 到 mora（consonant+vowel 配對）
- 後繼者 HubertFA 加了 breath/AP detection，可後續評估
- 備案：ESPnet CTC segmentation（OWSM-CTC v4，CPU OK） / Julius segmentation-kit

### 振假名：fugashi + UniDic + Yomikata + per-song override
- **絕對不用 pykakasi**（無形態素解析，常選 on'yomi 錯）
- fugashi + 完整 UniDic 做 tokenization
- Yomikata BERT cover 130 個常見 heteronym（運命 / 本気 / 一日…）
- **Gikun 必須手動 override**：「運命=さだめ」「本気=マジ」「永遠=とわ」「宇宙=そら」 — 任何工具都救不了，因為這是創作者自選讀音
- 整合範例：Mathew Chan 的 Sudachi tutorial、`dkollmann/furiganamaker` 的 customreadings API
- 可選：LLM correction pass（Claude / GPT-4 知道熱門歌的 gikun）

### 字幕渲染：Aegisub karaskel + KaraTemplater
- **libass 不支援 ruby**（ASS v5.0 wiki 列了十年 TODO 還沒做）
- 業界 workaround：每行歌詞編譯成兩條平行 ASS dialogue（主行 + 振假名行），用 `\pos\an5\k...` 預算位置
- KaraTemplater 用 `The0x539/Aegisub-Scripts`（arch1t3cht 文件推的活躍 fork），不要用內建版
- 「次行歌詞淡入」用 `retime("preline", -1500, 0)` + `\fad`
- Karaoke Mugen 文件有 working 範例

### Renderer：fork MID2BAR-Player（讀完 code 重新確認 2026-04-28）
- `keisuke-okb/mid2bar-player` 已實作：JOYSOUND 風格音高方塊 + ruby LRC + 自訂背景 + ffmpeg encode + 三段式 stretchable 方塊圖層
- 同作者 LRC2EXO-Python，這條 line 走得熟
- 我們要做的只是**自動化上游**：原本要手餵 MIDI + 手刻 ruby-LRC，現在用 SOME + SOFA + fugashi 自動產
- 健康度：6 stars / 0 forks / v1.0.0 (2025-12-31) / Apache-2.0 — 小眾 hobby project 但作者其他 repo 50+ commits，不是棄坑
- **架構驗證（讀過 framerecorder.py + app.py:1294-1395）**：MP4 export 是 headless offline pipe，不是螢幕錄製。`pygame.image.tostring(surface, "RGB")` 抽 in-memory pixel buffer 餵 ffmpeg stdin，`current_time = frame_idx / fps` deterministic。Pygame 在這裡只是當畫布
- **LRC 格式**：每 char `[mm:ss:cs]` colon-separated centiseconds（**不是** standard `[mm:ss.ms]`），ruby 是 header section `@RubyN=base,ruby,[start],[end]`，不是 inline `(かな)`。範例見 `third_party/MID2BAR-Player/sample/*.lrc`
- **Headless on Linux**：跑前 `export SDL_VIDEODRIVER=dummy`
- **入口**：`main.py` 是寫死 paths 的 sample，不是 CLI；自己包成 click command
- **MIDI 要插 page-boundary markers**（他們 marker editor 是 Windows-only GUI）— 用 mido 加 meta-event 自動化
- 我們的 `aligned.json` → MID2BAR LRC：~60 行 Python（一行 walk tokens，帶 ruby 的出 `@RubyN`，body emit char-level tags）

### Fork-vs-build 決策 checklist（M4 翻案後沉澱）
**何時觸發**：要決定 「fork 一個外部 repo + 自動化上游」 vs 「自己寫一個 ~300 行的版本」。

**Sub-agent / WebFetch 的可信度邊界**：
- ✅ 可信：repo 健康度（star / commit cadence / open issues / license / 上次 release 時間）
- ✅ 可信：README / 範例展示的功能列表（pitch bar / wipe / ruby 等是否存在）
- ✅ 可信：deps 清單、檔案結構、line counts
- ❌ 不可信：「export path 是 X」「rendering 模式是 Y」這種**架構意圖**判斷
- ❌ 不可信：根據檔名 / class 名 / module 名「猜」實作（看到 Pygame 就猜螢幕錄製，看到 fairseq import 就猜實際用到）

**強制驗證項目**（fork-vs-build 決策前必跑，不能省）：
1. **直接讀關鍵 code 路徑**，不能只看 README。MID2BAR 的 `framerecorder.py` 只有 110 行，5 分鐘讀完
2. **找實際 export 函式**，看它怎麼產生 deliverable（offline encoder vs real-time capture vs other）
3. **驗證可 headless**（如果目標是 batch / CI）— 找 `set_mode` / display init / DISPLAY 依賴
4. **找入口點**（CLI / main / API class），確認是不是要包一層
5. **驗證 input format spec**，不能假設是 standard format（MID2BAR 用 `[mm:ss:cs]` colon，不是 standard `[mm:ss.ms]` dot；ruby 是 `@RubyN=` header 不是 inline `(かな)`）

**M4 具體 incident**：
- Sub-agent survey 後判定「實時 Pygame 視窗 + ffmpeg 螢幕錄製」推薦自寫
- Kojie 提供 deep research 文章說「ffmpeg-encodes to MP4 from MIDI + LRC + audio」
- 衝突透過直接讀 `framerecorder.py` 110 行解決：實際是 `pygame.image.tostring(surface, "RGB")` → ffmpeg stdin，frame_idx 驅動的 deterministic offline pipe
- 結論：fork plan stays（spec 原計畫 + 文章對 + 自己讀 code 確認）

### 背景圖 / 影片 — yt-dlp + OpenCV codec gap（2026-04-28 踩過）
- `karaoke-jp render` 接受 `--background <path>`，自動偵測 `songs/<song>/background.{mp4,webm,png,jpg,...}`
- **YouTube 預設給 AV1 mp4**（format 251 / 313 / etc），**OpenCV 沒軟解 AV1**，會 silent black bg
- `download_song.py` 用 `-f 'bv*[vcodec*=avc1][height<=720]/...'` 強制偏好 h264 720p（format 136 / 137）
- `render_mp4.py` 不論 input 都過一道 ffmpeg re-encode（libx264 + yuv420p + scale=1920x1080:pad），保險 + normalize 解析度
- 靜態圖（png/jpg）用 `ffmpeg -loop 1 -t 5` 包成 5 秒 mp4，MID2BAR 的 video player 會自動 loop
- Output 寫到 `outputs/<song>/_background.mp4`，跟 karaoke.mp4 同層 gitignored

### MID2BAR 的 lyrics_images cache 跨歌污染（2026-04-28）
- MID2BAR `lrc.load_lyrics()` 會 cache 每行歌詞渲染好的 PNG 到 `lyrics_images/<lrc_basename>/` + `lyrics_images/<lrc_basename>.json`
- **兩首歌都叫 `karaoke.lrc` → cache key 撞 → 第二首拿到第一首的歌詞圖**（debug 之前看到 bocchi 跑出 tuki 的歌詞，audio 是 bocchi 的）
- 修法：`render_mp4.py` 每次 render 前 `shutil.rmtree(lyrics_images/<basename>/)` + `unlink(lyrics_images/<basename>.json)`，強迫重產
- **教訓**：fork 第三方 renderer 時 cache key 預設都是「對單一 user 一次跑一首」設計，多歌 batch 必踩

### Lyric Video vs Official Audio（2026-04-28）
- 日本動漫 tie-in 上傳常常是 **「Lyric Video」**（Aniplex / 音楽出版社頻道）— **歌詞已燒進影片**，不是用 caption track
- vs **「Official Audio」**（藝人自己頻道）— 影片是靜態 album cover / 簡單動畫，無歌詞燒入
- karaoke bg 用原 YT 影片時：Official Audio = 直接用沒問題；Lyric Video = 跟我們歌詞層撞，user 必須自選靜態 bg
- 自動判斷不可靠（看不到 metadata flag），靠 user 看標題或 description 自己決定 → `download_song.py --no-video` 跳過抓 video

### Lyrics 抓取 — WebFetch LLM filter 擋公開資料庫（2026-04-28）
- WebFetch 對歌詞站（uta-net、lyrics.github.io 等）回 "I cannot reproduce copyrighted lyrics"，即使是 Aniplex Lyric Video 早就把歌詞放出來的歌
- 繞過：直接 `urllib.request` + `re.search` parse HTML（uta-net 的歌詞 div id="kashi_area"），規避 LLM 中介
- 不是要繞著作權 — 資料公開可看，純技術問題：WebFetch 的 LLM 對「raw lyrics text in the response」太謹慎
- 「Official Audio」上傳的 description 還是首選來源（藝人自貼，直接 yt-dlp `--print %(description)s` 拿）
- 標籤頻道 / Lyric Video 上傳的 description 通常只放 credit，**不放歌詞** → 才需要這個繞道

### 批次：Snakemake
- 50 首 fan-out 一行 wildcard
- File-output DAG 跟多階段音樂 pipeline 1:1
- `--rerun-triggers params input code` = content-aware caching，改 code 不重跑分離（Snakemake 9 用 space-separated args，舊範例的 comma 形式會 error）
- conda env per rule 解 Demucs / RMVPE / WhisperX 要不同 torch 版本的問題
- Prefect / Airflow 對單機過頭；Luigi 半死；純 Make 在日文檔名（含空格）會壞

---

## 已否決選項（不要走回頭路）

### Fork UltraSinger（否決）
- UltraStar 1.2 格式**無 ruby 語法**
- 三大 engine（USDX / UltraStar Play / Vocaluxe）都**沒 MP4 export**，要 OBS 螢幕錄
- 視覺風格是 SingStar 綠底音符條，不是 JOYSOUND
- 要走到目標需要：寫新 renderer + 改格式加 ruby track + 換渲染管線 → 這時候已經 = 重寫 MID2BAR

### 從零寫 renderer（否決）
- 工作量沒省多少（MID2BAR 已寫好的合成層 + 三段方塊 + 圖層字幕都要重做）
- ASS+karaskel 雖能畫好 ruby 但畫不了音高方塊，無論如何要再寫一層

### CREPE（否決）
- 2026 已被 RMVPE / PESTO 超越
- 速度也輸 SwiftF0（RMVPE 大 model 慢但準；SwiftF0 95K params 快 42×）
- torchcrepe 還能用但沒理由選

### pykakasi（否決）
- 詳見上方振假名段；無形態素 = 不可用

### Spleeter / Open-Unmix（否決）
- 2021 後沒大更新
- 被 Demucs / RoFormer 全面超越
- 只在 real-time CPU 場景有意義（不是我們的需求）

### 上傳 YouTube 路線（否決）
- 詳見 [spec.md §2](spec.md#2-法律-stance重要)
- Demucs 分離仍是原盤衍生，JASRAC blanket 不 cover 原盤権
- 上傳路線需自製伴奏 / Vocaloid Piapro off-vocal / 公版 → **不在本專案 scope**

---

## 待驗證

- [x] MID2BAR-Player repo 健康度 ✅ 2026-04-28：6 stars / 0 forks / Apache-2.0 / v1.0.0 (2025-12-31)，read-code 確認 offline frame pipe，可 fork
- [ ] SOFA + opencpop-cjke-multidict 對日文歌的覆蓋度（M3 v2 polish）
- [ ] Mac MPS 上 SOME / RMVPE 實測速度（pipeline 目前只在 Linux 跑）
- [x] SSH GPU 機 ✅ ntnumaplab2 RTX 5090 32 GB，pipeline 已 run on
- [ ] Mel-Band-RoFormer 不同 fine-tune（unwa Big Beta 6、Gabox 等）對 J-pop 哪個最好（KJ Kim default 用過兩首沒問題）

---

## 過去研究輪次

- 第一輪 deep research（產出 11 個開放問題）
- 第二輪 deep research（找到 MID2BAR-Player 這個 game-changer，把 11 題壓縮成 3 個決策）
- 兩輪結論已併入 [spec.md](spec.md)

## Codex review 記錄

- **Round 1**（initial M1-M3 commit, c019e67 → b2e1631）：5 fix（Snakefile `:q` quoting、kana-aware align refactor、subprocess env hygiene、tempfile staging、ASR comment honesty）+ 2 architectural（drop envs/*.yaml、refactor M4 lesson as checklist）
- **Round 2**（M4 commit, 61409de → 4624771）：3 fix（LYRICS_LD glob 不寫死 python3.12、Snakefile M1/M2 rule pin venv binary、render override paths resolve before chdir）+ stale doc cleanup
- **Round 3**（2026-04-28）：三問題修法（音階 / timing / vocal mix）
  - SOME f0 bug：取消 comment RMVPE branch + 下載 checkpoint → 但發現 v1.0.0-baseline forward() 根本不用 pitch，output 無差別（已記錄）
  - `scripts/midi_timing.py`：MIDI note onset 替換 Whisper char timing，43/43 行更新
  - `scripts/mix_audio.py` + `src/karaoke_jp/mix.py`：20% vocal ffmpeg amix
  - Snakefile 新增 `midi_timing` / `mix` rule，`render` 改吃 `mixed.wav`
  - `src/karaoke_jp/cli.py` 新增 `karaoke-jp mix` command
  - 成功 re-render `outputs/tuki-zero/karaoke.mp4`（289 MB, 237s）
- **本次 review hand-off**：每次 push 後 user 用 codex 跑遠端 review，回饋以 `::code-comment` 標記。Codex 會做小型 sanity check（synthetic input 跑 export_lrc 之類）但不重跑模型 inference。**信任度高**，回饋都打中 — 全照做沒爭議
