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

### Pitch + Note：RMVPE → SOME
- RMVPE 2023 出，**2026 仍是 vocal pitch SOTA**（96.0% on MIR-1K，超 SwiftF0 的 95.0%）
- SOME（openvpi）原本宣稱 RMVPE backbone，**但 v1.0.0-baseline checkpoint 實測：me_infer.py 的 RMVPE 分支被註解掉**，inference 時是 `get_pitch_parselmouth`（praat-parselmouth）抽 f0；`modules.rmvpe` 只 import `MelSpectrogram` 工具類別。config 寫 `pe: rmvpe` 是 stale 設定。
- 同樣 config 寫 `units_encoder_ckpt: pretrained/contentvec/...`，但實際 `units_encoder: mel` 走 mel-spec → **fairseq / ContentVec checkpoint 都不用裝**
- 結論：minimal install runtime 只需 torch + torchaudio + librosa<0.10 + numpy<1 + parselmouth + lightning + mido + click + matplotlib，沒有 fairseq 折磨
- **SOME 直接輸出 onset/offset，不用另跑 madmom**
- SOME 輸出非整數 MIDI（實測 v1.0.0-baseline 還是整數 0-127，非整數 MIDI 是 GAME / 連續模型才有），保留浮點可畫 vibrato 那條 plan 要等之後升級
- v1.0.0-baseline zip 名字 `0119_continuous128_5spk.zip` 解出來資料夾叫 `0119_continuous256_5spk/`（128 vs 256 不一致，是 release 命名 typo）
- ckpt 檔名是 `model_ckpt_steps_100000_simplified.ckpt`，配套 `config.yaml` 同層
- SOME weight CC-BY-NC-SA 4.0 → 自用 OK
- SOME 作者已釋出 GAME（successor），值得追蹤但先用 SOME

實測 RTF：8 秒跑完 4 分鐘 vocals.wav（RTX 5090，CUDA 13），626 notes，pitch range MIDI 49-79（C#3-G5，tuki. 17 歲女歌手合理），piano roll 結構符合 J-pop verse-chorus 結構。

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

### M4 翻案教訓（記下來避免重蹈覆轍）
- 中間 sub-agent survey MID2BAR-Player 後判定「實時 Pygame 視窗 + ffmpeg 螢幕錄製」推薦自寫 300 行
- Kojie 提供之前研究文章說「ffmpeg-encodes to MP4 from melody MIDI + ruby-LRC + audio」
- 衝突點透過直接讀 `framerecorder.py` 110 行解決：sub-agent 看到 Pygame 就誤判，沒讀進 push_frame 的實作
- **教訓**：fork-vs-build 這種方向性決策一定要直接讀 code 驗證關鍵假設，不能只 WebFetch README + 看 stars。Sub-agent 對「結構」的判斷可信，對「架構意圖」的判斷不可信

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

## 待驗證（M1 進場時要實測）

- [ ] MID2BAR-Player repo 健康度（最後 commit、issue、API 穩定性）
- [ ] SOFA + opencpop-cjke-multidict 對日文歌的覆蓋度
- [ ] Mac MPS 上 SOME / RMVPE 實測速度
- [ ] SSH GPU 機是否還能用、權限、磁碟
- [ ] Mel-Band-RoFormer 不同 fine-tune（unwa Big Beta 6、Gabox 等）對 J-pop 哪個最好

---

## 過去研究輪次

- 第一輪 deep research（產出 11 個開放問題）
- 第二輪 deep research（找到 MID2BAR-Player 這個 game-changer，把 11 題壓縮成 3 個決策）
- 兩輪結論已併入 [spec.md](spec.md)
