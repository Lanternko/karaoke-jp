# UltraSinger × COnPOff+L — 首次評測計畫（2026-07-02）

**目標**：拿 COnPOff+L 對 UltraSinger（唯一生產完整卡拉OK artifact 的開源工具，零已發表評測）
做第一次量化評測，Kiritan N=50 協定，與 GAME×MMS_FA（.408）同表比較。

## 已確認的地基（不用重新偵察）

- Harness：`benchmarks/kiritan/conpoff_l.py` — GT=`gt_timefix.json`（{song: [[on,off,midi],...]}，song key "01".."50"）；
  GT morae 來自 `~/side_projects/kiritan/kiritan_singing/mono_label/*.lab`（phone→mora 由 `group_morae()` 組）。
- Parser：`scripts/parse_ultrastar.py` 已實作 UltraStar txt→notes（beat→秒 = GAP/1000 + beat·60/(BPM·4)；pitch 0 = C4 = MIDI 60；freestyle/rap 無 pitch）。直接 reuse，不要重寫。
- 音檔：`~/side_projects/kiritan/kiritan_singing/wav/{01..50}.wav`，**96kHz mono**，每首 ~2–5 分鐘（01 = 272s）。清唱 a cappella。
- Python 3.10：`~/.local/bin/python3.10`（UltraSinger 要求 3.10）。
- GPU：RTX 5090 **sm_120** — 舊 torch cu126 wheel 會直接 CUDA kernel error，必須 cu128/cu129 系（前例：`~/venvs/karaoke-jp-game`）。
- 對照數字（RESULTS.md §COnPOff+L）：GAME×MMS_FA COn .860 / COnP .643 / COnPOff .499 / **+L .408**、P(L|match) 81.6%。

## 方法學定調（寫 RESULTS 時必須帶）

UltraSinger 是 **lyrics-unknown** 全自動系統（Whisper 自聽歌詞），我們的 MMS 行是
**lyrics-known** forced alignment。UltraSinger 的 L 稅 = 聽錯字 + 時間歸屬錯 兩者之和，
設定天生較難 — 這是它作為 full-stack 系統的誠實讀法，不是不公平，但表格註記必寫。
副指標補償：除 +L 外一定回報 COn/COnP/COnPOff（純 note 軸，與設定無關、完全公平）。

## 步驟

### Phase 0 — 環境
1. Clone `https://github.com/rakuri255/UltraSinger` 到 `third_party/UltraSinger`（karaoke-jp repo 內慣例位置）。
2. `~/.local/bin/python3.10 -m venv ~/venvs/ultrasinger`，照官方 README 裝依賴。
3. **sm_120 檢查**：裝完先 `python -c "import torch; print(torch.__version__, torch.cuda.is_available()); torch.zeros(1).cuda()"`。
   若 kernel error → 在 venv 內升 torch 到 cu128/cu129 wheel（參考 karaoke-jp-game 前例），再驗依賴沒被拉壞。
   GPU 救不動 → 各元件 fallback CPU（UltraSinger 有 force cpu 類 flag，讀它的 README/--help），先量 1 首耗時再決定 N。
4. 模型下載需網路（whisper large / demucs / crepe），會吃幾 GB — 放預設 cache 即可（/ 還有 235G）。

### Phase 1 — 輸入準備
96kHz mono 可能超出部分元件測試範圍：先 ffmpeg 轉 44.1kHz 16-bit 到
`/tmp/claude-1005/-home-kojiek-side-projects/ce d5e44d…/scratchpad` 下（實際 scratchpad 路徑見環境；中繼音檔不進 repo）。
UltraSinger 參數：語言強制日文（`--language ja` 類 flag，讀 --help 確認拼法）、
輸出到 `benchmarks/kiritan/ultrasinger/out/<song>/`。分離（demucs）對清唱是 no-op 級，可留預設；若有 skip-separation flag 可 A/B 一首確認無差後擇快者。

### Phase 2 — 冒煙測試（song 01）
1. 跑通一首，找到輸出的 UltraStar .txt。
2. `parse_ultrastar.py` 解析 → 檢查：note 數 vs GT（`len(gt["01"])`）、onset 範圍落在 [0, 272s]、
   **pitch 八度校驗**：est MIDI median vs GT MIDI median，若差 ≈12 的倍數 = 八度 convention 錯，回頭查 parser 假設 vs UltraSinger 實際輸出。
3. 記錄單首耗時 → 估 50 首總時。>8h 則先跑 N=10 出初步數字並在 RESULTS 標 preliminary。

### Phase 3 — 批次 50 首
逐首跑（失敗的記下、跳過、最後回報清單），log 存 `benchmarks/kiritan/ultrasinger/batch.log`。

### Phase 4 — 轉 harness 格式
1. notes：`ultrasinger_pred.json` = {song: [[on, off, midi], ...]}（只取 pitched notes）。
2. **L 軸 est morae**：`ultrasinger_morae.json` = {song: [[onset, romaji_mora_label], ...]}，每個 note 一筆，
   label = 該 note 的 UltraStar syllable 文字的**第一個 mora** 的 romaji。
   轉換鏈：syllable 文字（Whisper 出的，可能含漢字）→ 讀音 kana（**fugashi+UniDic**，
   絕不用 pykakasi — repo NEVER 條款）→ kana→phone 用 `~/side_projects/kiritan/kiritan_singing/japanese.table`
   （kiritan 官方 kana→phoneme 表，保證與 GT mono_label 同一套 phone 拼法）→ 依 conpoff_l.py 的
   mora 規則組 label（子音+母音串接、N standalone、cl 併入下一 mora）。
   fugashi 在 `~/venvs/karaoke-jp-lyrics`（tokenize 鏈用過）。
3. 邊角：無法轉換的 syllable（空白、記號、非日文）label 記 "?"（自動判 L fail，如實計稅）。

### Phase 5 — 評測
新腳本 `benchmarks/kiritan/ultrasinger_eval.py`：import conpoff_l 的 `load_notes/group_morae/read_lab/eval_song/match_notes/f1/bootstrap_diff`（不要複製貼上邏輯）。
- UltraSinger 的 est morae 直接用 ultrasinger_morae.json 的 (onset,label) list 餵 `attribute()` 同一套機制
  （每 note 一筆 → attribution 自然回到 note 自己的 syllable，正是卡拉OK「對的 bar 對的字」讀法）。
- 輸出：UltraSinger 的 COn/COnP/COnPOff/COnPOff+L/P(L|match) + 與 GAME×MMS_FA 的 paired bootstrap（同 50 首、2000 iters、seed 7）。
- Sanity：先用 GAME×MMS_FA 重跑一次確認與 conpoff_l_results.json 數字一致（harness 沒被改壞）。

### Phase 6 — 寫入與收尾
1. RESULTS.md 加 §UltraSinger：表格（與現有 6 行同格式加 UltraSinger 行）、協定紀錄
   （UltraSinger commit hash、flags、whisper/crepe 模型版本、單首均耗時）、
   **lyrics-known vs unknown 不對稱註記**、失敗清單。
2. `docs/holistic-benchmark-design.md` §7.3 補一行「已執行，數字見 kiritan/RESULTS.md」。
3. git add（新腳本 + RESULTS + PLAN + 兩個 pred json；**out/ 原始輸出和音檔不進 repo**，加 .gitignore）、
   commit（訊息照 repo 風格，繁中）。不 push。

## 風險與後備
- **torch/sm_120 依賴地獄**：最可能卡點。優先序 = GPU cu129 修 → 元件級 CPU fallback → 全 CPU N=10 preliminary。
- **UltraSinger 對清唱輸入崩潰**（假設混音）：試 skip-separation flag；還不行就把清唱混一條 -30dB 粉噪的伴奏軌騙過它（記入協定註記）。
- **Whisper 日文 syllable 切分怪**（UltraStar hyphenation 對日文未必成熟）：如實評，這正是「第一次評測」要暴露的東西；只在 parser 層面確保不 crash。
