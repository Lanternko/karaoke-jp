# Holistic Singing Benchmark — 音高辨識 × 歌詞對齊 統一評測設計

> 2026-06-22 起草。動機：把 note transcription（音符轉譜）與 lyric alignment（歌詞對齊）
> 合成一張表，跨多個歌唱資料集做系統間比較。

## 0. 核心問題：為什麼兩軸不能混成一欄

| | 音高辨識 (Note Transcription) | 歌詞對齊 (Lyric Alignment) |
|---|---|---|
| 問的問題 | 唱了幾個音？每個音的 onset/pitch/offset 是什麼？ | 已知歌詞 N 個字/音素，每個的時間戳是什麼？ |
| 數量關係 | **允許不等**（預測 99 個 vs GT 100 個是常態） | **必須等長**（你知道有幾個字，只預測時間） |
| 配對方式 | **greedy matching**（onset 容差內的最近配對） | **1:1 index pairing**（第 i 個預測對第 i 個 GT） |
| 主要指標 | P/R/F1（COn, COnP, COnPOff） | MAE, median AE, hit% |
| 指標庫 | `mir_eval.transcription` | boundary MAE / PCO@threshold |
| 「99 vs 100」時 | P=95/99, R=95/100, F1=95.5% — 自然處理 | 報錯或先 edit-distance 對齊再算 — 不該發生 |

**設計鐵律：兩個 column group 永不混欄。** 同一個系統填兩半 = 兩套獨立的 eval pipeline
在同一份音檔上跑；數字語意不同，放在一起是為了 **比較投入/產出的完整性**，不是為了合併成一個分數。

## 1. 資料集 × GT 能力矩陣

| Dataset | Lang | N | 條件 | 有 note GT | 有 phone GT | 有 word GT | 有 line GT | 在 repo |
|---|---|---|---|---|---|---|---|---|
| **Kiritan** | ja | 50 | 清唱 | ✓ `midi_label` | ✓ `mono_label` | ✗ | ✗ | ✓ 已跑 |
| **Itako** | ja | 50 | 清唱 | ✓ `midi_label` | ✓ `mono_label` | ✗ | ✗ | ✓ 已跑 |
| **MIR-ST500** | zh | 100 | 分離 | ✓ note annotation | ✗ | ✗ | ✗ | ✓ 已跑 |
| JamendoLyrics | en+ | 20 | 混合 | ✗ | ✗ | ✓ word timing | ✗ | ✓ 已跑 |
| 自建 gold | ja | 4 | 分離 | ✗ | ✗ | ✗ | ✓ 人耳行 | ✓ 已跑 |
| Opencpop | zh | 100 | 清唱 | ✓ note+phoneme | ✓ phoneme timing | ✗ | ✗ | **✗ 未取得** |

**Kiritan / Itako = 唯一雙軸資料集**（同時有 midi_label 與 mono_label）→ 是 holistic 表的基石。
Opencpop 加入後可覆蓋中文雙軸。

## 2. 系統 × 輸出能力

| System | 輸出什麼 | 填哪半 | 條件 |
|---|---|---|---|
| GAME | 音符 (onset, offset, MIDI pitch) | Transcription | zero-shot |
| CE+CTC | 音符 | Transcription | in-domain (MIR-ST500) / zero-shot (Kiritan/Itako) |
| ROSVOT | 音符 | Transcription | zero-shot |
| MMS_FA | 音素時間戳 | Alignment (phone) | zero-shot |
| MMS-JA | 音素/行時間戳 | Alignment (phone/line) | zero-shot (微調 ckpt) |
| SOFA | 音素時間戳 | Alignment (phone) | ⚠ Kiritan/Itako 洩題 |
| classic (Whisper) | 行時間戳 | Alignment (line) | legacy |
| **Pipeline (GAME+MMS)** | **兩者** | **兩半都填** | **未做 — 論文 contribution** |

## 3. 統一表結構

```
Dataset  │ System   │ Cond.      │ ── Note Transcription ──────── │ ── Lyric Alignment ────────────────
         │          │            │ COn    COnP   COnPOff  (F1)    │ Gran.  MAE     median  hit%   thr
─────────┼──────────┼────────────┼────────────────────────────────┼────────────────────────────────────
Kiritan  │ GAME     │ zero-shot  │ .862   .644   .502             │  —      —       —       —
Kiritan  │ CE+CTC   │ zero-shot  │ .860   .652   .492             │  —      —       —       —
Kiritan  │ ROSVOT   │ zero-shot  │ .414   .290   .190             │  —      —       —       —
Kiritan  │ MMS_FA   │ zero-shot  │  —      —      —               │ phone  .112s   .043s   56.0%  ≤50ms
Kiritan  │ SOFA ⚠   │ contam.    │  —      —      —               │ phone  .018s   .007s   89.3%  ≤50ms
─────────┼──────────┼────────────┼────────────────────────────────┼────────────────────────────────────
Itako    │ GAME     │ zero-shot  │ .824   .494   .400             │  —      —       —       —
Itako    │ CE+CTC   │ zero-shot  │ .828   .509   .374             │  —      —       —       —
Itako    │ MMS_FA   │ zero-shot  │  —      —      —               │ phone  .078s   .048s   48.0%  ≤50ms
─────────┼──────────┼────────────┼────────────────────────────────┼────────────────────────────────────
ST500    │ GAME     │ zero-shot  │ .732   .655   .411             │  —(no phone GT)
ST500    │ CE+CTC   │ in-domain  │ .779   .728   .554             │  —
─────────┼──────────┼────────────┼────────────────────────────────┼────────────────────────────────────
Jamendo  │ MMS_FA   │ zero-shot  │  —(no note GT)                 │ word   .233s    —      94.5%  PCO@.3
─────────┼──────────┼────────────┼────────────────────────────────┼────────────────────────────────────
自建gold │ MMS-JA   │ canonical  │  —                              │ line   .037s   .031s   100%   ≤250ms
自建gold │ SOFA+isl │ zero-shot  │  —                              │ line   .177s   .046s   83%    ≤250ms
```

### 3.1 欄位語意

**Transcription 半（左）：**
- COn F1 = onset-only F1（50ms 容差，greedy matching，mir_eval.transcription）
- COnP F1 = onset+pitch F1（50ms onset, 50 cents pitch）
- COnPOff F1 = onset+pitch+offset F1（加 offset max(50ms, 0.2×dur)）
- 報 F1 不報 P/R = 文獻慣例（P/R 在逐首細表裡給）

**Alignment 半（右）：**
- Gran. = 粒度（phone / word / line）— **不同粒度的 hit% 不可比**
- MAE = 平均絕對邊界誤差（秒）
- median = 中位數絕對邊界誤差
- hit% = 落在門檻內的比例
- thr = 門檻值（phone ≤50ms, word PCO@0.3s, line ≤250ms）

### 3.2 填表規則

1. 一個 system 只填它的輸出能力對應的半邊（notes → 左, phones/words/lines → 右）
2. 資料集沒有該 GT → 整半打 `—`
3. ⚠ = 洩題/汙染警告（SOFA 在 Kiritan/Itako 上屬 in-distribution,不作可比）
4. 同一 system 填兩半 = 跑了兩條 pipeline,在 Cond. 欄標明

## 4. 指標定義速查

### 4.1 PCO / PCS / AAE（你問的「PCOS」）

都是 `mir_eval.alignment` 三件套（JamendoLyrics 文獻標準）：

| 指標 | 全名 | 算法 | 語意 |
|---|---|---|---|
| PCO@τ | Percentage of Correct Onsets | 預測 onset 落在 GT onset ±τ 內的比例 | 「起點對了幾%」 |
| PCS | Percentage of Correct Segments | 預測 segment 與 GT segment 的時間重疊率 | 「時間佔了幾%」 |
| AAE | Average Absolute Error | 每個 onset 誤差絕對值的平均 | 「平均偏多少秒」 |

你問的「多少百分比重合」= PCS；「起點對了幾%」= PCO。兩個不同的「百分比」。

### 4.2 為什麼我們的 alignment eval 沒用 PCO/PCS

`eval_alignment.py` 和 `bench_aligners.py` 用的是 **MAE + median + IoU**（產品導向：
MAE 看平均、median 看中心、IoU 看整行佔用、invariants 看渲染品質）。
PCO/PCS/AAE 是文獻標準（JamendoLyrics 用這套）。

holistic 表的 alignment 半用 **MAE + median + hit%**：
- phone/line 粒度保留 MAE/median/hit%（跟現有 Kiritan/Itako 報告一致）
- word 粒度加報 PCO@0.3s（跟 JamendoLyrics 文獻一致）
- 必要時附 PCS（段重疊）但不放主表（頂多放逐首明細）

## 5. 現有數字入帳 vs 缺口

### 5.1 已有（直接入表）

| 資料集 | 軸 | 系統 | 來源 |
|---|---|---|---|
| Kiritan | Transcription | GAME, CE+CTC, ROSVOT | `benchmarks/kiritan/RESULTS.md` |
| Kiritan | Alignment (phone) | MMS_FA, MMS-JA, SOFA | 同上 phone-boundary 段 |
| Itako | Transcription | GAME, CE+CTC | `benchmarks/itako/RESULTS.md` |
| Itako | Alignment (phone) | MMS_FA, MMS-JA | 同上 phone-boundary 段 |
| MIR-ST500 | Transcription | GAME, CE+CTC, ROSVOT | `benchmarks/mir-st500/RESULTS.md` |
| JamendoLyrics | Alignment (word) | MMS_FA | `benchmarks/jamendolyrics/` |
| 自建 gold | Alignment (line) | MMS-JA, SOFA+isl, classic | `docs/alignment-benchmark.md` |

### 5.2 缺口

| 缺什麼 | 難度 | 產出 |
|---|---|---|
| ① 統一輸出腳本 `holistic_benchmark.py` | 低 — 讀現有 eval JSON,組表 | 一鍵出完整 markdown 表 |
| ② Pipeline row（GAME notes + MMS timing on Kiritan/Itako） | 低 — 兩邊都已跑,只差合併 | **第一個雙軸 system row** |
| ③ Opencpop（中文雙軸） | 中 — 下載 + 寫 GT adapter | 中文進 holistic 表 |
| ④ ICASSP 2024 label-priors CTC baseline | 高 — 實作/複現 | 文獻可比 alignment 基線 |
| ⑤ Kiritan/Itako alignment 用 PCO/AAE 重算 | 低 — 現有 HTK 邊界轉 mir_eval.alignment | 跟 JamendoLyrics 同指標可比 |

## 6. 評測 Harness 架構

```
holistic_benchmark.py
├── registry.json          ← 資料集 × 系統 × 結果路徑
├── eval_note()            ← wrap mir_eval.transcription（from eval_note_metrics.py）
├── eval_phone_boundary()  ← wrap phone_boundary logic
├── eval_word_alignment()  ← wrap eval_jamendo.py logic
├── eval_line_alignment()  ← wrap bench_aligners.py logic
└── render_table()         ← JSON → markdown / TSV / LaTeX
```

輸入：`registry.json` 列出每個 (dataset, system) 的結果 JSON 路徑。
輸出：
- `holistic_results.json`（機器可讀,所有數字）
- `holistic_table.md`（人類可讀,上面 §3 的表）
- `holistic_table.tex`（投稿用 LaTeX）

**不重新跑 inference** — harness 只讀已有的 eval 結果 JSON,組裝成統一表。
新系統要加入 = 跑自己的 eval → 把結果 JSON 放進 registry → 重跑 harness。

## 7. COnPOff+L — 單一數字 joint metric（2026-07-02 提案 + 文獻查證）

兩軸表回答「各自做到什麼程度」；COnPOff+L 回答「**誰是更好的卡拉 OK 系統**」——
一顆音要 onset(±50ms)、pitch(±50¢)、offset(max(50ms,0.2dur))、**romaji 音節**四關全過
才算配對成功，P/R/F1 算法與 COnPOff 相同（greedy matching，容數量差）。

分解式讀法：COn → COnP → COnPOff → COnPOff+L 每一階的跌幅 = 該維度的代價。

### 7.1 文獻查證結論（web 驗證,2026-07-02,一手全文）

**沒有任何已發表工作定義過 lyric-conditioned 的 COnPOff。** 逐一驗證：

| 系統 | 歌詞指標 | 音符指標 | joint？ |
|---|---|---|---|
| SongTrans (arXiv:2409.14619) | WER (char/phone) | pitch-WER + dur MAE | ✗ 分開報；alignment 用 notes-per-word 結構代理 |
| STARS (ACL'25 Findings, arXiv:2507.06670) | BER(20ms)/IOU (phoneme) | COnPOff + RPA | ✗ 兩張表分開 |
| VocalParse (arXiv:2605.04613) | WER | MAE_pitch/note/dur | ✗ 連 COnPOff 都不用 |
| Gu et al. (TOMM'24, ALT+AMT 多模態) | WER | COn/COnP/COnPOff | ✗ 獨立 task 報 |
| Nishikimi CRNN-HSMM (APSIPA'21) | **無歌詞輸出** | 編輯距離錯誤率 (Ep/Em/Ee/Eon/Eoff) | 不適用 |

- 唯一的 COnPOff 家族擴充先例：**T3MS (arXiv:2502.12438) 加 note-value（節奏）第四條件**
  — 是「加第四關」的可引用先例，但加的是節奏不是歌詞。
- **`mir_eval.transcription_velocity` 是現成的實作模板**：它正是「標準三關 + 每音符第四
  條件（velocity 門檻）」的結構——把數值條件換成 romaji 標籤匹配即是 COnPOff+L。
- 術語掃描（joint note lyric F1 / syllable-level singing transcription eval / COnPOff lyric
  extension）零命中。

**⇒ COnPOff+L 是真缺口：做 joint 任務的系統存在（SongTrans/STARS/VocalParse），
做 joint 指標的不存在。metric 本身即是 contribution。**

### 7.1b 第一批數字（Kiritan, 2026-07-02, `benchmarks/kiritan/conpoff_l.py`）

「都低分爛蘋果打架」的質疑實測否決（詳表在 `benchmarks/kiritan/RESULTS.md`）：
GAME×MMS_FA COnPOff+L = **.408**（地板由 COnPOff .499 決定,L 稅 6–13pp 非崩盤）；
aligner 軸鑑別力顯著（MMS_FA vs MMS-JA +3.6pp,CI [+2.8,+4.4],效應是 CI 的 4 倍）；
同 aligner 時稅近乎常數（GAME −9.1 vs CE+CTC −8.6pp）→ note-model 排名不變,
與預測一致。oracle 稅 −5.5pp = 指標本身的 mora 歸屬噪音地板（誠實 caveat）。
產品讀法：41% 的卡拉 OK 方塊四項全對。

### 7.2 實作路徑（原料已全數在庫）

| 原料 | Kiritan | Itako |
|---|---|---|
| GT 音符 | `gt_timefix.json` | `itako_gt.json` |
| GT 音素時間+romaji | `mono_label` (HTK) | `mono_label` (HTK) |
| 音符預測 | `game_raw_ja.json` / `ctcce_pred.json` | 同款 |
| 音素時間預測 | `phone_boundary/mms_{fa,ja}_htk/*.lab` | 同款 |

1. GT 合體：midi_label × mono_label → FullNote(onset, offset, pitch, romaji)。
   日文 1 mora ≈ 1 note；melisma (N:1) 採**寬鬆版**判定 — note onset 落在正確 mora
   的時間 span 內即 romaji 對（卡拉 OK 渲染真正在乎的是「這顆方塊配的字對不對」）。
2. 預測合體：GAME/CE+CTC notes × MMS phone timings，onset proximity 配對。
3. Matcher：仿 `mir_eval.transcription_velocity` 加第四條件。
4. 引用座標：T3MS 的 note-value 擴充當先例、transcription_velocity 當模板。

## 7.3 卡拉OK 轉譜 benchmark 地景調查（2026-07-02,雙 agent web 一手驗證）

**判決：不存在。** 學術線 + 資料集線獨立確認：

- **正式 benchmark**（共用測試集 + joint 協定 + 卡拉OK 成品評測）：零。MIREX 從無
  karaoke 任務（ALA 只有 lyrics timing）；日文學術圈 CiNii 零篇（被專利與
  JOYSOUND/DAM 封閉管線佔據,學術只做反向任務=給歌手打分）。
- **碎片**：SongTrans notes-per-word MAE（結構代理,私有中文資料 ad-hoc;全文 grep
  無 karaoke/YouTube 字樣）;**Masclef & Moussallam ISMIR 2021** = 唯一卡拉OK 框架的
  評測方法學論文（lyrics-only）— 兩個必引發現:**PCO 0.3s 門檻從無心理學驗證**、
  **感知容忍不對稱**（歌詞早出現>晚出現,受節奏/tempo 調變）→ COnPOff+L 容差設計
  的引用座標 + 未來不對稱容差版本的依據。
- **最大諷刺**：唯一實際生產此 artifact 的工具 **UltraSinger**（UltraStar 檔:音高條+
  歌詞 timing）**零已發表評測** → 用 COnPOff+L 評測 UltraSinger = 現成的第一次。
  **已執行（2026-07-02，數字見 kiritan/RESULTS.md §UltraSinger）**：commit e94d942、
  Kiritan N=50 全跑、0 失敗。UltraSinger COn .304/COnP .120/COnPOff .039/**COnPOff+L
  .002**/P(L|match) 6.0%，全軸顯著低於 GAME×MMS_FA（+L −.406 [−.439,−.368]）；lyrics-
  unknown 設定 + `~` over-segmentation（13361 vs GT 10370 notes）+ syllable over-hold
  是主因。清唱端到端轉譜真困難的首次量化證據。

**GT 資料源地景**（做「真卡拉OK」複音 benchmark 的原料）：

| 源 | note pitch | 音節 timing | 規模 | 狀態 |
|---|---|---|---|---|
| **USDB/UltraStar** | ✓ (int, C4=0) | ✓ (beat 制) | ~18k+ 檔 | 社群無 QC;**從沒人洗成乾淨 eval 集 = 真空**;做法同構 build_itako_gt(抽 N 首→RMVPE audit→人耳校驗) |
| DALI v1/v2 | ✓ (weak) | ✓ 4 層粒度 | 5,358/7,756 | 只修 global offset,local note 錯保留;人工驗證僅 105 首;YouTube link rot |
| kara.moe | ✗ | ✓ (.ass \k) | 29,785 | timing-only;MMS membership 汙染已證 |
| vocadito | ✓ | ✓ | 40 clips | 太小,但雙軸 GT 齊,可當 sanity 集 |
| DAMP | ✗ | 行級 | — | 2026 已停止發放,死路 |
| DAM/JOYSOUND 內部 | ✓ 完美 | ✓ | 百萬級 | 永不外流(CEDEC 2020 講座+專利 JP2016031394A 側寫) |

**定位結論**：COnPOff+L(指標) + Kiritan/Itako(清唱協定)已是第一個 joint 評測；
補上「複音真卡拉OK」一塊的路 = USDB 抽樣清洗(itako playbook)或 DALI-105 子集,
受測系統首選 UltraSinger。缺口在指標、測試集、受測系統三側同時成立。

### 7.3.1 USDB×kara.moe 一手量測（2026-07-02,資料在 data/usdb_kara/,本機 untracked）

兩個設計前提實測通過：

- **USDB 存取**：歌單搜尋要登入,但 (a) `usdb.hehoe.de` 有免登入全量索引
  （24,869 首 artist/title/id,快照 2024-11-20,~7% 已刪除 attrition）;
  (b) **detail 頁免登入**含 Language/Year/Genre/BPM/Golden/Songcheck 欄
  （`crawl_detail.py`）。TXT 下載才需帳號（免費表單註冊）。
- **DALI 對日文=0 首**（v1 metadata 一手解析,30 語言全無 japanese）→
  JA 核心 benchmark 的 DALI 訓練汙染顧慮直接消失;DALI-105 子集路線同時出局。
- **USDB 日文規模**：隨機 400 頁抽樣,live 371 頁中 13 首 JA（3.5%）→
  估 **~800 首**（95% CI 406–1337,以 live≈23k 計）。Genre 以 Anime 為主。
- **USDB∩kara.moe = 308 首驗證日文**（title+artist 雙確認 321,detail 頁
  Language 驗證 96% 精確;Anime 175/J-Pop 25/Vocaloid 9,year 1984–2021 中位 2008）。
  另 1,711 title-only 候選大多偽陽性（西洋歌撞羅馬字題名）,含少量真漏網
  （長音母音/姓名順序問題已修一輪 267→321）。
- **Songcheck=No 佔 306/308** → 「社群 GT 無 QC」前提坐實,gold 清洗是真工作。
- **MMS membership 實驗兩臂都夠**：∩=308（汙染臂,雙軸 GT）、
  JA∖kara.moe≈500（乾淨臂,需全量爬 detail 頁或登入列舉才能點名）。
  kara.moe 現存 23,568 筆中 jpn=19,939（比 29,785 少=repo/版本口徑差）。

## 8. 論文敘事位置

這張表回答的研究問題：**在同一批歌唱音檔上，note transcription 系統與 lyric alignment 系統各做到什麼程度？有沒有系統能同時做好兩件事？**

- 現狀：沒有。GAME 填左半,MMS 填右半,從來沒人把兩者放在同一張表裡。
- 貢獻 ①：首次在 Kiritan/Itako 上同框比較 note transcription 與 phone alignment。
- 貢獻 ②：首次把 GAME（SVS 社群工具,無已發表基準）放進 COnPOff 座標系。
- 貢獻 ③：Pipeline row = 第一個同時填兩半的 system（即使只是 GAME+MMS 組合）。
- 未來：end-to-end 系統（一個模型同時出 notes + phone timings）是終極目標,
  這張表就是它的評測框架。
