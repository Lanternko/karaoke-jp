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

## 7. 論文敘事位置

這張表回答的研究問題：**在同一批歌唱音檔上，note transcription 系統與 lyric alignment 系統各做到什麼程度？有沒有系統能同時做好兩件事？**

- 現狀：沒有。GAME 填左半,MMS 填右半,從來沒人把兩者放在同一張表裡。
- 貢獻 ①：首次在 Kiritan/Itako 上同框比較 note transcription 與 phone alignment。
- 貢獻 ②：首次把 GAME（SVS 社群工具,無已發表基準）放進 COnPOff 座標系。
- 貢獻 ③：Pipeline row = 第一個同時填兩半的 system（即使只是 GAME+MMS 組合）。
- 未來：end-to-end 系統（一個模型同時出 notes + phone timings）是終極目標,
  這張表就是它的評測框架。
