# Pitch Benchmark — gold 方法論、計分板、canonical 鏈

> 2026-06-10 整輪「樂譜層音高推論」工作的耐久文檔。
> 細節決策與踩坑在 [MEMORY.md](../MEMORY.md)；本檔是可重現的全貌。
> ⚠ 全域 `*.mid` gitignore 讓所有工作區 MIDI（含 tmp/ 的 gold）從不進 git —
> 不可再生資產存在 **`gold/`**，現為**私有 submodule**（`Lanternko/karaoke-jp-gold`，private；
> 含版權歌曲旋律轉譜故不公開）。dev 機用 `git submodule update --init` 取得。

## TL;DR 計分板（2026-06-10）

**Chidori**（vs humangold，312 顆全驗證音符，frame=60fps）：

| 候選 | frame exact | within2 | note-level（主 KPI） |
|---|---|---|---|
| q70 classic 鏈起點 | 0.600 | 0.832 | 52.6% |
| classic 鏈極限（+postfix+BP） | 0.651 | 0.831 | 60.6% |
| GAME raw | 0.695 | 0.844 | 69.2% |
| **GAME union（冠軍）** | **0.728** | **0.886** | **72.8%** |
| 原始 YT 官方 guide | 0.766 | 0.978 | —（信用軌跡 1.0→.908→.866→.851→.817→**.766**） |

**Byoushin**（vs pYIN-anchored 參考，僅看 delta 方向）：fusion 鏈 0.500 → GAME union **0.537** exact。

## Gold 方法論

### 信任層級（由教授的「譜面怎麼寫就怎麼標」原則確立）

```
1. シータピアノ鋼琴譜影片（YT 3hZPgk90KJA）＋ 2. Kojek 人耳裁決（虛擬鋼琴）
   >> 3. YT 官方 karaoke guide（gZ2oLzFVtYU） ≈ 4. RMVPE F0
```

關鍵教訓：**guide 與 F0 tracker 的「一致」不是獨立證據** — 兩者被同一種演唱偏差
（唱平、しゃくり）同向帶偏。B3 家族曾因「guide+RMVPE 一致」被誤判為真，
全音域 sheet 重讀後 22 顆全部翻案（B3→C4/Bb3）。

### 證據工具

- **亮鍵偵測**（[gold/chidori/read_lit_keys.py](../gold/chidori/read_lit_keys.py)）：
  Synthesia 式鋼琴影片逐 frame 讀按鍵（HSV 向量化、88 鍵幾何、R/L 手分類）。
  校準交叉驗證：偵測出的和聲與譜面和弦標記（Cm7/Fm7/Bbsus4…）逐一吻合。
- **+88s 詩節鏡像**：verse2 = verse1 + 88.0s（精確）；7 個鏡像推定後來全部被直接掃描證實。
- **人耳 dictation**：高跳樂句（`D5 Eb5 | C5 Bb4 G4 Eb4 | Eb4 G4 F4 | Eb4 D4 Eb4`）、
  verse 尾句（`Eb3 | Eb4×2 | D4 Bb3 G3 F3 | F3×3 G3 F3→Eb3`）、姊妹句同形（+88 證實）。

### 定罪的 guide 錯誤家族（全部 sheet/人耳裁決）

| 家族 | 數量 | 修正 | 性質 |
|---|---|---|---|
| F#4（副歌反覆樂句） | 44 | → G4 | 唱平的 G 被標半音低；F#3/F# 家族共 5 例同型 |
| B3（verse 導音假說） | 22 | → C4 / Bb3 | 唱平的 C；「兩來源一致」陷阱的教材 |
| A4 殘片（0.07–0.1s） | 10 | → Bb4 | Bb onset under-shoot 殘影 |
| 半音樓梯 / 長 F#4 | — | 滑音，非譜面音 | portamento 被 guide 畫成階梯/獨立音 |
| E3/E4 個案 | ~12 | 逐案（E 有真有假，E2 bass 佐證） | 調內錯誤要靠 sheet 全音域讀 |

### 指標注意事項

- **note-level majority 是主 KPI**；frame-exact 含 ~9pp 的「guide 畫 bar 慣例 vs mora
  切點」計時噪聲（boundary 錯誤在 guide 計時區 11.4%、在 mora 計時窗 2.3%）。
- humangold 的 patch 窗口時值抄 q70 mora 邊界（音高獨立驗證，但 q70 系邊界誤差在窗內不可見）。
- byoushin 參考是 pYIN-anchored，**只用於 delta 方向**，不可宣稱絕對值。
- **MIREX note metrics 已落地**（`scripts/eval_note_metrics.py`，mir_eval；onset 100ms/pitch
  50 cents/offset max(50ms, 0.2×dur)，`--merge-same-pitch` 正規化合併慣例）。
  ⚠ 絕對值尚不可直接比文獻：gold 的 onset 來自 karaoke-bar 慣例而非歌唱 onset —
  連 YT guide 對 gold 也只有 COnP 0.50。**當相對追蹤器用**（排名與 frame/note-level 一致）。
  現值（merge 0.08）：chidori game_union COnP 0.413 / COnPOff 0.205；
  byoushin 生產鏈 gamescore 0.457 / scorefix 0.369 / base 0.360。

## Canonical 鏈（Snakefile opt-in，flags pinned 在 driver 內）

```
classic：refit(q70 strict，手動 sidecar) → octavefix
         → bp_hybrid_relabel(d1/s0.02/r1.2) → score_note_postfix(refine/extend/
           shakuri/tail-falls/fill) → markers        # scripts/run_score_chain.py
GAME：   GAME large -l ja（專用 venv，cu129）→ postfix(僅 extend)
         → melody_union(fallback=classic 輸出) → markers  # scripts/run_game_chain.py
規則：   snakemake outputs/<song>/melody_markers.scorefix.mid   （classic）
         snakemake outputs/<song>/melody_markers.gamescore.mid  （GAME，依賴 classic）
```

實測否決（不要走回頭路）：GAME 直推混音（exact .48、八度錯 6.4%）、
refine/shakuri 套在 GAME 輸出（它的天然邊界更好）、align 模式當主旋律來源
（強制 mora 切分傷邊界，exact .627）、chroma ±1 snap（−0.5pp）、
latent-note Viterbi v1（−5pp）、pYIN 主軌（chidori −12pp）。

## 檔案地圖

| 資產 | 位置 | git |
|---|---|---|
| **Gold（不可再生）** | `gold/chidori/`、`gold/byoushinwo-kamu/` | 🔒 私有 submodule（karaoke-jp-gold） |
| 工作區 gold/實驗品 | `tmp/reference/{chidori,byoushinwo-kamu}/` | ⚠ untracked；其中 *.mid 被全域 ignore |
| 研究日誌 | `tmp/chidori_auto_research_report.md` | ⚠ untracked |
| 鏈 driver | `scripts/run_score_chain.py`、`run_game_chain.py`、`melody_union.py` 等 | tracked |
| eval harness | `scripts/eval_pitch_against_reference.py` | tracked |
| GAME | `third_party/GAME/`＋`~/venvs/karaoke-jp-game/`（**5090 必須 cu129**） | 部分 |
| Render sidecar | `outputs/chidori/karaoke...gameunion.mp4` | gitignored |

## 生產鏈跨歌實證（byoushin，2026-06-10）

byoushin 上游（mora pipeline 含 aligned/midi_timing）前輪已建；只跑三條新規則即得：
`octavefix base exact .382 → scorefix .438 → gamescore .529`（vs grid2 參考）—
**單純請求 `melody_markers.gamescore.mid` target 就 +14.7pp**，雙鏈端到端生產驗證通過。

## 待辦（接力點）

1. **User 目檢/試唱 gameunion render** → 決定 chidori 顯示預設是否切 GAME 鏈
2. 第三首歌 gold（方法已熟，~半天/首；哪首有鋼琴譜影片由 user 指定）
3. 殘餘人耳項：B 家族抽查（29.9–31.6）、低音 Eb3@52.0
4. note metrics 的文獻可比性：需要「歌唱 onset」版 gold（mora onset 可derive）
5. 更下一級：GAME fine-tune（屆時需做日文歌唱轉譜資料集 survey）
