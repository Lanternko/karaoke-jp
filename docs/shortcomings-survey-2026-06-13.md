# karaoke-jp 不足之處 Survey（2026-06-13）

> **Scope**：本文整合三維度程式碼稽核（音高/音符 AST + gold 方法論、歌詞對齊/timing + render、文字/振假名 + 調性/HUD + 分離 + 涵蓋）與七個主題的 deep research（含對抗式查證）。目標：把 karaoke-jp 現行 pipeline 的所有已知不足攤平、排序，並給出與既有負結果相容、可直接開工的下一步。
>
> **信心標記說明**：研究 findings 同時帶兩種標記——
> - **confidence**（原始 research 給的：high/medium/low）= 該主張在文獻上的證據強度。
> - **verdict**（對抗式查證給的：`confirmed` / `uncertain` / `refuted`）= 我方逐字回溯一手來源後的裁定。**凡 verdict=refuted 的主張一律不進結論**，只在原節留痕說明它為何被打掉；verdict=uncertain 的主張會明確標注「數字/歸因需修正」並降級使用。
> - **既有負結果鐵則**：interior-snap、F0 re-entry guard、chroma-snap key、latent-note Viterbi v1、GAME 直推混音、transcribe-then-align、naive ensemble（雙模型同錯）、SOFA 對分離人聲——這些已被人耳否決或實測崩潰，本文所有 recipe **不得**繞回它們；任何新方案必須說明它「為何不是這些的變形」。

---

## 0. TL;DR — 最重要的 5 個結論

1. **雙鏈 drift 是目前最大、最便宜可修的洞（純工程）**。README/GUI 走的 `Snakefile` 預設 `rule render` 吃的是 `octavefix.mid`（RMVPE+pYIN 八度修），**不是**文件宣稱的 canonical v14（GAME union + MMS + `make_display_grid` + pitch_patch + Essentia HUD）。72.8% note-level 與全部 v14 改良只活在 `/render-song` skill。任何照 README 跑的人拿到的是降級品。同時 `Snakefile` 硬編碼 `VOCAL_RATIO=0.30`（versions.json 已是 0.40），且不讀 `versions.json`——single-source-of-truth 是破的。**這不需要任何研究，是設定層的事，應該第一刀就修。**

2. **橫式（canonical 16:9）靜默丟棄 melisma_split / drop_notes / lyric_retime**。`make_display_grid.py:apply_pitch_patch()` 只認 `at`-keyed 的 relabel/ensure，完全不解析那三種 key（它們只在 `make_portrait_grid.py` 實作）。後果：chidori 的三連音 melisma 拆分、night-dancer 兩個 HAND CORRECTION drop_notes + lyric_retime 在橫式成品裡**無聲不套用、且無 WARNING**。使用者以為修過的橫式其實沒修。**純工程、高人耳影響、零知識瓶頸。**

3. **最大已知槓桿（CE+CTC 整合）仍閒置，但要先做歸因實驗**。CE+CTC 在 MIR-ST500 COnPOff 0.554 vs GAME 0.411（+14pp）、權重公開、harness 已驗證，卻沒進生產鏈。**但**：york135 已實證兩個強模型對 chidori humangold 的 COn@100ms 都卡 0.51-0.58（而 MIR-ST500/Kiritan 是 0.78-0.86），代表 chidori gold 的 note inventory 不是 MIREX 慣例——**所以「整合 CE+CTC」與「先用 RMVPE 協定自審 chidori gold」必須綁在一起做**，否則你拿一個更強的模型去對一個慣例不同的 gold，分數會自相矛盾。

4. **句尾 offset / 延音洞有了治本方向，且不踩任何否決過的路**。文獻最 actionable 的是 **Kong 2020 offset-time 回歸頭**（三角目標 g=1−|Δ|/(JΔ), J=5 + 拋物線內插取 sub-frame；對 ±50ms 標籤噪聲 robust，96.39% vs 分類 76.52%，verdict=confirmed），凍結主模型只加 offset 頭、只校 offset。比現行 `line_end_repair` 純 RMS 尾延伸治本，且 `extend-sustains` 已實證對基準無效（.411→.409）——所以**不要再調 RMS heuristic 的閾值，要嘛加 learned offset 頭，要嘛先做 rule-based「母音尾 + 能量衰減」升級版**。

5. **進字點（visual onset）的科學基礎已坐實，且解釋了所有 F0 onset 方案為何被人耳撤回**。人耳節律錨點是 **PAT / p-center ≈ 母音起始**，不是 physical/F0/breath onset（Polfreman 2013、Sundberg 2007、Marcus 1981、Huggins 1972 全 verdict=confirmed）。落地是**子音類別感知後推**（清子音 +30~50ms、濁子音 +5~15ms、純母音 0），完全離線、零模型、不碰 interior-snap 也不碰 F0 re-entry guard。**這是「為什麼 F0 中靶心卻被撤回」的最終解釋，也是最低成本的 timing 質感升級。**

---

## 1. 不足之處盤點

> 類別＝知識（K，需新方法）/ 工程（E，純整合）/ 驗證（V，缺 gold/metric）。severity 沿用稽核。「已 survey」＝既有 ast-survey/key-survey 或本批 deep research 已涵蓋。

| # | 不足 | 類別 | severity | 已survey | 一句現況 |
|---|------|------|----------|----------|----------|
| **E1** | Snakefile 預設鏈 ≠ canonical v14（雙鏈 drift） | E | high | 否 | README/GUI 跑出降級舊鏈，v14 只活在 skill |
| **E1b** | Snakefile 硬編碼 vocal_ratio=0.30，不讀 versions.json | E | medium | 否 | single-source-of-truth 破洞 |
| **B3** | 橫式靜默丟棄 melisma_split/drop_notes/lyric_retime | E | high | 否 | 兩首主打歌人工修正只在直式生效、無 WARNING |
| **B1** | 句尾 offset/延音是結構性天花板，extend-sustains 已證無效 | K | high | 是 | .411 vs SOTA .625，無 learned offset 校準 |
| **B2** | CE+CTC 大勝 GAME +14pp 卻沒進生產鏈 | E | high | 是 | 權重到手、實測勝出，純整合缺口 |
| **A1** | 自建 gold 只 2 首、byoushin 是 pYIN-anchored 非 sheet | V | high | 是 | 最高仲裁 gold 實質只 1 首，72.8% 不可外推 |
| **A4** | 顯示 grid 建在單一全域 BPM，無變速/局部 tempo | K | medium | 否 | ritardando/轉拍/自由速度間奏會破等寬前提 |
| **C** | 長間奏後 re-entry 偏移，現有 guard 全被否決/dead code | K | high | 是 | haru 型，production 知道有但放著 |
| **D1** | 句中 mora warp（整詞慢一拍）line gold 量不到、無事後解 | K | high | 否 | interior-snap 已否決，缺 mora-level metric |
| **D2** | eval shift sweep 在 test 指標上挑最佳平移 | V | medium | 否 | 分數系統性樂觀偏高 |
| **D3** | canonical override 含未耳測的 F0-derived 音高 | V | medium | 否 | NEEDS KOJEK EAR-CONFIRM 已進成品 |
| **F1** | 低音/弱音 miss 靠 classic fallback 硬撐，fallback 品質未量化 | E | medium | 否 | 偵測弱→補洞來源弱(60.6%)→gating 又誤砍，三段都脆 |
| **F2** | 橫式 note gating 仍砍輕聲段，落後 portrait | E | high | 否 | Tu-tu-lu 全靠手工 ENSURE 硬撐 |
| **G1** | 振假名 heteronym 未上 Yomikata，純 UniDic+override | E | medium | 否 | gikun 全靠人手，無信心分數/告警 |
| **G2** | MMS kana→romaji G2P 手刻表，表外字靜默丟、無告警 | E | medium | 否 | 兩支 G2P 表各自維護可能 drift |
| **H1** | Essentia HUD key 餵料脆，float32/非PCM16 silent fallback | E | medium | 是 | 整條 key→HUD 無 fallback 可見性 |
| **H2** | 調性無轉調/局部調性（J-pop 副歌升 key 直接錯） | K | low | 是 | survey 有對策但未落地，無轉調 gold |
| **H3** | 調性方法軸 n=1 pop gold，profile 未在 pop 調校 | V | medium | 是 | Essentia 大勝是在 EDM 量的 |
| **I1** | 分離單一固定模型，無 bleed 量測/無 fallback 切換 | K | medium | 否 | 下游全吃 separation 結構性誤差，無告警 |
| **J1** | 涵蓋極窄：14 首中 2 首有 override+耳測，gold 集中 chidori | V | high | 否 | 所有數字來自 ≤3 首，曲風/音域/語言邊界沒系統測 |
| **K1** | 曲風 OOD：演歌こぶし/melisma 與 rap 是兩個正交崩點 | K | medium | 否 | GAME 過切/欠切 + rap 無 pitch 錨點 |
| **misc** | RMVPE GT 稽核協定只用在別人資料集，沒自審自家 gold | V | medium | 否 | 自家地基沒被自己最佳工具量過 |
| **misc** | line_end_repair next-guard 0.25s 硬地板與 peaky 句尾衝突 | V | medium | 否 | butted boundary 可能系統性短切 |
| **misc** | warp↔MID2BAR 浮點對齊靠手工常數，單點 regression test | E | medium | 否 | try/except 吞例外，無 golden frame |
| **misc** | HUD up/down/long/high 計數是 chart 幾何非演唱判定，7半音任意 | V | low | 否 | 綁上游 pitch 品質卻無獨立檢查 |
| **misc** | ASR 幻覺過濾仍是硬編碼黑名單+魔術閾值 | E | low | 否 | 已退 timing 鏈但 override 生成仍經過 |
| **misc** | split_furigana 在不規則 okurigana/長音上 bail 成整詞 ruby | E | low | 否 | 邊界靜默退化、無告警、無 bail 率統計 |
| **misc** | LRC 句尾 cs 量化 + ruby +0.05s pad 無 collision 檢查 | E | low | 否 | 密 melisma 下 ruby 可能套錯 char |

---

## 2. 句尾 offset / 延音洞的學習式校準（B1）

### ① TL;DR
GAME 的 offset 是已量化定罪的最弱環節（MIR-ST500 COnPOff 0.411 vs Mel-RoFormer SOTA 0.625，差 21pp），而現行唯一補法 `line_end_repair`（純 RMS 尾延伸 heuristic）**已被實證不能替代 learned boundary**（`extend-sustains` 在基準上 .411→.409）。文獻共識：2024-2026 SOTA（Mel-RoFormer/ROSVOT/STARS/T3MS）幾乎都把 offset 當「note boundary 的一部分」處理，且 **T3MS 明確把「演唱 offset 時間」與「樂譜 note value」拆成兩個 token**——卡拉OK 要的是**前者（延音尾 visual sustain）**，不是樂譜時值。最 actionable 的後處理是 **Kong 2020 offset-time 回歸頭**。

### ② 關鍵 findings
- **Mel-RoFormer SOTA 不設獨立 offset head，offset 由 frame deactivation 推得**。MIR-ST500：Mel-RoF-large COnPOff .625 vs SpecTNT .550（+7.5pp）。`confidence: high / verdict: confirmed`。[arXiv 2409.04702](https://arxiv.org/abs/2409.04702)。含意：offset 失分主要來自 frame head 對延音/弱音的**早截**。
- **Kong 2020 三角目標 offset-time 回歸 + 拋物線內插，對 ±50ms 標籤錯位 robust**。g(Δ)=1−|Δ|/(JΔ), J=5；±50ms 標籤噪聲下 96.39% vs 分類法崩到 76.52%。`confidence: high / verdict: confirmed`（數字、機制、Eq.10 內插公式逐字核對）。[arXiv 2010.01815](https://arxiv.org/abs/2010.01815)。**這是最能直接當 GAME post-hoc offset 校準頭、且對 piano-roll gold 人工抖動 robust 的機制。**
- **T3MS 拆「offset_time vs note_value」**，MIR-ST500 Onset F1 .806 但 Offset F1 只 .759、full COnPOff .610——offset 是主瓶頸。`confidence: high / verdict: confirmed`（正式論文名「Note-Level Singing Melody Transcription for Time-Aligned Musical Score Generation」，系統名 T3MS）。[arXiv 2502.12438](https://arxiv.org/abs/2502.12438)。
- **COnPOff offset 容差公式 = max(50ms, 0.2·dur)**：⚠️ verdict=**uncertain**——此公式出處是 **Molina et al. 2014（ISMIR）**，**不是**原 research 掛的 2015 violin 論文。「offset 比 onset 難（release 斜率平緩、結束 vs 衰減需主觀閾值）」才是 2015 violin 論文（Liang/Su 等）支持的。引用時必須分開歸屬。對短音給 50ms、對長延音給 20% 看似寬，但 frame deactivation 在母音能量衰減處早截使長延音被切短失分——這個失分結構成立。
- **ROSVOT/STARS 用 word-boundary-conditioned 1D segmentation + focal loss**，而 **karaoke-jp 已有 MMS word/phoneme 邊界**，等於免費拿到它們通常要另跑 MFA 才有的條件輸入。ROSVOT noisy COnPOff 77.0、STARS GTSinger COnPOff .710。`confidence: high / verdict: confirmed`（含 STARS 數字——查證時 WebFetch 曾幻覺出 81.23/75.42，被原文 pdftotext 推翻為 71.0/70.2/50.2，採信後者）。

### ③ 可落地建議
**Phase 0（rule-based，零訓練，先做）**：把 `line_end_repair` 從「句尾 RMS 延伸」升級成**逐 note「MMS 母音段尾 + 分離人聲 RMS 衰減（跌破 peak−15dB 且持續 N frame）取交集」**，只動句尾/延音 note，非句尾 note 維持 GAME offset。完全複用既有 MMS 對齊 + 分離人聲 + RMS。成本低（幾小時）。**風險**：母音段尾 ≠ 人眼判定延音結束（與 F0 re-entry guard 被否決同源——生理 onset≠視覺 onset，offset 同理可能 breath/trailing voicing 拖尾）。必須人耳仲裁、保留 HAND CORRECTION 通道。

**Phase 1（learned offset 頭，不重訓 GAME）**：凍結 GAME，加一個 Kong 式 offset-time 回歸小頭（三角目標 + 拋物線內插），只在 GAME 給的 note 區間附近修 offset，不產生新 note、不改 pitch。成本中（piano-roll gold 弱監督即可，RTX 5090 一晚）。**風險**：GAME 是否暴露 frame-level 特徵未知（見開放問題），若否則退化成獨立小模型吃 mel。

**不要做**：繼續調 `line_end_repair` 的三個硬編碼 guard（tail_top_db=26 / next_guard=0.25 / tail_gap）——`extend-sustains` 已證 RMS 路線到頂。

### ④ 2024-2026 最新
- Mel-RoFormer（ISMIR 2024）：分離骨幹特徵→音符頭，COnPOff .625 為公開 SOTA。
- STARS（ACL Findings 2025）：MIT license + HF checkpoint，但推論需 word/phoneme 輸入（karaoke-jp 的 MMS 鏈剛好供得起）。
- T3MS（TASLP 2025）：offset_time/note_value 分離建模。
- GTSinger（NeurIPS 2024 Spotlight）：含日文子集 6.45h，可作日文 offset 監督/評測的潛在來源。

### ⑤ 開放問題
- GAME 是否暴露 frame-level activation 介面？決定 Phase 1 是真 post-hoc 還是獨立小模型。
- 「卡拉OK視覺 offset」缺標註規範：母音段結束？能量跌破 −X dB？還是人眼覺得方塊該收？沒先定義就無法產生一致監督——建議先在 1-2 首人耳標「理想方塊尾」建 micro-gold。
- offset-only 改善能否反映到最終 render？固定四分音符 grid + piecewise-linear warp 下，offset 誤差幾百 ms 可能被 grid 量化吸收，需確認可見增益門檻（呼應 interior-snap 教訓）。

---

## 3. 句中 mora 邊界與 note-ownership 對齊（D1）

### ① TL;DR
句中「慢一拍/配錯顆」（haru line29「世界/咲き誇る」三條都慢）的根因是 **mora→note 的歸屬（ownership）沒被建模**。純 nearest-onset MAE 對「把一顆 note 配給錯的 mora」幾乎無感——這正是 **interior-snap 紙面 MAE 降但耳測更錯** 的數學原因：MAE 把一顆配錯當成小誤差，而 IOU/token-accuracy 會直接記為該 segment IOU≈0。解法是**改變解碼的搜尋偏好（軟約束）而非事後硬 snap**，並**換 ownership-aware metric** 才抓得到弱類。

### ② 關鍵 findings
- **note-onset 是 syllable 邊界硬證據**（Dzhambazov 2016 VTHMM）：oracle onset 把對齊 accuracy 從 baseline 70.2% 推到 83.5%。`confidence: high / verdict: confirmed`，但**脈絡修正**：這是 Turkish Makam 單聲道**乾聲**，polyphonic 上效益大幅縮水；headline 改善是 absolute 5.5%（83.5/70.2 是 oracle 上限 vs baseline，非偵測 onset 實得）。
- **duration-informed HMM 大幅勝裸 HSMM**（Gong & Serra 2018）：25ms 容差 syllable onset F1 75.8% vs 41.0%。`confidence: high / verdict: confirmed`，但**關鍵限定**：duration prior 來自「老師範唱的參考時長」，不是抽象的每行 mora 配額——對 karaoke-jp 的對應是「若 GAME/gold 提供每 mora reference duration 才適用」。
- **STARS：word 邊界 overlap note 邊界→當約束**（hierarchical joint 解碼）。BER 18.6 / IOU 80.9。`confidence: high / verdict: confirmed`（機制原文「since the word boundaries overlap with the note boundaries, we employ these boundaries as constraints」逐字命中）。但僅 ZH/EN、無 melisma 機制、無日語——架構靈感非可用件。
- **SongTrans：顯式預測「每詞 note 數」處理 melisma**，MFA 把 melisma 誤判 silence 需併回前 phone。`confidence: high / verdict: confirmed`。
- **nearest-onset MAE 掩蓋配錯，正解是 segment IOU + boundary-F1 + signed error**。`confidence: medium / verdict: 機制成立`。
- **label-priors CTC 修 offset/interior**（Huang 2024）：offset 51→39ms、word boundary −26%、已在 TorchAudio。`confidence: high / verdict: confirmed`。**注意**：這是語音（Buckeye/TIMIT）評測，歌聲長母音/melisma 是 OOD，遷移幅度未知。

### ③ 可落地建議
**Recipe A（軟約束，首選）**：對每行做 monotonic 解碼時，在 transition cost 加一項 g(t)=高斯(σ≈75ms) 峰值落在最近 GAME note onset，使「mora i→i+1 邊界」偏向吸附到 note onset（半徑 0.15s 內生效）。**為何不是 interior-snap**：它不是事後硬 snap（那被否決），而是改變解碼搜尋偏好，仍讓聲學證據主導。成本低（改 cost 函數或加 Viterbi rescoring pass）。**風險**：GAME 句中 onset 本身可能 miss（低音弱音已知弱點），錯 onset 會把 mora 拉錯——需對 onset 信心加權、melisma 段不施約束。

**Recipe B（評測換骨，幾乎零風險，應立刻做）**：在現有 `eval_note_metrics.py` 旁加 per-mora segment IOU + boundary-F1@{20,50,100ms} + signed onset/offset（不取 abs）+ 一個 **mis-ownership rate**（被配給錯 note/錯鄰格的 mora 比例）。interior-snap 災難在 IOU/mis-ownership 下會立刻現形。**前提**：gold 必須有 per-mora start+end span（不只 onset）；現有 line-gold 若只標 onset，用下一 mora onset 當 offset 代理會略高估 IOU，需註明。

**不要做**：interior-snap（句中 char 邊界硬 snap 到 note）——已人耳否決。

### ④ 2024-2026 最新
VocalParse（2026, arXiv 2605.04613）interleaved word→note 解碼學 ownership，但僅 Mandarin、無 syllable 級、時間靠 retrained SOFA——架構靈感。

### ⑤ 開放問題
- 日文 special mora（ん/っ/ー/拗音）的 ownership 慣例無公開基準，karaoke-jp 的 line-gold 是少數能量化此處的資料。
- GAME 句中 onset precision/recall 未量化——軟約束該給多少權重取決於此。
- melisma 偵測（一 mora 多 note）karaoke-jp 只能靠「mora 數 vs GAME note 數」啟發式，誤判會讓硬分組鎖錯位（這是「每行 note 數當硬先驗」這條更激進 recipe 的主要風險，故列為可選而非首選）。

---

## 4. 進字點（visual onset）感知語意（D1 的物理基礎 / timing 質感）

### ① TL;DR
人耳判定的「進字點」不是 physical onset、也不是 F0/breath onset（常落在換氣/氣聲/塞音閉合），而是 **PAT / p-center ≈ 母音起始**。這在語音與歌唱兩邊獨立收斂，且**直接解釋了 F0 re-entry guard 為何被人耳否決**。落地是一張離線、零模型的「子音類別→δ 後推表」，完全不碰任何否決過的路。

### ② 關鍵 findings（全 verdict=confirmed，這是本批最硬的一組）
- **三層時間定義 PhOT<POT<PAT，正確對齊對 PAT（=p-center）**（Polfreman 2013）。`confirmed`，唯原文是「PAT *similar to* p-centre」非「等同」，改寫為「概念上對應」。
- **physical→PAT 延遲依攻擊型態**（Polfreman Table 1，逐字核對）：Bow +23.5ms、Reed +51.2ms、Strike +3.5ms、Beep +2.1ms、Pluck −0.46ms。`confirmed`。對應：母音前長清子音=慢攻擊需後推；乾淨母音起頭=幾乎不動。（注意：合成樂器非人聲，跨域屬類比。）
- **人耳辨時：打擊型一致性 12-20ms、非打擊型 ~42ms、辨別極限 ~10ms**（Collins 2006）。`confirmed`。→ 校正 quantum ~10ms、慢攻擊容忍窗 ~40ms。
- **歌唱：伴奏最常對「母音起始」同步，慢歌 lag 大快歌趨零**（Sundberg & Bauer-Huppmann 2007）。`confirmed`。
- **語音節律建立在母音起始（vowel onset rule）**（Huggins 1972）。`confirmed`，書目小修：companion 論文全名含「in natural speech」。
- **p-center 是整段刺激屬性，初子音越長越領先母音起始**（Marcus 1981）。`confirmed`。
- **Karaoke Mugen 社群慣例與科學一致**：「每音節在第一個母音被唱出時開始」「含濁子音、排清子音」。可直接抄。
- **重大 caveat**：JamendoLyrics/Jam-ALT annotation guide 對「word start 放時間軸哪」**完全沒規範**（只管文字格式），實務 = energy-onset 慣例。`confirmed`。→ **公開 benchmark 的 onset 慣例與 visual onset 有系統性 gap，紙面分數無法仲裁 visual onset 對錯。**
- **裸 CTC onset 偏晚、句尾偏早是兩回事**：label-prior 降 CTC peakiness，但通用 onset 偏晚 ≠ karaoke-jp 句尾偏早，**別混用同一校正**。`confirmed`。

### ③ 可落地建議
**Recipe（核心規則，離線零模型）**：對每個 mora，用既有 MMS/fugashi 音素序列判斷起頭類別，把 F0/CTC onset 往母音核推一個類別常數 δ：
- 清塞音（k/t/p + 拗音）δ≈+30~50ms；清擦音（s/sh/h/f/ts/ch）δ≈+25~45ms（按該 mora 實測子音時長封頂）；
- 濁子音（b/d/g/z/j）+ 鼻音/流音（m/n/r/w/y）δ≈+5~15ms；
- 純母音起頭 + N/っ/長音 special mora δ≈0。
- 硬上限：δ 不超過該 mora 實際子音段時長、onset 不越過下個 mora。

**為何不是 F0 re-entry guard**：guard 用 F0 voicing onset（落在 breath/pre-phonation），這個用「子音類別 + 母音核」當錨——正是 guard 被否決的反面。成本低（25-50 行查表 + per-mora 位移函數）。**風險**：δ 是經驗常數，首版用文獻量級起跳，日文唱腔實際值要用鋼琴譜/人耳 gold 校；擦音時長變異大需動態封頂。

**評測協定調整**：在 gold JSON 加 `onset_convention` 與 `attack_class` 欄，讓自家 vowel-onset gold 與公開 benchmark 的 energy-onset 慣例**脫鉤、各自評**——這是解決「F0 中靶心卻被撤回」評測錯配的根本。

### ④ 2024-2026 最新
Nature Comm Biol 2025（speech-to-speech sync governed by p-center，日耳曼≈母音起始）；Frontiers Psychology 2023（歌唱節律規律性高於語音，對 karaoke 有利）。

### ⑤ 開放問題
- 日文唱腔子音類別 δ 實測值未知（文獻量級來自西方語音/器樂/美聲）。
- melisma/連唱母音（る與一度音高那類）無子音邊界也無能量谷，proxy 失效——回退純 pitch-bar 邊界 + 手動？
- geminate っ（塞音閉合是靜默）、devoiced vowel（です的す、した的し，日文特有，文獻幾乎沒碰）需 special-mora 專門規則。
- 半獨立性疑慮：energy-rise proxy 雖非 F0 tracker，但仍從同一段唱腔抽，是否真脫離同向偏差？需與鋼琴譜或不同歌者翻唱對照驗證。

---

## 5. 長間奏後 re-entry（C）

### ① TL;DR
根因不是「F0 進場點抓不準」，而是 **CTC 在長間奏的兩個結構性病徵**：(a) blank dominance/peaky——間奏整段塞給 blank，後一句 spike 被累積 blank 質量拖移；(b) 整曲單次 forced_align 無 re-sync，monotonic 路徑漂掉無法回收。最低成本最高勝率的落地是 **`<star>` token + 結構化切段**，且 `<star>` 只吸收音訊、不決定進字點——天然繞開 F0 re-entry guard 被否決的點。

### ② 關鍵 findings
- **blank dominance 造成 peaky**（Huang 2024 §2.3）。`confidence: high / verdict: confirmed`（原文「blank is the most versatile and frequent token」+ torchaudio「blank 兼任 repetition 與 silence」逐字命中）。
- **MMS/torchaudio 原生支援 `<star>` wildcard**，治「文字沒涵蓋整段音訊」，tutorial 範例 star span 0.000-2.595s 正確吃掉前段。`confidence: high / verdict: confirmed`（MMS_FA bundle 預設帶 `<star>` 維度）。**karaoke-jp 已用 MMS CTC，可直接注入。**
- **CTC-segmentation 首字 zero-cost + s_seg 信心分**（Kürzinger 2020）。`confidence: high / verdict: uncertain`——機制與多數數字 confirmed，但**一處實質誤述**：robustness 實驗 prepend/append 的是「每檔自身的真實語音片段」（更難的 distractor），**不是「噪音」**。TEDlium CTC mean 0.31s、88.8%<0.5s、加語音前後綴仍 89.2%（MAUS 掉 66.9%）。
- **Demirel anchor-based 兩段式**（biased 20-gram LM 找連續 5 字 anchor 切段）。`confidence: high / verdict: confirmed`，記憶體 13740MB→343MB、Demucs+Jamendo mean AE 0.31s/median 0.05s/PCS 0.93。
- **WhisperX VAD cut-and-merge**：`confidence: high / verdict: uncertain`——核心機制 confirmed，但**超參數錯**：claim 寫的 onset/offset 0.1、min-silence 1s **不是 WhisperX 論文值**（論文是 onset=0.767/offset=0.377/min_duration_on=0.136s）；merge 門檻是 30s（=Whisper 訓練輸入長度）。「VAD 只決定切段不決定字 onset」是對 karaoke-jp 的延伸建議，超出論文。
- **Gupta 把間奏當顯式類別建模**：`confidence: medium / verdict: uncertain`——主張正確但引用對象修正（實際描述機制的是 arXiv 1909.10200 / ICASSP 2020，非所引 MIREX abstract），且「>0.5s 門檻」是 claim 自加。

### ③ 可落地建議
**Recipe 1（`<star>` 注入，首選，零訓練）**：用 lyric line 行間 gap 偵測長間奏（gap > 8~12s），在 gap 兩端 transcript 位置插 star token，跑 forced_align，間奏被 star 吸收。`forced_align_mms.py` 目前是整曲單次無 star，這是最小改動點（~30-50 行 + 升 torchaudio）。**風險**：gap 判定過鬆會把呼吸包成 star，用保守門檻 8-10s，並在 haru/night-dancer 耳測。

**Recipe 2（VAD 段內逐段對齊）**：把整曲一次改成人聲活動段內逐段，**VAD 只當切段邊界、不當 onset oracle**（繞開 guard 被否決的因）。**風險**：SVD 漏判輕聲段（RMS 已知漏 Tu-tu-lu）會把整行歸錯段——務必用「RMS voiced ∪ MMS char 證據」聯集 gating（portrait 已驗），且只在長間奏切、句內絕不切。

**Recipe 3（anchor re-sync 信心閘）**：forced_align 後用 s_seg 信心分打分，低信心且前有長間奏的行觸發局部重對齊，仍低則標 HAND CORRECTION。**風險**：「雙模型同錯」的 re-entry 行 s_seg 是否也給高分（自信地錯）需在 haru 2/12 錯行驗證。

**不要做**：F0 re-entry guard（breath onset≠visual onset，已否決）；naive ensemble（ja/fa 同向漂 +0.61，已否決）；回退 aeneas/gentle（對長間奏比 CTC 更糟）。

### ④ 2024-2026 最新
CrisperWhisper（Interspeech 2024）：160ms 停頓均分 + 50ms 過濾，詞分段 F1 0.847——可移植到 rap 高音節率解碼。masked-CE（DAFx 2025）Jamendo median 0.041s（屬重訓選項）。

### ⑤ 開放問題
- star span 會不會反吃掉間奏後第一字的起音 frame 使 onset 偏晚？需在 line gold 量 star 邊界與真進字距離。
- label-prior/masked-CE 重訓在「卡拉OK日文 + 分離人聲」能否複製論文增益？域差距未知，且 MMS ckpt 已微調過，二次微調可能 regression。

---

## 6. 全域 BPM / tempo 假設（A4）— 文獻空白區

### ① TL;DR
整個顯示 grid（固定四分音符寬度、duration 量化、page span、count-in、warp anchor）全乘上 `quarter = 60/bpm_2dp` 這**單一全域常數**，來自對 instrumental 跑一次 `librosa.beat.beat_track` 取第一個值，只做 40-240 範圍檢查，**無 confidence、無 half/double-tempo 消歧、無段落級/變速偵測**。對 J-pop 常見的 ritardando、轉拍、自由速度間奏、慢板副歌，「等寬四分音符＝等速游標」前提會破裂。**現有兩份 survey（ast/key）完全沒碰 tempo/節奏推斷這條。**

### ② 現況（稽核，無對應文獻 finding——這是真空白）
- `quantize_durations.py:114-120` `_estimate_bpm()` 單值 beat_track。
- `make_display_grid.py:381-382,406-423` 全 grid 乘單一 quarter。
- piecewise-linear warp 把 bar 對回真實時間救了**同步**，但 quantize 階段已把 note 的**相對寬度比例**定錯（slot 比例錯，游標就會在頁內忽快忽慢）。

### ③ 可落地建議（謹慎，因無 deep research 背書）
- **最小**：先加 half/double-tempo 消歧 + confidence 門檻，低信心曲走 per-song override BPM（沿用 override 哲學）。
- **中期**：對 ritardando/自由間奏曲做段落級 tempo（librosa 動態 tempo 或 sibling beat-tracking 子專案的 DBN）——但這條**需先有 gold 證明問題真的可見**（呼應 §2 開放問題：誤差可能被 warp 吸收）。
- **驗證優先於實作**：先在一首明顯有 ritardando 的 J-pop ballad 上人耳檢查「頁內游標是否忽快忽慢」，確認這是真痛點再投入。

### ④/⑤ 2024-2026 / 開放問題
- 這條沒做 deep research，**標為「需先實驗驗證是否為真痛點」**，不是已 survey 完可動手的知識瓶頸。tempo 推斷 SOTA（TempoCNN、Beat-This）若要上需另開一輪 research。

---

## 7. 文字 / 振假名 / G2P（G1, G2）

### ① TL;DR
讀音消歧的最佳路 = **用音訊當仲裁器**（與專案「音訊/人耳是最高仲裁」哲學同構），而非靠 LLM。gikun/義訓在定義上不可從漢字或上下文還原，override 是唯一正解但來源應改抓官方 ruby。歌唱 phonology：說話會無聲化的高母音在歌唱中傾向復聲，forced-align 應走有聲 pron。

### ② 關鍵 findings
- **音訊仲裁讀音**（Transcript-Prompted Whisper + Dictionary-Enhanced Decoding，Baidu 2025）：MeCab/UniDic 列每 phrase 全候選讀音，選與音訊預測發音編輯距離最小者，phonemic CER 0.57%（純文字 OpenJTalk 3.56%、純音訊 1.13%），全 local open weights。`confidence: high / verdict: confirmed`，唯殘留錯誤例是「ブィ↔ビ」非「プィ↔ビ」。**屬 TTS 標註場景，方法可移植到歌唱讀音仲裁。**
- **純 LLM 推 furigana 仍嚴重不可靠**（PACLIC 2025）：GPT-5 ruby F1 僅 10.26%、本機 gpt-oss-20b 4.72%；對照 kuromoji.js 純讀音錯誤只 1.3%。`confidence: high / verdict: confirmed`，唯分心數字張冠李戴（GPT-5 錯題 1/30→7/30；gpt-oss-20b 數學 60%→46% 是兩個不同模型，皆真但別黏成一句）。**硬支持 GUI no-LLM 政策。**
- **字典法瓶頸是斷詞邊界（51.3%）非讀音查找（1.3%）**，且「無上下文反而更準」（CER 1.44% vs 2.84%）。`confidence: high / verdict: confirmed`。
- **Yomikata 只覆蓋 130 heteronym、整體 94%**，補不到 gikun。`confidence: high / verdict: uncertain`——數字全對，但「明言不處理 creative/gikun」在官方頁查無此明文，是由「字典內 heteronym 選擇器」設計推得，非官方聲明。
- **gikun 定義上不可預測**（運命=さだめ、悪夢=しんじつ、死の支配者=オーバーロード）。`confidence: high / verdict: confirmed`，唯「歌手只唱 furigana」硬出處是 japanesewithanime 非 en.wikipedia/Furigana。
- **歌唱高母音復聲，forced-align 應走有聲**：⚠️ verdict=**refuted**（部分）——無聲化的**語音事實**（PMC6476939：高母音縮短 25%/−39ms）成立，但「certain female/formal speech 會復聲」這句**不在該 PMC 論文**（出自 Wikipedia Japanese phonology 承 Vance 1987），且歌唱-復聲連結**只有 enka/pop 坊間觀察、無學術 primary source**。→ **工程含意（歌唱不強制無聲化 phone）方向合理，但須標為「弱證據/待耳測」，不可掛在 PMC6476939 名下當定論。** narabas「不分有聲/無聲母音」是這條的真正硬支撐（`confirmed`）。

### ③ 可落地建議
**Recipe 1（讀音音訊仲裁層）**：對分離人聲跑 whisper-large-v3-turbo（條件=gold 歌詞）取音訊預測發音，fugashi+UniDic 列每詞全候選，選編輯距離最小者，輸出再餵 romaji→MMS CTC。**為何不違反 no-LLM**：非生成式、本機、ASR 只驗讀音不進 timing 鏈（與 ASR 退出 timing 決策相容）。**風險**：Whisper 對分離人聲可能幻覺，用 gold 歌詞硬約束候選集、只信落在候選內的預測，集合外退回字典首選 + 標人工複查；歌唱母音拉長干擾發音預測，只取 consonant/mora 骨架比對。

**Recipe 2（gikun 偵測器）**：當 fugashi 讀音與歌詞本 ruby 不符且不在 UniDic 候選集內，自動標記 gikun 並以 ruby 為準；override 來源從「手動聽打」升級為「官方歌詞本/MV 燒入 ruby 抓取」。**風險低**（純規則 + 人工確認）。**前置**：須先量化 fugashi+UniDic 在歌詞域的斷詞錯率（PACLIC 顯示斷詞是最大錯源），否則 gikun flag 被斷詞噪音淹沒。

**Recipe 3（歌唱 phonology）**：forced-align 對「說話會無聲化的高母音」一律展開全母音；長音/促音/撥音各保留一 mora slot。現行 MMS Latn 本就不標無聲化＝**已隱性正確**，此 recipe 是「確認不要誤引入無聲化」+ 把 special mora slot 規則寫死。**風險低**，但句尾です/ます的す在快歌可能真氣聲化短促，`line_end_repair` 對弱尾要 robust（已是已知弱點）。

**Recipe 4（可選，離線 batch）**：FLFL 1B 或 MFA `--num_pronunciations 3` 當 override 候選產生器（人工勾選關卡，不進 GUI/render pipeline）——守住 no-LLM 紅線，需 Kojek 裁定「離線輔助 vs GUI 即時」界線。

### ④ 2024-2026 最新
Baidu Transcript-Prompted Whisper（arXiv 2506.07646，最高可落地價值）；PACLIC 2025 Ehara（硬數字證 GUI no-LLM）；FLFL（HF，離線 override 候選）。

### ⑤ 開放問題
- 音訊仲裁在「分離人聲 bleed + 歌唱母音拉長」上能否準確供編輯距離比對？需在三首 gold 做 PoC。
- gikun 自動偵測 precision：多少「假 gikun」其實是 UniDic 缺詞/斷詞錯？
- 歌唱無聲化復聲是否真 near-100%？只有間接證據，需在句尾段人耳抽查。

---

## 8. 分離 artifact 對下游 + 曲風 OOD（I1, K1）

### ① TL;DR
分離對下游 MIR **不是免費午餐**：note transcription 端混音直推常勝分離人聲（T3MS ablation），alignment 端則是「artifact 少的分離器贏，不是 SDR 高的贏」（Demirel：波形 Demucs 勝頻譜 Spleeter）。karaoke-jp 已用對的東西（Mel-Band-RoFormer bleedless = 低 bleed），換模型邊際小，真正 leverage 是**輸入策略**。曲風 OOD：演歌こぶし/melisma 與 rap 是兩個正交崩點，最小代價是 **genre router**（按曲風關掉會反咬的後處理）而非換模型。**2026 generative/diffusion 分離器對 gold 仲裁是禁區。**

### ② 關鍵 findings — 分離
- **混音直推勝分離人聲（同模型 ablation）**（T3MS）：`confidence: high / verdict: uncertain`——方向與機制 confirmed，但**承重數字標錯指標**：0.808/0.677 是 Onset Precision/F-measure，**真正 COnPOff 是 mixture 0.610 vs vocal 0.494**（差 0.116 非 0.131）。引用須改。**對 GAME 已否決**（GAME 直推密伴奏混音實測崩 exact .48/八度 6.4%），但「分離抹掉句尾延音低能量諧波」可解釋 offset 洞，值得做局部 A/B。
- **SOTA 是「分離特徵→音符頭」共享編碼器非「分離 wav→獨立轉譜」**（Mel-RoFormer）：`confirmed`，唯小修——是差異化學習率 fine-tune（核心 blocks 被更新），非「凍結」。GAME 落後一個架構世代且無公開權重，屬不可得 SOTA。
- **ASR 餵分離人聲傷 WER**（Jam-ALT：27.9→33.5）：`confirmed`。反向確認 ASR-on-分離不可靠（karaoke-jp 已踢 ASR 出 timing，正確）。
- **配 VAD 切段 + 長 context 時分離轉益**（arXiv 2506.15514）：`confirmed`。karaoke-jp 的 `line_end_repair` + 已知 line 邊界**本質就是這個 VAD-segment 機制**（獨立重發明了讓分離轉益的配方）。
- **波形分離器勝頻譜分離器**（Demirel：Demucs MAE .31 > Spleeter .38）：`confirmed`。支持用 Mel-Band-RoFormer 而非 Spleeter-class。
- **MAJL 量化分離誤差級聯傷 pitch**：`confidence: high / verdict: uncertain`——數字 confirmed，但「noisy pitch labels」是 MAJL 自身要解的 Challenge 非 pipeline 條列病灶；venue 是 ACM MM 2024。
- **generative/diffusion 分離（2026）對 gold 是禁區**：會幻覺出 plausible-but-wrong 人聲，毒化 pitch/timing gold（同否決 transcribe-then-align 的同源風險）。`confidence: medium`。**應寫進 spec/NEVER。**

### ② 關鍵 findings — 曲風 OOD
- **melisma 重曲風 note 切分掉約 8pp**（Gómez flamenco）：`confidence: high / verdict: uncertain`——polyphonic 數字（72.6%/67.4%/21%）與機制 confirmed，但「8pp 純 melisma 稅」是**容差錯配**（50¢ pop/jazz vs 100¢ flamenco），不可承重；該系統 50¢ flamenco 其實 ~76.8%。
- **glissando/vibrato 跨域第一個崩**（GTSinger：跨語 F1 0.70→0.51/0.58）：`confirmed`，唯「ROSVOT vibrato 0.374」無法溯源，移除。
- **J-POP 本身已 22.8% 是 portamento，演歌是程度非種類 OOD**（COSIAN）：`confirmed`。
- **note-transition 連續音高被誤判成獨立事件=過切**（PrimaDNN'）：`confirmed`，補救是**顯式餵 pitch 證據而非調邊界閾值**。
- **rap 對齊骨架是 beat/tatum grid + 語言學先驗非 pitch/F0**（CS229 2009）：`confirmed`，唯精確 10.7%/7.7% 在 Figure 6（圖）只能文字核到「~3pp gap」；屬 2009 課堂單曲，當引導性先驗非強證據。
- **曲風偏差 note-F1 掉 14pp，dynamics 比 onset 脆**（arXiv 2512.14602）：`confirmed`，唯是鋼琴 AMT 非歌聲，14pp 是量級方向非歌聲精確值。

### ③ 可落地建議
**分離 Recipe 1（選型標準換骨）**：維持 CTC-on-分離人聲，但分離器選型從「SDR/聽感」改成「下游 alignment MAE + 子音清晰度人耳」，只有某 bleedless 變體在三首 line-gold 句首/句尾 MAE 實測勝出才換。成本低（重跑既有 harness）。

**分離 Recipe 2（句尾延音解耦）**：alignment 用低-bleed 分離人聲定字邊界，但 offset/延音尾的能量判據改回參考混音或弱分離訊號（避免 bleedless 砍太狠抹掉延音尾低能量諧波）。**風險中**：混音 RMS 會被伴奏污染，需 gate，先在 chidori/byoushin 驗。

**分離 Recipe 3（防呆）**：把 generative/diffusion 分離列入 NEVER-for-gold，寫進 spec。

**曲風 Recipe（genre router，不換模型）**：偵測高 portamento 密度/低音節率→演歌模式（關 interior 合併/snap、放寬 GAME note 最小時長容 melisma、offset 加大上限）；高音節率/低 pitch 變異/spoken→rap 模式（接 sibling beat-tracking 子專案產 tatum grid，mora-per-beat 先驗均分，MMS CTC 當局部微調，`line_end_repair` 放寬/關閉）。**為何不換模型**：VocalParse/STARS 等 SOTA 是把 rap/rock **過濾掉**而非解掉，反證 router + 先驗比換模型務實。成本低（純啟發式 + config 分支）。**風險低**（最壞退化成現行行為，router 誤判走預設 J-POP）。

### ④ 2024-2026 最新
ISMIR 2025「Perceptual Errors in MSS: looking beyond SDR averages」（SDR 平均漏掉感知/下游關鍵錯誤——直接背書「不要用 SDR 榜選分離器，要用下游 COnPOff/MAE/人耳子音」）；ISMIR 2025 LBD「對齊準度隨分離工具大幅變動」；COSIAN（J-POP 多技巧標註，可請求取用）。

### ⑤ 開放問題
- 句尾 offset 洞有多少來自分離 dropout vs GAME 本身截尾條款？需 offset-only 對照才能歸因。
- Mel-Band-RoFormer bleedless ft 的 SAR（musical-noise）實際多少？社群只報 SDR/bleedless（SIR 向），沒人報 SAR（你關心的 dropout/子音抹除是 SAR 向），可能需自己在 Kiritan-mix 量。
- CTCCE coverage miss ~20% 中 leakage vs dropout 各佔多少？兩者修法相反（前者要更乾淨分離、後者要更少分離），不分清會互相抵消。
- sibling beat-tracking 鏈在密 hip-hop 鼓（808/trap hi-hat）上是否夠穩當 rap grid？未在 rap 實測。
- 公開含時間對齊的演歌（enka）基準仍是空白——可能需自建並貢獻。

---

## 9. 優先序路線圖 + 推薦

排序軸：**人耳影響 × 落地成本 × 是否被否決過**。每項標瓶頸所在。

| 序 | 動作 | 瓶頸 | 人耳影響 | 成本 | 對應 |
|----|------|------|----------|------|------|
| **1** | **修雙鏈 drift**：讓 Snakefile 預設 render 走 canonical v14，或明確標 README/GUI 改走 skill；Snakefile 讀 versions.json（含 vocal_ratio） | 純工程 | 高（README 用戶全拿降級品） | 極低 | E1, E1b |
| **2** | **橫式補 melisma_split/drop_notes/lyric_retime**：把 portrait 的三個 apply_* 移植進 make_display_grid，並對未知 key 印 WARNING | 純工程 | 高（兩首主打歌橫式沒修） | 低 | B3 |
| **3** | **橫式 note gating 改 RMS∪MMS-char 聯集**（跟進 portrait v15） | 純工程 | 高（Tu-tu-lu 不再靠手工 ENSURE） | 低 | F2 |
| **4** | **評測換骨**：加 per-mora IOU + boundary-F1 + signed error + mis-ownership rate；eval shift 改固定全域/報 shift=0 | 純工程（驗證） | 間接（抓得到弱類才能改） | 低 | D1, D2 |
| **5** | **進字點子音類別後推表**（離線零模型 δ 表 + onset_convention gold 欄） | 已survey完→可動手 | 高（timing 質感，解釋 F0 撤回） | 低 | D1/§4 |
| **6** | **`<star>` token 注入長間奏** | 已survey完→可動手 | 高（haru 型 re-entry） | 低 | C |
| **7** | **gold 自審**：對 chidori/byoushin 跑 RMVPE GT 稽核協定（自家已有腳本）；補第三首 gold | 純工程（驗證） | 間接（地基） | 中 | A1, misc |
| **8** | **CE+CTC ∪ GAME 整合**（綁 #7 一起做） | 還需實驗驗證（gold 慣例先釐清） | 高（+14pp 潛力） | 中 | B2 |
| **9** | **句尾 offset Phase 0**（母音尾 + RMS 衰減升級 line_end_repair） | 已survey完→可動手 | 中-高 | 低 | B1 |
| **10** | **讀音音訊仲裁層 + gikun 偵測器** | 已survey完→可動手 | 中（讀錯字） | 中 | G1 |
| **11** | **genre router**（演歌/rap 關反咬後處理） | 還需實驗驗證 | 中（OOD 曲） | 低 | K1 |
| **12** | **句尾 offset Phase 1**（Kong learned offset 頭）/ 句中軟約束解碼 | 還需實驗驗證 | 中 | 中-高 | B1, D1 |
| 後 | 分離選型換骨、tempo 消歧、調性 pop gold/轉調、HUD 計數驗證 | 混合 | 低-中 | 中 | I1,A4,H,misc |

### 我的推薦 —「你說 GO 我就動」的第一刀

**修雙鏈 drift（路線圖 #1）。**

理由：(a) **零知識瓶頸、零研究風險**——它不是哪個模型該不該換的問題，是「文件宣稱的 canonical 跟實際 default 跑的不是同一條鏈」的純設定 bug；(b) **人耳影響最大且最隱形**——任何照 README/GUI 跑的人（包括未來的你）拿到的是 octavefix 舊鏈降級品，72.8% 與全部 v14 改良都不生效，而這**不會報錯**；(c) **它是 #2、#3、#8 等所有後續整合的前提**——只要 Snakefile 不讀 versions.json、不走 make_display_grid，後面把 CE+CTC、melisma patch、聯集 gating 接進 canonical 都會繼續 drift。

具體第一刀：讓 `Snakefile` 的 `rule render` 依賴鏈走 `make_display_grid`（含 game_union + pitch_patch + Essentia HUD），並把 `VOCAL_RATIO` 等參數改成讀 `config/versions.json`；若短期不想動 Snakefile 結構，至少在 README/GUI 入口加明確警告「canonical 成品請走 `/render-song`，Snakefile default 為舊鏈」，避免 silent 降級。確認方式：對 chidori 同時跑 Snakefile default 與 skill，diff 兩個 mp4 的 grid/HUD，確認差異消失或被正確標示。

---

## 10. 主要來源（去重，附 venue/年）

**進字點 / 對齊感知（§4, §3, §5）**
- Polfreman, *Comparing Onset Detection & Perceptual Attack Time*, ISMIR 2013 — `confirmed`
- Sundberg & Bauer-Huppmann, *When Does a Sung Tone Start?*, J. Voice 21(3) 2007 — `confirmed`
- Marcus, *Acoustic determinants of P-center location*, Perception & Psychophysics 1981 — `confirmed`
- Huggins, *On the Perception of Temporal Phenomena in Speech*, JASA 51(4B) 1972 — `confirmed`
- Nature Comm Biol 2025, *Speech-to-speech synchronization governed by the P-center* — `confirmed`
- Karaoke Mugen contributor guide (syllable timing on vowel onset), 2024 — `confirmed`
- MIREX 2024 Lyrics-to-Audio Alignment / Jam-ALT GUIDELINES (energy-onset 慣例 caveat) — `confirmed`

**CTC 對齊 / re-entry（§5, §3）**
- Huang et al., *Less Peaky and More Accurate CTC FA by Label Priors*, ICASSP 2024 (arXiv 2406.02560) — `confirmed`
- torchaudio CTC FA API tutorial / MMS (`<star>` token), 2023 — `confirmed`
- Kürzinger et al., *CTC-Segmentation*, SPECOM 2020 — `uncertain`（前後綴是語音非噪音）
- Demirel et al., *Low Resource Audio-to-Lyrics Alignment*, ICASSP 2021 — `confirmed`
- Bain et al., *WhisperX*, INTERSPEECH 2023 — `uncertain`（超參數修正）
- Gupta et al., *Lyrics Alignment in Polyphonic Music*, ICASSP 2020 (arXiv 1909.10200) — `uncertain`（引用對象修正）
- CrisperWhisper, Interspeech 2024 (arXiv 2408.16589) — `confirmed`

**note-AST / offset / ownership（§2, §3）**
- Wang et al., *Mel-RoFormer for Vocal Separation and Vocal Melody Transcription*, ISMIR 2024 (arXiv 2409.04702) — `confirmed`
- Kong et al., *High-resolution Piano Transcription by Regressing Onset/Offset Times*, TASLP 2021 (arXiv 2010.01815) — `confirmed`
- *Note-Level Singing Melody Transcription* (T3MS), TASLP 2025 (arXiv 2502.12438) — `confirmed`(§2) / `uncertain`(§8 指標)
- Li et al., *ROSVOT*, ACL 2024 Long (aclanthology 2024.acl-long.526) — `confirmed`
- Guo et al., *STARS*, ACL Findings 2025 (arXiv 2507.06670) — `confirmed`
- Wu et al., *SongTrans*, 2024 (arXiv 2409.14619) — `confirmed`
- Dzhambazov et al., *Note Onsets for Lyrics-to-Audio Alignment (Makam)*, ISMIR 2016 — `confirmed`（脈絡：乾聲/Makam）
- Gong & Serra, *Singing Voice Phoneme Segmentation*, Interspeech 2018 (arXiv 1806.01665) — `confirmed`（脈絡：參考時長）
- VocalParse, 2026 (arXiv 2605.04613) — `confirmed`
- Molina et al., *Evaluation Framework for Automatic Singing Transcription*, ISMIR 2014（COnPOff 容差公式正確出處）
- Liang/Su et al., *Musical Offset Detection: Violin*, ISMIR 2015（offset 主觀性）

**G2P / 振假名（§7）**
- Hu et al. (Baidu), *Transcript-Prompted Whisper + Dictionary-Enhanced Decoding*, 2025 (arXiv 2506.07646) — `confirmed`
- Ehara, *Grade-Aware Kanji Reading Estimation in Browsers*, PACLIC 2025 — `confirmed`（分心數字修正）
- Yomikata (Passaglia 2023) — `uncertain`（gikun 非官方明言）
- Wikipedia *Furigana* / japanesewithanime (gikun 例) — `confirmed`
- PMC6476939, *Tokyo Vowel Devoicing Is Not Gradient*, Frontiers Psych. 2019 — 無聲化事實 `confirmed`；歌唱復聲歸因 `refuted`
- narabas (darashi 2024)；Japanese MFA dictionary v2.0.1a；PolySinger (arXiv 2407.14399)

**分離 / 曲風 OOD（§8）**
- *Exploiting MSS for ALT with Whisper*, 2025 (arXiv 2506.15514) — `confirmed`
- Cífka et al., *Jam-ALT*, ISMIR 2024 (arXiv 2408.06370) — `confirmed`
- MAJL, ACM MM 2024 (arXiv 2501.03689) — `uncertain`（病灶歸因/venue 修正）
- SDX'23 Music Demixing Track, TISMIR 2024
- *Perceptual Errors in MSS: looking beyond SDR averages*, ISMIR 2025
- Gómez/Bonada/Salamon, *Flamenco Transcription*, MTG-UPF/TASLP — `uncertain`（8pp 容差錯配）
- GTSinger, NeurIPS 2024 Spotlight (arXiv 2409.13832) — `confirmed`
- Yamamoto et al., *COSIAN* (ISMIR 2022) / *PrimaDNN'* (EUSIPCO 2023) — `confirmed`
- Lee & Oh, *Automatic Beat Alignment of Rap Lyrics*, Stanford CS229 2009 — `confirmed`（課堂單曲，引導性）
- *Sound and Music Biases in Deep Music Transcription Models*, EURASIP 2026 (arXiv 2512.14602) — `confirmed`（鋼琴 AMT）
