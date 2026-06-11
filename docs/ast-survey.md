# 歌唱轉譜(AST)× 歌詞對齊 — 文獻 Survey 與 Benchmark 計畫

> 2026-06-11 起草。動機:教授指出方向性問題 — survey 不到位(Fu & Su ISMIR'19、
> Wang & Jang CTC/CE 等台灣 AST 主線全部漏掉)、不該從 pitch-level 起手、
> note 切分沒考慮 CTC 校準、有公開 benchmark 沒跑。
> 本文檔 = 補課 + 對照我們已有結果 + 可執行的 benchmark 計畫。

## 0. 教授批評 × 我們現狀的誠實對照

| 批評 | 對 / 部分對 / 需澄清 | 現狀 |
|---|---|---|
| 不該用 pitch-level(F0)轉樂譜,該直接 note-level 模型 | **對,且我們繞了遠路才到** — classic 鏈(RMVPE F0 → mora refit → 手工槓桿)花了一整輪,天花板 60.6%;後來 GAME(note-level)raw 即 69.2%。教授的方向判斷與我們的實測結論一致,但我們是「試錯撞到」而非「survey 導出」 | 生產鏈已是 note-level 為主(GAME union 72.8%),classic 鏈降級為補洞 fallback |
| note 切分該用 CTC(至少校準較強) | **對,未探索** — 我們的邊界全是啟發式(mora gate、RMS 端點修復)。Wang & Jang (TASLP'22) 的 CE+CTC 弱監督正是我們缺的「校準」途徑;我們的歌詞先驗(每行 mora 序列)其實是 CTC 對齊的天然 target | 未做。見 §4 行動項 |
| 關鍵論文沒用(Fu & Su '19、CTC/CE、近期 SOTA) | **對** — survey 是從工具(SOME/GAME/BasicPitch)出發,不是從文獻出發;台灣 AST 主線(NTU/中研院)整條漏掉 | 本文檔 §1–3 補齊 |
| 公開資料集 benchmark 沒跑 | **對** — 我們只有自建 gold(2 首 + 1 進行中),絕對值不可比文獻(handoff 早已自知:onset 慣例不同)。MIR-ST500 / ISMIR2014 可直接跑,我們的 mir_eval harness(`eval_note_metrics.py`)已存在,缺的只是資料集 + 跑 | 見 §5 benchmark 計畫 |

**我們已有、且站得住的部分**(回應時不必全盤自我否定):
- 評測哲學(譜面 gold、人耳仲裁、「guide≈F0 同向偏差不是獨立證據」)與 Fu & Su 的
  「note segmentation 是核心」結論相容
- mir_eval COn/COnP/COnPOff harness 已落地(`scripts/eval_note_metrics.py`),
  與 MIR-ST500 系列論文同一套指標 — 接上公開資料集即可比
- 兩首歌 + 雙鏈的 ablation 紀律(負結果照記)是合格的實驗習慣;缺的是外部效度

## 1. 教授點名文獻(精讀)

### 1.1 Fu & Su — Hierarchical Classification Networks for Singing Voice Segmentation and Transcription(ISMIR 2019)

- 問題形式:**onset/offset 偵測 = 階層式狀態分類**。狀態空間
  S(silence)/ A(activation)/ T(transition),T 再分 O(onset)/ X(offset),
  並顯式處理 **onset 與 offset 重疊**(連唱換音:A→T→A,T = XO)— 正是我們
  「mora 連唱邊界」的問題。
- 輸入表徵:頻譜差分(S⁺/S⁻)+ CFP 音高顯著度(spectrogram / GC / GCoS),
  ResNet 或 RNN-attention。
- 損失:階層式 taxonomy 上的多個 BCE(HCN1/HCN2),用 [S,A] activity 輔助
  抑制 T 類別不平衡。
- 結論:note segmentation 品質主導轉譜表現;與 pitch detection 整合後超越先前
  singing transcription。
- 對我們的啟示:我們的 absorb_shakuri / refine_boundaries 手工槓桿,本質是在
  無監督地逼近這個「transition 狀態分類」— 該用學的,不該用寫的。

### 1.2 Wang & Jang — Training a Singing Transcription Model Using CTC Loss and Cross-entropy Loss(IEEE/ACM TASLP 2022;github: york135/CTC_CE_for_AST)

- CE 管 frame-level onset 分類(強標註);**CTC 讓「只有音符序列、沒有時間戳」
  的弱標註資料也能參與訓練** — 解決譜面多、對齊標註少的根本瓶頸。
- 輸出 note-level:量化音高 + 非量化 onset/offset。
- 資料:MIR-ST500(訓練/評測)+ ISMIR2014(額外評測);指標 COn / COnP / COnPOff。
- 對我們的啟示:
  1. 我們手上每首歌都有「歌詞 → mora 序列」= 天然的弱標註序列;CTC 對齊
     (mora-token 對 frame)是句中 ownership 問題的有監督替代品
  2. 我們的 ad-lib(よ×3)失誤正是 CTC blank/repeat 機制天生能吸收的型態

## 2. Note-level AST 系譜(2014–2026,已調查;數字皆有來源,n/v = 未驗證)

### 2.1 主線一頁版

```
評測框架:Molina (ISMIR'14) — COn/COnP/COnPOff 定義 + ISMIR2014 資料集
古典 F0→切分:SiPTH (TASLP'15, 遲滯切分) · Tony/pYIN-notes (TENOR'15, HMM)
深度切分轉向:Fu & Su (ISMIR'19, 階層狀態分類) → VOCANO (ISMIR'21, +VAT 半監督)
              → Omnizart 工具箱
資料集時代:MIR-ST500 (Wang & Jang ICASSP'21, 500 首中文流行 + EfficientNet baseline)
弱監督:CE+CTC (Wang & Jang TASLP'22) · teacher-student 偽標 (Kum ICASSP'22)
表徵時代:SSL frontends (Yamamoto'23: MERT/wav2vec2) · 音視覺 (Gu'23/24)
         · 音素資訊 (Yong ICASSP'23 — ISMIR2014 最佳 COnPOff 0.77)
物件偵測:MusicYOLO (ICASSP'22/TASLP — 頻譜圖上 YOLO 框音符)
SVS 標註線:ROSVOT (ACL'24) → SOME/GAME (openvpi, 無論文無基準)
            → SongTrans/STARS/VocalParse('24–'26, 用 WER/MAE 不用 COnPOff)
Token seq2seq:PerceiverTF · YourMT3+ · T3MS (TASLP'25)
分離預訓練:Mel-RoFormer (ISMIR'24, ByteDance) — MIR-ST500 現任 SOTA
日文線(京都–AIST):Nishikimi Bayesian/HSMM 系列 → CRNN-HSMM (APSIPA Trans'21,
            RWC 日流行 61 首, audio-to-SCORE) → Deng tatum-CTC (APSIPA'23)
```

### 2.2 MIR-ST500 test 計分板(COnPOff F1,50ms/50cents)

> **我們的實測已落地(2026-06-11,N=82,GAME raw `-l zh`,零後處理,zero-shot)**:
> COn **0.732** / COnP **0.655** / COnPOff **0.411**。詳見
> `benchmarks/mir-st500/RESULTS.md`。注意這是 82/100 subset，不是完整官方
> test 成績；18 首 YouTube 音訊需補 cache 後重跑。方法學上最接近的是
> **ROSVOT zero-shot(0.721/0.659/0.474)— 我們 onset 同級、offset 輸**
> (GAME 截延音尾,
> 正是 melody_union 要補的;offset 條款 max(50ms,0.2dur) 罰最重)。
> ✻ 以下監督式全部在 400 首 train split 訓過,GAME 從沒看過 MIR-ST500。

| 系統 | 年 | COn | COnP | COnPOff |
|---|---|---|---|---|
| **GAME raw -l zh(本研究,zero-shot,82/100 subset)** | 2026 | **73.2** | **65.5** | **41.1** |
| **CE+CTC ckpt 重跑(本研究,同分離前端,N=82)** | 2026 | **77.9** | **72.8** | **55.4** |
| ROSVOT(zero-shot,唯一可比) | 2024 | 72.1 | 65.9 | 47.4 |
| EFN(資料集 baseline ✻) | 2021 | 75.4 | 66.6 | 45.8 |
| JDC_note(teacher-student) | 2022 | 76.2 | 69.7 | 42.2 |
| **CE+CTC(教授點名)** | 2022 | n/v | n/v | **57.4** |
| wav2vec2-L(Gu) | 2023 | 78.3 | 70.7 | 52.4 |
| MusicYOLO-I | 2023 | n/v | n/v | 58.6 |
| ROSVOT(zero-shot,中文訓) | 2024 | 72.1 | 65.9 | 47.4 |
| **Mel-RoFormer-large(SOTA,98/100 首)** | 2024 | **81.9** | **79.8** | **62.5** |
| T3MS(完整 100 首最佳) | 2025 | 80.6 | 77.1 | 61.0 |

補充 ablation:GAME raw 後加 `--extend-sustains` 在同一 82 首上 **COnPOff
0.411→0.409**(COn/COnP 不變)。這確認「顯示用延音尾」不能替代 learned
offset calibration;教授的 CE/CTC 邊界監督建議仍是正路。

ISMIR2014 線:Ryynänen 30.8 → Tony 52.0 → Fu&Su 59.4 → VOCANO 68.4 →
**Yong'23 77.3**(注意:Yong 是 pYIN F0+學習式切分的「管線」— 見 §2.5 反證)。

### 2.3 Kiritan(日文清唱,internal MIR-style)

> **我們的實測已落地(2026-06-11,N=50,GAME raw `-l ja`,zero-shot,清唱免分離)**:
> COn **0.793** / COnP **0.531** / COnPOff **0.400**。詳見
> `benchmarks/kiritan/RESULTS.md`。這不是公開 leaderboard,而是我們把
> MIR-ST500 同款 evaluate.py 套到 Kiritan `midi_label` 的 internal protocol。
> WAV 含長尾 `pau`;評測用全檔推論,所以 silence false positives 自然計入 precision。

| 系統 | 條件 | COn | COnP | COnPOff |
|---|---|---:|---:|---:|
| GAME -l ja(raw GT) | 清唱,zero-shot,N=50 | **79.3** | 53.1 | 40.0 |
| **GAME -l ja(移調校正 GT)** | 同上,GT 用 RMVPE 校正 | **79.4** | **57.9** | **43.8** |

**2026-06-12 更新:GT 第二缺陷(per-song 時間平移 5/50 首,+150~340ms,
RMVPE 交叉相關獨立確認)修正後,雙模型同框**:
GAME -l ja **COn .862 / COnP .644 / COnPOff .502**;
CE+CTC(中文訓,zero-shot)**.860 / .652 / .492** — 日文清唱上幾乎打平。
MIR-ST500 上 CE+CTC(in-domain)則大勝 GAME zero-shot(COnPOff .554 vs .411,
同分離前端重跑,與其發表值 .574 一致)。GT 慣例敏感度:±30ms 標籤平移使
兩系統 COnPOff 各擺 ±4pp 且方向相反 — 50ms 容差下標註慣例是一階項。

**深度診斷(關鍵):COn→COnP 暴跌不是 GAME 弱,是 Kiritan GT 有 per-song
半音移調錯誤。** 證據鏈:(1) 全域調音 +2.0 cents(沒偏);(2) 八度錯只 1 顆;
(3) −1 半音錯**集中**(13/50 首 >40% flat、30/50 首 <10% flat = 雙峰,per-song
而非 model 性質);(4) 用**獨立的 RMVPE F0**(非 GAME)逐首算 median(GT−實唱 F0):
**34 首 offset 0、14 首 +1、各一首 +3/−2 → 16/50 首 GT 與錄音差移調**;
(5) song 08(最嚴重)抽查:**GAME vs RMVPE 中位 +0.00、GT vs RMVPE +0.88** —
GAME 完全對齊實唱音高,是 GT 標高了。校正那 16 首後 COnP 0.531→**0.579**。
殘餘 COn→COnP gap = 「測量 vs 樂譜意圖」(歌手唱平、Melodyne GT snap 到意圖音)—
正是教授在意的 pitch-vs-note 區別,現在在公開資料集上量化了。**連學界 SVS
資料集都有標註對不上音訊的問題,被我們的 RMVPE 交叉驗證抓出來。**

### 2.4 日文線 = 京都–AIST(我們最該引而沒引的)

- Nishikimi et al.(APSIPA Trans. 2021)**CRNN-HSMM audio-to-score**:
  RWC 日本流行 61 首,輸出量化樂譜(MusicXML)— 與卡拉OK用例同構。
  關鍵數據:**integrated 15.2% vs「F0 估計後量化」cascade 34.2% 總錯誤率** —
  教授「不懂為什麼用 pitch level」的最強文獻彈藥。
- Deng, Nakamura & Yoshii(APSIPA 2023)**tatum-level CTC**:note-level CTC 的
  時值符號會錯位,把 CTC 字母表移到單調的 tatum 網格才穩 — 直接指導我們
  「CTC 校準」該怎麼做對。

### 2.5 三個誠實的 nuance(回應時用,避免被反問倒)

1. 「note-level 全面勝出」**不無條件**:ISMIR2014 現任最佳(Yong'23 0.77)其實是
   pYIN F0 + 學習式切分的管線;general token decoder(YourMT3+)在 50ms 容差下
   輸給專用模型;SOTA Mel-RoFormer 本身是「分離預訓練 + onsets-and-frames 頭」
   的混合體。可防守的命題是:**「學習式、聯合訓練的邊界建模」處處勝過
   「手工 F0 切分」** — 我們的 classic 鏈正是後者,GAME 屬於前者。
2. **plain CTC 定位的是符號不是邊界**(spiky/peaky;Yong'23 原話、Zeyer'21 證明)
   — 每個成功案例都把 CTC 配上邊界感知損失(CE 強標/重建損失/BCE 邊界頭)
   或搬到 tatum 網格。教授說「CTC 至少校準比較厲害」成立,但工程上要照這些
   配方做,不是裸 CTC。
3. **SOME/GAME 沒有任何已發表基準數字** — 學界(MIR-ST500/COnPOff)與 SVS
   標註線(M4Singer/WER/MAE)是兩個互不報數的世界。我們把 GAME(union)放上
   MIR-ST500 跑出 COnPOff,本身就是沒人做過的資料點。

## 3. 歌詞對齊方法線(2018–2025,已調查)

### 3.1 系譜與 SOTA 數字(JamendoLyrics EN,word-level)

| 系統 | 方法 | Mean AE | Median | PCO@0.3s |
|---|---|---|---|---|
| Stoller 2019 (ICASSP) | Wave-U-Net→char **CTC**,弱行級標註 | 0.82 | 0.10 | 85% |
| Vaglio 2020 (ISMIR) | Spleeter+BiLSTM CTC,**IPA 音素跨語言** | 0.37 | — | 92% |
| Demirel 2021 (ICASSP) | Kaldi+anchor 詞、`<NOISE>` 標籤 | 0.31 | 0.05 | 93% |
| Gupta/GGL (MIREX 19/20 冠軍) | TDNN-F,**音樂/靜音當顯式 phone** | 0.22 | 0.05 | 94% |
| Huang 2022 (ICASSP) | CTC+**音高 MTL**+**行邊界偵測進 Viterbi** | 0.23 | — | 94% |
| Durand 2023 (ICASSP, Spotify) | **對比學習**跨模態嵌入(無 CTC),87k 歌 | **0.15** | — | 92% |

定錨:現代 aligner 的 **median ≈ 40–50ms(≈1 video frame)**,mean 被少數
大錯拖高。MIREX 2024 多語 Jamendo v2 上最佳也只有 0.58–0.65s/73–89% — 未解問題。

### 3.2 對教授「CTC 校準較強」的回答:結構性成立,有一個必須引用的 caveat

- 成立的理由:forced alignment **以歌詞為解碼約束**,在 frame 級聲學後驗上找
  最大似然路徑;我們的 NW-on-ASR 是先讓 ASR 定稿、再做符號對符號 —
  聲學證據在那一步已經丟了。**2019 後沒有任何競爭力系統用 transcribe-then-text-align 當 timing 來源**。
- Caveat — **peaky CTC**:Zeyer et al. 2021(arXiv:2105.14849)證明 CTC 收斂到
  blank 主導的單尖峰後驗 → onset 被量化到尖峰、**offset 幾乎無意義**(正是
  我們延音尾問題的所在)。修法:**label-priors CTC**(Huang et al., ICASSP 2024,
  boundary error −12~40%,TorchAudio 已收錄)、行邊界感知解碼(Huang 2022)、
  F0 後處理校正 offset(FZZ, MIREX 2024)。

### 3.3 我們的失誤類型,文獻怎麼處理

| 我們的失誤 | 文獻機制 |
|---|---|
| ad-lib 重複(よ×3 不在歌詞) | **wildcard/star token**(torchaudio MMS `*`、FZZ `<*>` 行首尾)、SOFA matching mode(對齊最佳「連續子序列」,不必吃完全部 transcript)、CTC blank 天生吸收 |
| 行邊界 ownership / 延音尾 | Huang 2022 **行邊界偵測器進 Viterbi**;FZZ「長尾被提早 timestamp 截斷」→ **F0 後處理**;Gupta 長母音擴充詞典 |
| 間奏空隙 | Gupta **音樂/靜音 = 顯式 phone(分曲風)** — 其 MIREX 奪冠最被歸功的設計;VAD 先切段 |

註:我們的 line gate / RMS 尾修復其實**獨立重新發明了** Huang 2022 BDR 與
FZZ F0 修正的同型機制 — 方向沒錯,但沒引文獻、沒在公開基準上驗證。

### 3.4 日文現況 = 文獻空白(機會)

- **沒有任何已發表的日文複音歌曲 word/mora 級對齊 benchmark**(Jamendo MultiLang
  只有 EN/ES/DE/FR)。學界最近的是 LyricSynchronizer(Fujihara & Goto 2011,
  phrase 級)與 Songle/TextAlive(自動估計+群眾修正,方法未發表為基準)。
- 社群工具已可用:**yohane**(MMS CTC 對齊器 + 日文卡拉OK微調 ckpt
  `mms-300m-ForcedAligner-karaoke-ja-Latn`)、**SOFA 日文 checkpoint**(音素級+
  信心分數+matching mode)、Julius segmentation-kit(speech 模型,長音會劣化)。
- **定位機會**:我們的三首 line-gold(+人耳協議)可以整理成第一個日文
  mora 級對齊小型基準。

### 3.5 遷移路徑(對齊換骨計畫)→ **PoC 已完成,雙 checkpoint 複現**

兩條獨立實作同日落地,數字互相印證(句首 MAE,vs 我們 canonical):

| 歌 | canonical(gate+repair) | stock MMS_FA(`ctc_line_align.py`) | karaoke-ja 微調(`forced_align_mms.py`) |
|---|---|---|---|
| chidori | 0.176(w250 60%) | **0.053**(median .021,w250 95%) | **0.056** |
| tuki-zero | 0.106 | **0.035**(w250 100%) | 0.045 |
| haru-hikage | 0.106 | 0.170 | 0.149(2 離群=長間奏 re-entry) |

- **句首:CTC 在 2/3 首歌好 3 倍以上**,median 21–48ms = 文獻 SOTA 帶
  (Jamendo median 40–50ms);haru 的長間奏 re-entry 是殘存弱類
- **句尾:裸 CTC 全面提早(bias −0.5~−0.7s)— survey §3.2 的 peaky 預言
  逐字應驗**;微調版 + 我們的 line_end_repair 組合:chidori 句尾 0.103、
  tuki 0.110、haru 0.131 — RMS 尾修復正好是 offset 解藥
- 結構性勝利(微調版實測):ように/余所に 助詞有自己的聲學 span、
  16 秒吞掉的 Tu-tu-lu 段修到 0.7s — NW-on-ASR 做不到的型態
- **canonical 未動**;night-dancer `karaoke_mms_v10grid.mp4` 等 Kojek 耳測
  → 通過則把 timing 源切到 CTC 鏈(gate 降級為後驗檢查)
- 下一級仍是 SOFA 日文 ckpt(音素級+信心分數+matching mode)

## 4. 行動項:CTC 在我們管線的兩個落點(定稿)

1. **Note 邊界校準(教授的直接建議)**:
   - 短期:york135 CTC/CE 預訓練模型直推我們的 vocals,與 GAME union 在
     雙歌 gold + MIR-ST500 同框評測(同一 harness、同一容差)
   - 中期:照 **Deng'23 tatum-CTC** 配方(CTC 字母表 = 單調時間網格,
     不是音值符號)或 CE+CTC 配方(CTC 只吃弱標,邊界靠 CE/BCE 頭)做
     校準頭 — **不要裸 CTC**(peaky,offset 無意義,見 §2.5/§3.2)
2. **歌詞對齊換骨**(§3.5 細節):NW-on-ASR → CTC forced alignment
   (SOFA 日文 ckpt 或 MMS 日文卡拉OK微調版),ad-lib 用 star token /
   matching mode,offset 用 label-priors CTC 或 F0 後處理;
   line gate / RMS repair 降級為後驗檢查。

## 5. Benchmark 計畫(已調查定稿)

### 5.1 資料集現實(調查結論)

- **AST 標準基準 = MIR-ST500**(500 首中文流行、複音、note 標註,固定 split
  #1–400 train / #401–500 test)。協定:`mir_eval.transcription`,
  **onset 50ms / pitch 50 cents / offset max(50ms, 0.2×dur)**(MIREX 2020 變體用
  100ms,不可混比)。官方 baseline EfficientNet-b0 **COn 75.4 / COnP 66.6 /
  COnPOff 45.8**;目前查到最佳 T3MS(2025)**80.6 / 77.1 / 61.0**。
  取得:YouTube 重抓(~5% link 已死,作者可信箱提供 cache)。
- **ISMIR2014(Molina)**:38 段 19 分鐘,歷史通用基準;原站已死,需向作者要。
- **日文**:**沒有公開的日文複音 AST 基準、也沒有日文歌詞對齊基準**(雙重空白,
  兩個 agent 獨立確認)。最接近:**Tohoku Kiritan**(50 首清唱,`midi_label`
  note 事件,zunko.jp 註冊下載)、PJS(CC BY-SA,可商用)、GTSinger-JA
  (6.45h,作者自flag 標註仍在修)。jaCappella 只有譜無時間對齊。
- **歌詞對齊**:JamendoLyrics EN(20 首)/ MultiLang(EN/ES/DE/FR ~80 首)
  **連音訊直接下載**;指標 AAE / PCS / PCO@0.3s(`mir_eval.alignment`)。
  Hansen/Mauch 不公開且與 DALI 訓練集重疊(MIREX 自己警告僅供參考)。

### 5.2 執行計畫(優先序)

| # | 跑什麼 | 怎麼跑 | 產出 |
|---|---|---|---|
| 1 | **MIR-ST500 test(100 首)** | 已完成 GAME raw `-l zh` 82/100 subset + extend 負結果;下一步補 18 首 failed audio cache → 重跑 GAME union vs CTC/CE 公開 ckpt vs baseline 數字;`eval_note_metrics.py` 改成文獻設定(onset 50ms + offset 條款) | 第一個可比文獻的 COn/COnP/COnPOff;直接回應「沒跑 benchmark」 |
| 2 | **Kiritan(日文,50 首清唱)** | 已完成 GAME raw `-l ja` 直推(清唱免分離);同官方 evaluate.py / 50ms | 日文 internal protocol 第一個數字:COn 0.793 / COnP 0.531 / COnPOff 0.400 |
| 3 | **ISMIR2014** | 向作者取得後同跑 | 歷史可比點(Tony 0.66 / Omnizart 0.80 / SOTA 0.93 COn) |
| 4 | **JamendoLyrics MultiLang** | 對齊換骨(SOFA/MMS CTC)完成後跑 EN 子集 | 對齊數字進入文獻座標系(SOTA AAE 0.15–0.22s) |
| 5 | **自建日文對齊基準** | 把我們的 line-gold(3 首,擴到 ~10–20 首)整理成 Jamendo 式發布(word/mora 邊界 + AAE/PCO 協定) | 填補文獻空白 — 從「沒跑基準」翻轉成「貢獻基準」 |

注意:MIR-ST500 是中文 — GAME 用 `-l zh`;這同時測語言條件的效果(`-l ja` vs
`-l zh` 對照可以是 ablation)。

## 6. 給教授的回應要點(定稿)

1. **承認**:survey 順序錯了(從工具出發而非文獻出發);台灣 AST 主線
   (Fu&Su → MIR-ST500 → CE+CTC)與日文線(京都–AIST CRNN-HSMM、tatum-CTC)
   兩條都漏,是硬傷。本文檔已補(§1–3,全部一手來源)。
2. **收斂點**:我們的實測獨立得到了與文獻相同的結論 — pitch-level 鏈天花板
   60.6%、note-level(GAME)69.2%,與 Nishikimi'21 的 integrated 15.2% vs
   cascade 34.2% 同方向;方向已在兩週前修正(生產鏈已是 note-level 為主)。
   問題不在最終方向,在「用試錯代替了 survey」。
3. **CTC**:成立,但文獻共識是「CTC 定位符號、不定位邊界」(Yong'23、Zeyer'21)
   — 採納 tatum-CTC / CE+CTC 配方而非裸 CTC;對齊側直接換 CTC forced
   alignment(SOFA/MMS 日文),失誤類型(ad-lib、延音尾)各有現成機制。
4. **Benchmark 承諾**(§5):MIR-ST500(可比 baseline 45.8 / CE+CTC 57.4 /
   SOTA 62.5)+ Kiritan(日文)先跑;附帶貢獻 — GAME 系從無已發表基準數字,
   我們會是第一個把它放進 COnPOff 座標系的。
5. **轉劣勢為貢獻**:日文複音 AST 與日文歌詞對齊**雙雙沒有公開 benchmark**
   (兩個獨立調查確認)— 我們的人耳 gold 方法論 + line-gold 可整理為
   首個日文 mora 級基準,這是缺口不是短板。

## 7. 主要來源(一手,全部實際抓取驗證;完整清單在調查紀錄)

- Molina ISMIR'14(指標定義):archives.ismir.net/ismir2014/paper/000298.pdf
- Tony/pYIN-notes:tenor-conference.org/proceedings/2015/04-Mauch-Tony.pdf
- Fu & Su ISMIR'19:archives.ismir.net/ismir2019/paper/000111.pdf
- MIR-ST500 + EFN:github.com/york135/singing_transcription_ICASSP2021
- CE+CTC TASLP'22:github.com/york135/CTC_CE_for_AST(IEEE 9961922)
- VOCANO ISMIR'21:archives.ismir.net/ismir2021/paper/000036.pdf
- Yong ICASSP'23(音素資訊):arxiv.org/abs/2304.05917
- ROSVOT ACL'24:arxiv.org/abs/2405.09940
- Mel-RoFormer ISMIR'24(SOTA):arxiv.org/abs/2409.04702
- T3MS TASLP'25:arxiv.org/abs/2502.12438
- 京都–AIST 日文線:staff.aist.go.jp/m.goto/PAPER/APSIPATRANS202104nishikimi.pdf;
  eita-nakamura.github.io/articles/Deng_SingingTranscriptionByCTC_APSIPA_2023.pdf
- 對齊:Stoller arxiv.org/abs/1902.06797;Gupta arxiv.org/abs/1909.10200;
  Vaglio ISMIR'20;Huang arxiv.org/abs/2202.01646;Durand arxiv.org/abs/2306.07744;
  peaky CTC arxiv.org/abs/2105.14849;label-priors CTC arxiv.org/abs/2406.02560
- 工具:github.com/qiuqiao/SOFA;github.com/Japan7/yohane;torchaudio MMS_FA
- 資料集:MIR-ST500(同上);Kiritan github.com/mmorise/kiritan_singing;
  PJS arxiv.org/abs/2006.02959;GTSinger arxiv.org/abs/2409.13832;
  JamendoLyrics github.com/f90/jamendolyrics;MIREX 2024 alignment wiki
