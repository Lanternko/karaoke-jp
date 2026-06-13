# 調性偵測（Key Detection）綜述 — 工具、基準、伴奏 vs 人聲

> Deep research（2026-06-13）整理。研究 scope：drop-in 開源工具為主、全曲調性+涵蓋轉調、
> 通用基準但加權流行/人聲。核心應用題：餵**混音 / 伴奏 / 人聲**哪個最準。
> 信心標記：**✓✓** 3-0 對抗式查證、**✓** 2-0、**◐** 一手來源已抓但本輪查證未完成
> （session limit 撞線，synthesis 步驟中斷）→ 採用前請複查。

---

## TL;DR（給我們 pipeline 的結論）

1. **我們「伴奏 > 人聲」的觀察是文獻定論**，不是巧合。對「調性/和弦」這種**和聲**任務,
   人聲旋律是最差的輸入(會把主音認成屬音),伴奏的和聲內容才帶調性資訊。✓✓
2. **但不必特地去餵「伴奏 stem」—— 混音本身就夠**。SOTA 神經 drop-in（madmom 的
   CNN）就是吃**整個混音**、零分離,而且是流行樂 SOTA。把伴奏 stem 加強只比混音多
   ~0.2pp(和弦辨識實測)。所以：**繼續餵混音(不要餵人聲),把力氣花在「換更好的方法」**。✓✓
3. 我們現在的「手刻 peak-PCP + Krumhansl-Schmuckler」**方向對、但是最弱的那一檔**。
   升級路線二選一：(A) **Essentia KeyExtractor**(一樣是模板法,但 HPCP 比手刻穩、profile 可選),
   (B) **madmom CNNKeyRecognitionProcessor**(流行樂 SOTA,full-mix CNN)。建議 A/B 兩個都跑我們 gold。
4. 我們看到的「主音↔屬音混淆」是這領域的**具名、系統性難題**：MIREX 評分專門給「五度錯」
   0.5 分的部分分,因為它太常見;連 SOTA CNN 在流行樂都還有 5.6–10.8% 的五度錯。✓✓

---

## 0. 我們自己的實測（2026-06-13，`scripts/eval_key.py`）

裝了 **Essentia**（`~/venvs/keytest`，py3.12 cp312 wheel 可裝）。madmom 在 py3.12 裝不動
（缺 `numpy.distutils`,且需 numpy<2 與 essentia 的 numpy2 ABI 衝突 → 要獨立 py3.11 env）→ 留 gated。

**(A) chidori gold（=E♭）MIREX 加權分數**：輸入正規化成 mono 44.1k PCM16 後（我們 peak-PCP 只吃
PCM16,float32 stem 直接餵會 fallback 到空 melody 回垃圾 "C" — render 餵 pcm16 mixed.wav 才沒爆,
**這是 `_audio_pcp` 的真實脆弱點,待修**）：

| 輸入 | peak-PCP（現有） | Essentia KeyExtractor |
|---|---|---|
| source（混音） | E♭ **1.00** | E♭ **1.00** |
| mixed（render 現用） | E♭ **1.00** | E♭ **1.00** |
| instrumental（伴奏） | E♭ **1.00** | E♭ **1.00** |
| **vocals（人聲）** | B♭ **0.50** | B♭ **0.50** |

兩法在 chidori **打平 0.875** → **輸入軸**（人聲爛、混音/伴奏滿分）一首就分得出,但**方法軸**
（peak-PCP vs Essentia vs madmom）要 ~10–20 首 gold 才分得出高下。

**(B) 12 首一致性分析（Essentia,伴奏 vs 人聲,不需 gold）**：人聲與伴奏一致 **6/12**;
**不一致的 6 首全部落在「五度」或「關係調」**——

| 歌 | 伴奏 | 人聲 | 型態 |
|---|---|---|---|
| chidori | E♭ | B♭ | +五度(屬) |
| stay | C♯ | A♭ | +五度(屬) |
| aina-kakumei | Em | G | 關係調 |
| byoushinwo-kamu | C♯m | E | 關係調 |
| night-dancer | Gm | B♭ | 關係調 |
| (bocchi/eric/haru/kuzurinen/tuki×2 = 一致) | | | ✓ |

→ **人聲只要偏,就偏成五度或關係調**(缺和聲脈絡時旋律的兩種典型錯),系統性坐實「key 不走人聲」。

**(C) 外部基準 GiantSteps Key（101 首,JKU 鏡像下載,真實 ground truth）**：速度 + 正確率同時量。
GiantSteps 是 EDM(曲風偏離我們的 pop,正確率絕對值僅供工具間相對比較)：

| 工具 | MIREX 加權 | exact% | 秒/首(2-min) |
|---|---|---|---|
| peak-PCP（我們手刻） | 0.546 | 40.6% | 0.079 |
| **Essentia KeyExtractor（預設 profile）** | **0.717** | **64.4%** | 0.070 |

→ **方法軸翻盤定案**：chidori 單首打平是假象,**真實 n=101 上 Essentia 大勝**（加權 +0.17、exact +24 分）
**而且更快**。兩者皆 <0.1s/首 → **沒有工具「太慢」,不需淘汰**。madmom 裝不動（py3.12）沒測到。

**(D) 已採用（2026-06-13）**：Essentia 接成 HUD `_detect_key` 的 **primary**（in-process,
`~/venvs/karaoke-jp-render` 已 `pip install essentia`,numpy 維持 2.4.4 不受影響;**import 失敗自動
fallback peak-PCP → melody**,所以非硬依賴）。`essentia.log.warningActive/infoActive=False` 關掉它
每次呼叫噴的 "No network created" log。預設 profile 即用;**profile 調校（temperley/krumhansl vs
EDM 的 edma/bgate）留待在 pop gold 上做**——不在這份 EDM 集上選（會選出 EDM-best 誤導 pop）。
已知:Essentia 在**短片段**會選關係小調（75s chidori 片段 → Cm,全曲 → E♭),render 餵全曲故顯示 E♭。

**(E) 外部 gold 驗證 + profile sweep（2026-06-13）**：web agent 查 11 首標準調性（chord/譜 源 > Tunebat 算法源）。
6 首**高信心 gold**（chord/sheet 推導:chidori E♭、night-dancer Gm、tuki-zero G、eric-chou A、
tuki-saitei Gm、stay D♭）—— **外部 chord 源獨立確認 chidori=E♭**（D 把位 + Capo 1 → 響 E♭),不靠 repo gold。
Profile sweep（餵 source mix）：

| profile | 高信心(6) | 全部(10) |
|---|---|---|
| **default（canonical 現用）** | **1.000** | **0.930** |
| bgate | 1.000 | 0.930 |
| krumhansl | 0.917 | 0.810 |
| shaath | 0.883 | 0.790 |
| temperley / diatonic | 0.767 | 0.720（small key → 相對大調） |

→ **canonical 的 default profile 已是最佳（高信心 gold 全中），不需改**。全部(10) 的 0.07 缺口在
medium 信心的 Tunebat 算法 gold（可能 gold 本身相對翻),非 essentia 之過。

**(F) madmom SOTA 對照（2026-06-13,終於跑成）**：madmom 0.16.1 在新 Python 三連雷
（`numpy.distutils` py3.12 沒、`collections.MutableSequence` py3.10 移除、`np.float` numpy1.24 移除）
→ 唯一可行組合 **uv 裝 py3.9 + numpy 1.23.5 + scipy 1.10 + cython<3**（`~/venvs/madmom`,**py3.9
與 render venv 的 py3.12 不相容,只能獨立 venv / subprocess**）。三方對照:

| 工具 | GiantSteps(EDM,n101) | pop 高信心(6) | pop 全部(10) | 速度 |
|---|---|---|---|---|
| peak-PCP | 0.546 / 40.6% | — | — | 0.079s |
| **Essentia（現用）** | 0.717 / 64.4% | **1.000** | **0.930** | **0.070s** |
| madmom CNN（SOTA） | **0.766 / 70.3%** | **1.000** | 0.860 | 1.78s（**慢 25×**） |

→ **madmom 確實是 SOTA**（GiantSteps EDM 上 0.766 > essentia 0.717,坐實論文宣稱),但
**在我們的 pop 高信心 gold 上與 essentia 並列 perfect 1.000**;全部(10) 反而輸（byoushin/kuzurinen
翻相對調,但那是 Tunebat 算法 gold 不一定對）。**結論:不採用 madmom** —— pop 上跟 essentia 打平、
慢 25–50×、且 py3.9 環境與 render venv 不相容（要 subprocess 增複雜度）、安裝脆弱。**Essentia 維持 canonical。**
**重要澄清**：先前「essentia 翻相對大調」的疑慮（night-dancer→B♭ 等）是**餵錯輸入**（peak-PCP 或 vocals stem）的產物;
餵 full mix 的 essentia default 對 6 首高信心 gold **全中** —— 再次坐實「餵 mix 不餵人聲」。
（sweep 腳本 `tmp/keytest/profile_sweep.py`,scratch。）

---

## 1. Drop-in 開源工具比較

| 工具 | 方法 | 語言/API | License | 流行樂基準(MIREX 加權) | 論文 |
|---|---|---|---|---|---|
| **madmom** `CNNKeyRecognitionProcessor` | 全曲 CNN（log-mag log-freq spectrogram, 65–2100Hz, 24 bins/oct）→ 24-way softmax,**吃 full mix,無分離/無 chroma** ✓✓ | Python | code BSD;**bundled model CC BY-NC-SA(非商用)** — 自用 OK | Billboard **85.1**、Isophonics **82.5**、KeyFinder **76.1**、GiantSteps(EDM) 74.6 ✓✓ | Korzeniowski & Widmer 2018(ISMIR)[1][2];前身 2017[3] |
| **Essentia** `KeyExtractor` | HPCP（諧波音高類別輪廓）+ `Key` 模板相關 = **Krumhansl-Schmuckler 衍生**(我們現在做的同一族,但 HPCP 更穩) ✓✓ | Python/C++ | AGPLv3(+商用授權) | profile 可選:`krumhansl`/`temperley`/`shaath`/`edma`/`bgate`… ✓✓ | Temperley 1999、Gómez 2006;EDM profile = Faraldo 2016/17[4] |
| **libKeyFinder** | DFT→chroma→模板 | C++11(需綁定) | **GPLv3**(copyleft,自用可) ✓✓ | — | Mixed-In-Key 風格 |
| **key-cnn**(Schreiber) ◐ | CNN,**支援 local/global key** | Python(keras) | 開源 | 流行/古典皆測 | Schreiber & Müller 2019/2020[5] |
| **Deezer STONE / skey** ◐ | 自監督 tonality estimator(2025) | Python(torch) | 開源 | 新,號稱少標註 | arXiv 2501.12907[6] |
| QM Vamp Key Detector(參考) | chroma+模板 | Vamp plugin | 開源 | 在 2018 MIREX 被 CNN 全面超越 ◐ | — |
| Mixed In Key(僅參考) | 閉源商用 | — | 閉源 | — | — |

**重點**：流行/人聲樂上,**madmom CNN(full mix)是目前最強的 drop-in**,在 Billboard/Isophonics/
KeyFinder/Robbie Williams 上贏所有 genre-specialized 對手 ✓✓。Essentia 是「跟我們同款但工程更成熟」
的模板法升級。

---

## 2. 基準與評分

- **MIREX 加權分數**(多來源 3-0 確認 ✓✓)：`w = 對×1.0 + 五度×0.5 + 關係×0.3 + 平行×0.2 + 其他×0`。
  「五度錯」= 預測的主音是真調的五度(就是我們的主音↔屬音混淆),這條被特別給 0.5 分。
- **流行/人聲加權基準**(AllConv CNN 成績,皆 full mix)：
  McGill Billboard(流行/搖滾)**85.1** ✓✓、Isophonics(Beatles/Queen)**82.5** ✓✓、
  KeyFinder(流行)**76.1** ✓✓、Robbie Williams 81.2 ✓✓。EDM(GiantSteps)74.6、古典 96.6。
- **曲風會翻盤**(◐,MIREX 2018):CNN 在流行/EDM 第一,但**古典上輸給模板法**
  (MIREX2005 古典:模板 CG1 88.8 > CNN FK1 82.5)。→ 沒有單一最強工具,**要在自己曲風上量**。

---

## 3. 全曲 + 轉調

- 全曲調性:上述工具都做(我們的 HUD 主需求)。
- **轉調/局部調性**(J-pop 副歌升 key)：**Schreiber & Müller 的 local-key CNN(`key-cnn` repo)**
  是最接近 drop-in 的答案;配套資料集 **Schubert-localkey**(audiolabs-erlangen)[5] ◐。
  注意該資料集偏古典藝術歌曲,流行轉調要自己補標。
- 實務建議:先把**全曲調性**做穩(主需求),轉調當第二階段 —— 對 J-pop 終段升 key,
  可以「分段(verse/chorus/final-chorus)各跑一次全曲偵測」當廉價 local key,不必上 local-key 模型。

---

## 4. 核心應用題：伴奏 vs 人聲 vs 混音

**(a) 分離有沒有用?有,但很小,且只有「伴奏」方向有用。** ✓✓
和弦辨識實測(IEEE 11249321)：把分離出的 **`other`(器樂伴奏)stem 加強**再混回 → 全指標微升
(triads 75.52→75.72%);但**加強人聲或 bass 都打不贏原混音**(vocals 74.75% < 混音 75.52%)。
→ 直接證實「**伴奏的和聲內容才是調性訊號,人聲不是**」,但混音本身已含足夠伴奏資訊,
分離的增益只有 ~0.2pp。

**(b) 人聲/旋律單獨偵測的主音↔屬音混淆 = 具名難題。** ✓✓
「五度錯」是 MIREX 的正式錯誤類別,連 SOTA CNN 在流行樂都有 5.6–10.8% 五度錯。我們 melody-only
把 E♭ 認成 B♭ 正是這個 —— 旋律重壓五度(B♭),缺乏和聲脈絡(A♭)就無法定主音。**結論:key 不要走人聲/旋律。**

**(c) HPSS(諧波-打擊分離)前處理**:研究有觸及但本輪未取得可驗證數字 ◐。理論上去打擊樂能讓
chroma 更乾淨,但對「已是 full-mix CNN」的方法幫助有限(CNN 自己學會忽略打擊樂)。

**(d) 具體建議(我們 pipeline)**：
1. **輸入**:繼續用 **full mix**(不要換人聲)。我們現在就是這樣 —— 這部分本來就對。
   想壓榨最後 0.2pp 才考慮餵 `other`/accompaniment stem(我們有 Demucs/RoFormer,成本低)。
2. **方法**:把手刻 peak-PCP K-S 換成 **Essentia KeyExtractor(快、穩、同族)** 當新 baseline,
   同時 A/B **madmom CNN(流行 SOTA)**。CNN 很可能修掉 chidori 這種五度混淆(它學的是和聲脈絡,
   不只 PCP 峰值)。我們現有 harmony peak-PCP 是好的「第三方對照」。
3. **轉調**:第二階段再上 `key-cnn` local key,或先用「分段各跑全曲偵測」的廉價法。

---

## 5. 怎麼在我們自己曲庫上評估

1. 建小 gold set:chidori 已有教授+譜驗證 = **E♭**;再人耳/譜標 ~10–20 首(涵蓋大小調、有/無轉調)。
   格式 `songs/<id>` + `key: "Eb"`。
2. 三個輸入 × 三個工具的矩陣:{mix, accompaniment, vocals} × {現有 peak-PCP, Essentia, madmom},
   用 **MIREX 加權分數**評(對/五度/關係/平行)。預期:vocals 那行全崩、mix/accomp 相近、madmom 最高。
3. 我們在 render 時已順手印 `[hud] key guess` —— 加個 `scripts/eval_key.py` 把它跑成表即可。

---

## 來源

- [1] Korzeniowski & Widmer 2018, *Genre-Agnostic Key Classification with CNNs*,
  ISMIR — https://arxiv.org/abs/1808.05340 ; https://ismir2018.ircam.fr/doc/pdfs/7_Paper.pdf ✓✓
- [2] madmom key eval/模組 — https://madmom.readthedocs.io/en/latest/modules/evaluation/key.html ◐
- [3] Korzeniowski & Widmer 2017, *End-to-End Musical Key Estimation Using a CNN* —
  https://arxiv.org/pdf/1706.02921 ✓
- [4] Essentia KeyExtractor — https://essentia.upf.edu/reference/streaming_KeyExtractor.html ✓✓ ;
  Faraldo EDM key — researchgate 309018542 ◐
- [5] Schreiber & Müller 2020, *Local Key Estimation*(ICASSP)— audiolabs-erlangen ◐ ;
  資料集 Schubert-localkey — https://www.audiolabs-erlangen.de/resources/MIR/schubert-localkey ◐ ;
  `key-cnn` — https://github.com/hendriks73/key-cnn ◐
- [6] Deezer STONE(2025 自監督 tonality)— https://github.com/deezer/stone ;
  https://arxiv.org/html/2501.12907 ◐ ; skey — https://github.com/deezer/skey ◐
- MIREX Audio Key Detection — https://music-ir.org/mirex/wiki/2025:Audio_Key_Detection ✓✓ ;
  2018 results — https://www.music-ir.org/mirex/wiki/2018:Audio_Key_Detection_Results ◐
- 分離 stem 加強 → 和弦辨識(伴奏有用、人聲沒用)— https://ieeexplore.ieee.org/document/11249321/ ✓✓
- libKeyFinder — https://github.com/ibsh/libKeyFinder ✓✓

> ⚠ 本輪 deep research 的 synthesis/部分 verification 因 session limit(2:20pm 重置)中斷:
> 25 條挑了 16 條完成 3-0/2-0 查證,9 條「abstain」未完成(非駁回)。MIREX 2018 排行榜細節、
> madmom/Schreiber/Deezer 的 license 與確切數字標 ◐,採用前複查一手來源。可在限額重置後重跑補齊。
