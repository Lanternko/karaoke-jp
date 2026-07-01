# CE+CTC ∪ GAME note union — benchmark (2026-06-14)

實驗目標(survey ensemble 突破瓶頸)：CE+CTC 在 MIR-ST500 COnPOff 大勝 GAME(+14pp,
learned offset)。能否用 union 把 CE+CTC 的 offset 接到 GAME 的 coverage 上,破 offset 洞?
工具：`scripts/make_note_union.py`(coverage / offset_transplant / both)、
`scripts/eval_notes_rich.py`(誠實 shift=0、IOU、signed offset bias、over/under-seg)。

## 1. MIR-ST500 test (N=82, 中文分離人聲, MIREX 慣例 GT, onset 0.05)

| 系統 | COn | COnP | COnPOff | offset_bias | optimism gap |
|---|---|---|---|---|---|
| GAME seg03 (canonical) | .732 | .670 | .416 | **+55.9ms** | +1.9pp |
| CE+CTC (in-domain) | **.779** | **.728** | **.554** | −1.9ms | +0.6pp |
| **GAME + CE offset 移植** | .732 | .670 | **.504** | +0.3ms | — |
| GAME + CE coverage 補洞 | .733 | .669 | .415 | +55.6ms | — |
| CE + GAME coverage 補洞 | .771 | .719 | .548 | +0.6ms | — |

**發現**：(1) CE+CTC 中文 in-domain 三指標全勝 GAME。(2) **offset 移植把 GAME COnPOff
.416→.504(+8.8pp)**,offset_bias +55.9→+0.3ms,COn/COnP 不變 → GAME 的 MIR-ST500
offset 洞主要是「分離人聲上 GAME 把音拖太長」,可被 CE 的 learned offset 校正。
(3) coverage union 無效(CE coverage 已足)。(4) 舊 shift-sweep 灌水 GAME ~1.9pp(D2 坐實)。

## 2. Kiritan (N=50, 日文清唱 a cappella, 雙修正 GT gt_timefix, onset 0.05)

| 系統 | COn | COnP | COnPOff | offset_bias |
|---|---|---|---|---|
| GAME -l ja | .860 | .643 | **.499** | +15.4ms |
| CE+CTC | .861 | **.652** | .492 | +19.3ms |
| GAME + CE offset 移植 | .860 | .643 | .496 | +14.9ms |
| GAME + CE coverage | .861 | .644 | .499 | +15.4ms |

**發現**：清唱上 offset 移植 + coverage **全部無效**(.499→.496/.499)。根因：清唱無分離
artifact,GAME offset 本來就準(+15ms 非 +56ms),沒有 offset 洞可補;瓶頸在 COnP(pitch
+ GT 慣例,survey 已知 Kiritan GT 殘留移調)。→ **offset 洞是 domain-specific:分離人聲才有。**

## 3. chidori 自家 gold (分離 J-pop, 卡拉OK視覺 bar 慣例 gold, 相對比較, onset 0.1)

| 系統 | COn | COnP | COnPOff | off_bias |
|---|---|---|---|---|
| GAME union (canonical) | .533 | .426 | **.122** | +90.8ms |
| CE+CTC (prof) | .444 | .299 | .067 | −67.8ms |
| CE+CTC (melro) | .513 | .382 | .106 | −112ms |
| GAME union + CE offset 移植 | .533 | .426 | **.092 ↓** | −20.6ms |
| GAME union + CE coverage | .533 | .426 | .122 | +90.8ms |

**發現**：(1) chidori 上 GAME union > CE+CTC(coverage,CE 分離域 miss ~20%,與 2026-06-11
記憶一致)。(2) **offset 移植在 chidori 反而傷 COnPOff(.122→.092)** — 因為 chidori gold
是卡拉OK視覺 bar 慣例(音 held 到 bar 尾),GAME 的長 offset(+90ms)才貼 gold,CE 的短聲學
offset 離 gold 更遠。

## 結論(分類)

- **CE+CTC offset 移植**：MIR-ST500(MIREX)**強(+8.8pp)**、Kiritan 中性、chidori 產品 gold
  **負**。→ 它優化的是**文獻 MIREX 聲學慣例**,不是**卡拉OK視覺 bar 產品慣例**。
  **歸類：競爭對手/文獻基準洞見,不進 canonical**。直接坐實 survey §2:卡拉OK視覺 offset ≠
  MIREX 聲學 offset 是兩個目標,追 MIR-ST500 COnPOff 會傷產品。
- **CE+CTC 當骨幹**：贏 MIR-ST500(中文 in-domain)、平 Kiritan、**輸 chidori(分離日文 coverage miss)**
  → 不替換 GAME 當日文分離 J-pop 的 canonical 骨幹。
- **GAME 仍是產品正確骨幹**;其 MIR-ST500「offset 洞」部分是慣例 artifact(卡拉OK要長 offset)。
- **路線圖修正**：offset 改善要對**視覺慣例 gold** 評(Phase 0 母音尾+RMS),Kong learned offset 頭
  (T9)對產品只在用視覺慣例 gold 訓練時才有價值,否則僅文獻競賽用 → **降級為競爭對手線**。
  **升級 T5/T6:視覺慣例 gold 是 offset 改善的前提。**
