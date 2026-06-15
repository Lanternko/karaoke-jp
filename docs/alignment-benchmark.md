# 歌詞對齊 benchmark（對人耳 gold）— 2026-06-14

各對齊模型在**逐行 timing**（line start/end）上的準確度,全部用**同一把尺**計分:
人耳優先 gold 的 human 行（`source != "machine"`),per song 與 pooled。

- 計分器:[`scripts/bench_aligners.py`](../scripts/bench_aligners.py)（自含,與 `scripts/eval_alignment.py` 逐毫秒一致）。
- gold:`data/alignment_gold/{chidori,haru-hikage,tuki-zero}.gold.tsv`（人耳優先;歌詞私有,只有此處的聚合數字公開）。
- 預測檔:`outputs/<song>/`（gitignored）;haru MMS 用 `tmp/haru_mms_fresh.json`（`outputs/haru-hikage/aligned_midi.json` 是 stale 的 10.9s 災難檔）。
- 重跑:`~/venvs/karaoke-jp/bin/python scripts/bench_aligners.py`

## Pooled — 87 條人耳行（chidori 37 + haru 33 + tuki 17）

| 模型 | start MAE | st median | st≤250ms | end MAE | line IoU med | 覆蓋 |
|---|---|---|---|---|---|---|
| **MMS-JA(canonical)** | **0.037s** | 0.031s | **100%** | 0.062s | 0.982 | 3 首 |
| SOFA + island 硬錨 | 0.177s | 0.046s | 83% | 0.309s | 0.925 | 3 首 |
| SOFA zero-shot | 0.857s | 0.130s | 59% | 0.417s | 0.891 | 3 首 |
| classic（Whisper 鏈） | 1.903s | 1.240s | 0% | 0.839s | 0.645 | 僅 tuki |

## 逐首 start MAE（看離群風險）

| 模型 | chidori | haru | tuki |
|---|---|---|---|
| MMS-JA | 0.029 | 0.043 | 0.045 |
| SOFA + island | 0.168 | 0.162 | 0.224 |
| SOFA zero-shot | 1.058 ⚠️ | 0.129 | 1.833 ⚠️ |
| classic | — | — | 1.903 ⚠️ |

## 「面對我的標註集」對照（同一個 MMS 預測,只換參考 gold）

`scripts/eval_human_vs_gold.py <markers> <old_gold> <pred>`;舊 gold = `tmp/<song>.gold.PREHUMAN.bak`。

| 歌 | 邊界 | 舊 gold（機器自衍生） | 人耳 gold |
|---|---|---|---|
| chidori | start | 0.056s（P90 0.188, bias −0.039） | **0.029s**（P90 0.056, bias −0.002） |
| chidori | end | 0.089s | **0.051s** |
| haru | start | 0.109s | **0.030s** |
| haru | end | 0.123s | **0.050s** |

換到人耳 gold,同一個 MMS 預測的誤差掉到 ~一半:殘差（~30ms start）就是**人耳標註雜訊地板**,
舊機器 gold（自 MMS 衍生、循環)才是先前所有分數的瓶頸。

## 判讀

1. **MMS-JA 壓倒性 canonical**:pooled start 37ms、100% 行 ≤250ms、IoU 0.98。對人耳 gold 已飽和。
2. **SOFA 結構性出局**:硬錨救回 ~5×(0.857→0.177)仍輸 MMS ~5 倍;zero-shot 在 chidori/tuki
   有秒級災難(分離人聲的間奏 bleed)。median 還行、離群會崩,production 不能賭。耳測三段 A/B 全否決,已收檔。
   詳見 [`benchmarks/SOFA` 的 RESULTS](../tmp/sofa-ourgold/RESULTS.md)。
3. **classic Whisper 鏈已死**:tuki 1.9s、0% 行 ≤250ms(ASR 幻覺時代;只剩 tuki backup,非公平三首對照)。

## Caveat

- n 小(3 首/87 行);chidori+haru 是 kara.moe 篩查過的乾淨 OOD,tuki 可能污染。
- classic 只有 tuki 一首。
- 同一首歌四個模型用**同一份**人耳 gold、同一計分器,故跨模型可比;跨歌難度不同,pooled 為 micro-average。

## 候選 OOD 篩查（kara.moe）+ 標註佇列 — 2026-06-14

擴 n 的瓶頸:每首歌要進 benchmark,得先有人耳 gold(我的標註集,人耳優先)。現有候選
（有 MMS 預測、無 gold）先過 kara.moe 污染篩查,決定值不值得花人力標。

方法:`GET https://kara.moe/api/karas/search?filter=<曲名>&size=5` → 讀 `infos.count`。
kara.moe = MMS 微調源 `karaoke-mugen-timings` 的**公開上界**;count=0 → 庫裡沒有 → 認證乾淨
OOD;count>0 → 可能在訓練快照裡 → 當 OOD 證據沒意義（與 SOFA「Kiritan 洩題」同型防灌水）。

| slug | 曲名 | 演唱 | kara.moe count | OOD 判定 |
|---|---|---|---|---|
| tuki-saitei-kaiwai | 最低界隈 | tuki. | **0** | ✅ 乾淨 OOD |
| aina-kakumei | 革命道中 | アイナ・ジ・エンド | 1（Dandadan OP2） | ❌ 可能污染 |
| kuzurinen | クズリ念 | ずっと真夜中でいいのに。 | 1（やにすう OP） | ❌ 可能污染 |
| byoushinwo-kamu | 秒針を噛む | ずっと真夜中でいいのに。 | 2 | ❌ 可能污染 |
| bocchi-guitar | ギターと孤独と蒼い惑星 | 結束バンド | 3 | ❌ 可能污染 |
| night-dancer | NIGHT DANCER | imase | cover 在庫 | ❌ 可能污染 |

對照(已 benchmark):chidori 0、haru 0 = 乾淨;tuki-zero 2 = 可能污染(見 [SOFA RESULTS](../tmp/sofa-ourgold/RESULTS.md)）。

**標註佇列**
1. ★ **最低界隈**(tuki-saitei-kaiwai)— 6 首中唯一乾淨 OOD。**四模型預測已全部 staged**(各 83 行):
   `aligned_midi.json`(MMS)、`aligned.sofa.json`(SOFA zero-shot)、`aligned.sofa_islands.json`
   (SOFA island)、`aligned_whisper_backup.json`(classic);`bench_aligners.py` 已接上並加 gold-presence
   guard(沒 gold 自動跳過)。**唯一缺口 = 人耳 gold**;把 `data/alignment_gold/tuki-saitei-kaiwai.gold.tsv`
   標好放進去,benchmark 一行指令就出全四模型。
   （備註:RMS VAD 對它 voiced 99%、分離 bleed 重,SOFA island 錨點預期偏弱——這是該歌的真實屬性。）
2. 其餘 5 首在 kara.moe → 標了只算 robustness,不算 OOD 證據,擴 OOD 不優先。

## 最低界隈 結果（逐行精確 gold,81 human,2026-06-15）

Kojek 逐行重標(82 句,近 1:1)→ DP 對到 83 行 → 81 human / 2 machine。**第一首打爆 MMS 的歌。**

| 模型 | start MAE | st median | st≤250ms | end MAE | start bias |
|---|---|---|---|---|---|
| MMS-JA | 1.071s | 0.497s | 27% | 1.135s | −0.726s |
| SOFA zero-shot | 11.867s | 0.442s | 32% | 1.241s | — |
| **SOFA +island** | **0.871s** | **0.153s** | **64%** | 0.806s | −0.385s |
| classic | 0.994s | 0.342s | 31% | 1.056s | −0.622s |

- **MMS 是三個可用模型裡最爛的**:start MAE / median / ≤250ms 全輸 SOFA+island 與 classic。canonical 在這首敬陪末座。
- 第二副歌 forced-align 完全脫軌(`aligned_midi.json` 一行 19.7s、5 行擠進 0.87s、一行 10.5s);
  分段 start median:前段 0.52s、中段 0.56s、**第二副歌 1.51s**、尾段 0.30s。
- **共享系統性偏移**:三模型 bias 全偏早(−0.4~−0.7s),人耳句首比模型晚約 0.5s → 絕對數字含一截共享 offset,
  **但相對排名穩**(SOFA+island bias 最小 + median 0.153 最小,最貼人耳)。median 是最乾淨比較:SOFA+island 0.15 << MMS 0.50。

**對全局的意義**:MMS「本 domain 無挑戰者」只在乾淨歌(chidori/haru/tuki 30–45ms)成立;**快速密集重複副歌會脆裂**。
pooled(4 首)SOFA+island MAE(0.512s)已微幅超過 MMS(0.536s),但 MMS median(0.071)仍最佳(易歌主導)。
**結論:MMS = 準但脆;SOFA+island = 在難 OOD 上更穩健。** markers/gold/recipe 存私有 `gold/tuki-saitei-kaiwai/`。
