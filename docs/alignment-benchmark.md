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
1. ★ **最低界隈**(tuki-saitei-kaiwai)— 6 首中唯一乾淨 OOD,已有 MMS 預測(`outputs/tuki-saitei-kaiwai/aligned_midi.json`);**缺人耳 gold**(待標)。標完即多一個真正 OOD 的 MMS 數據點;要補齊四模型則另需跑 SOFA(zero-shot+island)與 classic。
2. 其餘 5 首在 kara.moe → 標了只算 robustness,不算 OOD 證據,擴 OOD 不優先。
