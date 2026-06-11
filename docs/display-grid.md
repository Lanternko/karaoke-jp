# Display Grid — 標準化音高 bar 顯示系統

> Kojek 規格定稿（2026-06-11）。把「bar 怎麼畫」與「歌怎麼唱」解耦：
> bar 排在標準化的**顯示時間軸**上，renderer 用分段線性 warp 讓
> wipe/游標與真實演唱完全同步 — **只有游標速度在變**。
> 顯示專用：grid 輸出**永遠不可**餵給 eval harness。

## 視覺詞彙（全部固定）

| 元素 | 值 | 來源 |
|---|---|---|
| 四分音符寬度 | 恆定（頁寬 1760px ÷ 16 拍 = 110px/拍） | `--quarters-per-page 16` |
| 音符時值 | snap 至 2^-2 ~ 2^2 四分音符（log 距離最近） | `quantize()` |
| 音符 slot（v11） | **slot = 量化時值本身**，bar 畫 slot − gap — mora 縫住在音符自己的拍內（同樂譜排版），on-grid 演唱游標**恆速** | `layout_pages` |
| mora 間隙 | 恆 0.0625 拍（~7px）—「看得到就好」（0.25→0.125→0.0625 兩輪調降） | `--gap-units 0.0625` |
| 換氣間隙（v11） | 真實靜默 >0.25s → slot 間再插 0.5 拍 — **讀者看得出換氣點**，游標在真實換氣時掃過它 | `--breath-gap 0.25` / `--breath-units 0.5` |
| 換句間隙 | 恆 1.25 拍 | `--phrase-gap-units 1.25` |
| 每頁 | 固定 16 拍跨度，1–3 句（通常 2），塞不下整句下頁 | packing |
| 超大樂句切分 | **歌詞行邊界絕對優先**（aligned JSON），單行超頁才退回最大聲學空隙 | `split_oversized` |
| 長休息 | 上句結束 +0.5s 翻頁 → 游標 park 在左緣 → 首音前 4 拍開始掃（倒數） | `--count-in-quarters 4` / `--flip-delay 0.5` |
| 短句間翻頁 | 前句唱完即快翻（gap 的 25% 處），新頁音高可讀時間 ≈ gap 的 75% | quick-flip anchor |
| 人耳音高修正 | per-song sidecar，**只動顯示層**，eval 候選永不碰 | `--pitch-patch overrides/<song>_pitch_patch.json` |
| 變動項 | 游標速度只在**搶/拖拍與換氣**時變（on-grid 恆速 — 變速本身成了演唱偏差訊號） | warp |

## 資料流

```
melody_union 輸出（真實時間、未量化）
  → make_display_grid.py
      1. 歌詞窗 gating（margin 0.25 + tail allowance 1.0，救樂句尾延音）
      2. 碎音清理：dur < 0.09s 丟棄
      3. 抖音 wiggle 吸收：短音(≤0.22s)夾在同音鄰居間 ±2 半音 → 改標鄰居音高
      4. 時值量化（2^-2..2^2 拍）
      5. 樂句切分（真實空隙 >0.8s）+ 超大樂句遞迴切：行邊界優先、
         無行邊界才用最大內部空隙（副歌連唱整坨曾被切在 0.04s 小縫上）
      6. 固定頁打包 + 顯示座標排版（構造性零重疊，stats 監控）
         長休息加 flip/park/count-in 三段 warp 錨點（TV 式倒數）
  → 輸出兩件：display MIDI（顯示時間軸的音符+page markers）
              <out>.warp.json（real↔display 分段線性錨點，每音符兩個）
  → render_mp4.py --time-warp <warp.json>
      monkey-patch 只包 draw_notes / draw_now_bar（guard flag 防雙重 warp）：
      bar 區看到 interp(real→display) 的時間；歌詞/音訊/seekbar 維持真實時間。
      --time-warp 模式下跳過 _hide_notes_without_visible_lyrics（grid 已 gating）
```

## 一鍵命令

```bash
# 完整 GAME 顯示鏈（driver 內 flags 全 pinned）
~/venvs/karaoke-jp/bin/python scripts/run_game_chain.py \
  --vocals outputs/<song>/vocals.wav \
  --fallback-midi outputs/<song>/melody_markers.scorefix.mid \
  --f0 outputs/<song>/rmvpe_f0.npz \
  --aligned outputs/<song>/aligned_midi.json \
  --bpm-file outputs/<song>/melody_quantized.mid.bpm.txt \
  --pitch-patch overrides/<song>_pitch_patch.json \  # 有人耳修正 sidecar 才帶
  --out outputs/<song>/melody_markers.gamescore.mid
# → 同時產出 melody_markers.gamescore.warp.json

# render（三個 override 缺一不可）
SDL_VIDEODRIVER=dummy ~/venvs/karaoke-jp-render/bin/python scripts/render_mp4.py \
  --audio outputs/<song>/mixed.wav \
  --midi  outputs/<song>/melody_markers.gamescore.mid \
  --lrc   outputs/<song>/karaoke.lrc \
  --out   outputs/<song>/karaoke_grid.mp4 \
  --background songs/<song>/background.mp4 \
  --assets render_assets/assets_flat.json \
  --app-settings render_assets/settings_wide.json \
  --time-warp outputs/<song>/melody_markers.gamescore.warp.json
```

## Bar 外觀（TV 風格 flat skin）

- 生成器：`scripts/make_flat_bar_skin.py`（tracked、可重現）→ 寫入
  gitignored 的 `third_party/MID2BAR-Player/images/bar/8_flat/`，
  **新機器要先跑一次**。樣式常數在腳本頂部：細身（BODY 100–200/300）、
  微圓角 16、未唱純白／唱過純黃、glow 全透明。
- **PADDING 契約（v9 修正）**：`draw_stretchable_rounded_rect` 把 sprite 往外
  推 `bar_padding`(=100) 再 blit，**cap 最外 100px 必須全透明** — bar 本體只能
  畫在 `[PADDING, 3*SEG_W-PADDING]`。違反契約每根 bar 兩側各多畫 ~22 螢幕px，
  把 27.5px 的 mora 間隙整個吃掉（v8「沒有間隙」的根因）。
  有單元測試把關（`test_flat_skin_respects_padding_contract`）。
- `render_assets/assets_flat.json`（tracked）把全部 channel/type 指向 8_flat。
- `render_assets/settings_wide.json`（tracked）：**關鍵覆蓋** BAR_AREA_LEFT 80 /
  WIDTH 1760 — 原廠 settings.json 蓋成 1000px 是「右側死格線」幻覺的元兇。

## 演進史（為什麼長這樣）

v1 固定拍頁(每句重頭) → pack 變長頁(縮放跳動 5.4×) → v6 固定預算頁(1.85×)
→ v8 display grid（縮放恆定、間隙標準化、warp 同步）
→ v9（Kojek 驗收回饋輪）：sprite PADDING 契約修正（間隙真正可見）、
行邊界優先切分（1:13「酔った振り」/1:52「吹く青風」不再被腰斬）、
長休息 flip/park/count-in、mid2csv BPM 2dp 對齊
→ v10（第二輪驗收回饋）：mora 間隙 0.25→0.125 拍、短句間 quick-flip
（前句唱完即翻頁，預覽時間 ≈ 75% gap）、--pitch-patch 人耳修正 sidecar
（chidori 52.4s た Eb4→Eb3，鏡像句 140.3s union 本來就對是證據）
→ **v11（第三輪，Kojek 兩項設計）**：恆速 slot（gap 從「附加」改「內含」—
四分+gap vs 二分+gap 不是 2:1 的游標抖動根除）＋換氣間隙視覺分級
（>0.25s 真實靜默插 0.5 拍寬縫）；間隙 0.125→0.0625 拍。
中途實測否決：sprite 端內縮做間隙（縮放後剩 2px）、量化後直接渲染（吞音重疊）。

## 坑（修 bug 前先讀）

1. **顯示異常先查 `app_settings/settings.json` 的覆蓋值**，再懷疑自己的邏輯
   （BAR_AREA_WIDTH=1000 事件）。
2. render_mp4 的任何路徑選項**必須在 main 開頭 resolve**（os.chdir 之後相對
   路徑會重錨到 MID2BAR 目錄）— 已踩兩次。
3. 改檔案禁用 `s.index` 切片；一律 `assert s.count(anchor)==1` 的 replace
   — 空切片 + `replace('')` 會把檔案炸成碎片（同 session 踩過兩次）。
4. MID2BAR `draw()` 的 try/except 會吞 bar 區的一切例外 → 畫面異常但不報錯；
   懷疑時直接 print-debug 進 patch wrapper。
5. 自製 bar skin 必守 PADDING 契約（見上節）— 間隙/重疊異常先懷疑 sprite
   超界，再懷疑排版邏輯（v8 事件：排版數據 478/478 對有間隙，螢幕上全黏死）。
6. **mid2csv 把 tempo 轉 BPM 時 `round(_, 2)`** — grid 的 quarter 必須用
   2 位小數 BPM 推（`make_display_grid` 已內建），否則 warp 秒與 renderer 秒
   有 ~3e-5 相對漂移：歌中段 ~5ms，足以把 park 錨點推到 page marker 錯誤側
   →「間奏游標卡在前頁右緣」（v9 首渲事件）。回歸測試
   `test_bpm_2dp_survives_renderer_roundtrip` 把關。
