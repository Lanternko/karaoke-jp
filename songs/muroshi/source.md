# MyGO!!!!! — 無路矢 (むろし)

- **Artist**: MyGO!!!!!
- **Title**: 無路矢 (むろし)
- **Source URL**: https://www.youtube.com/watch?v=s3BTDeNKufQ (【Official Music Video】オリジナル楽曲)
- **Duration**: 3:46
- **Retrieved**: 2026-07-01

## Notes

- Source is an official MV (not a lyric video) — `background.mp4` safe to use as
  the portrait MV backdrop (AV1; `download_song.py` prefers avc1, renderer
  re-encodes to h264 regardless).
- **Call-response handling**: the lead sings the main melody; other members sing
  the 〈…〉 backing/echo lines, often simultaneously (e.g. the chorus repeats the
  whole lead line). Per the 2026-07-01 design decision we keep ONLY the lead
  lines in `lyrics.txt` (so MMS forced-align sees one clean vocal stream), and
  render the 〈…〉 lines as a dim bottom-strip caption keyed to each lead line —
  see `backing.json`. (Contrast haru-hikage, which kept backing lines in
  lyrics.txt and needed ctc_gap_fill; this song's overlapping repeats would
  misalign that way.)
- **Gikun (義訓)** go through `overrides/muroshi.json`, NOT inline parens —
  `ruby.py` does fugashi+override only and does not parse `地球(ほし)` inline:
  - 地球 → ほし   (every occurrence in this song)
  - 信号 → ことば (every occurrence)
  - 嚆矢 → こうし (bridge; rare kanji, override guards fugashi). Verify the
    token splits as one surface "嚆矢" in tokens.json; if fugashi splits it,
    patch per-char.
- 声 is read こえ in the lead line "投げ続けた声" (default, no override) but おと
  in the backing "その声(おと)で" — since 声=おと only lives in backing.json
  (plain text, no ruby in v1), the global override stays clean.
