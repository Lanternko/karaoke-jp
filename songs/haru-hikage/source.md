# MyGO!!!!! — 春日影 (MyGO!!!!! ver.)

- **Artist**: MyGO!!!!!
- **Title**: 春日影 (MyGO!!!!! ver.)
- **Tie-in**: TV アニメ「BanG Dream! It's MyGO!!!!!」#7 挿入歌
- **Released**: 2023-11-01
- **Lyricist**: 織田あすか (Elements Garden)
- **Composer / Arranger**: 藤田淳平 (Elements Garden)
- **Source URL**: https://www.youtube.com/watch?v=W8DCWI_Gc9c (BanG Dream Channel☆ — 本編中映像)
- **Duration**: 4:20

## Notes

- Source video is anime in-show footage from MyGO ep 7 (CRYCHIC stage scene),
  not a lyric video. No burned-in lyrics overlay → safe to use as karaoke bg.
- Lyrics scraped from uta-net /song/345583/. Parenthetical backing-vocal
  lines (e.g. "せつなくて　いとおしい") are concurrent harmonies, kept
  in this lyrics.txt so the karaoke renders them. faster-whisper picks
  up two of them as kanji-form transcriptions ("切なくて愛おしい",
  "大切で怖くて") and align_lyrics fuzzy-matches them to the harmony
  lines; the other two ("しあわせで くるおしい", "うれしくて さびしくて")
  need scripts/ctc_gap_fill.py because Whisper's VAD drops them.
- "悴" (kajikamu = 凍えて手がこわばる) is uncommon — fugashi+UniDic should
  read it かじかんだ. Watch the ruby in karaoke.lrc.
