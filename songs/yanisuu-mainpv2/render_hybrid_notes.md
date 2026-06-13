# Hybrid Render Notes

Target behavior:

1. Render karaoke normally from `lyrics.txt`.
2. Keep note bars visible only during lyric windows.
3. Overlay `dialogue.ass` on the rendered video for non-sung dialogue.

Expected final command after karaoke render:

```bash
ffmpeg -y \
  -i outputs/yanisuu-mainpv2/karaoke.<variant>.mp4 \
  -vf "subtitles=songs/yanisuu-mainpv2/dialogue.ass" \
  -c:a copy \
  outputs/yanisuu-mainpv2/karaoke.<variant>.hybrid.mp4
```

Do not put dialogue into `lyrics.txt`; that would make MMS/GAME treat speech as sung lyrics.
