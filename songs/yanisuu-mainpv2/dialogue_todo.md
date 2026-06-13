# Dialogue / Lyric TODO

Keep these out of `dialogue.ass` until manually confirmed:

- 00:50 region: user heard `日替わりじゃなん〜`; auto-caption is unreliable here.
- 00:58 region: user heard `さて　おも〜ることがあったな　こいつら`; needs another listen.
- 01:01 region: user heard `山田チュン`; likely name/callout but needs confirmation.
- Lyric line 3 in `lyrics.txt`: current candidate is low confidence.

Hybrid render rule:

- `lyrics.txt`: sung lyrics only. This drives MMS forced alignment and pitch-bar karaoke.
- `dialogue.ass`: spoken dialogue only. Overlay after karaoke render with ffmpeg.
