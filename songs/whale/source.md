# ヨルシカ (Yorushika) — Whale (くじら)

- **Artist**: ヨルシカ / n-buna
- **Title**: Whale
- **Source URL**: https://www.youtube.com/watch?v=F_cmjdKbWnU (ヨルシカ / n-buna Official)
- **Duration**: 4:24
- **Format**: portrait 9:16 karaoke
- **Lyrics**: Japanese + romanization (user-supplied); no Chinese translation
- **Retrieved**: 2026-07-01

## Notes

- **Background**: static illustration (user-supplied album/MV art — sleeping figure
  on a couch in a flooded room, fish + amber light). NOT the MV — audio-only
  download (`--no-video`). The still-image renderer path puts the sharp art in the
  MV band and a blurred/dimmed copy fills the whole 9:16 frame.
- **Line segmentation**: kept the user's granular phrase breaks (each short line
  its own subtitle, e.g. 柔らかに / 溶けた琥珀のよう). The song is slow and spacious,
  so rapid A/B flips are not a concern.
- **Romaji is an answer-key, NOT rendered**: `romaji.txt` is a 1:1 romanization used
  only to verify/correct furigana (`scripts/romaji_overrides.py`); the video prints no
  romaji strip. (`backing.json` removed — it is reserved for call-response vocals, which
  this song has none of.)
- **Readings verified** against romaji.txt via `romaji_overrides.py --dry-run`: fugashi
  got every reading right out-of-the-box — 胸びれ=むなびれ, 背びれ=せびれ, 尾びれ=おびれ
  (body-part+鰭 compounds; 胸 reads むな not むね), 微睡み=まどろみ, 木漏れ日=こもれび,
  潜って=もぐって. **No `overrides/whale.json` written** — the tool's lone candidate
  `自分 じぶん→じぶ` is a ん/を-boundary false positive, rejected.
- **Outro hand-correction** (`overrides/whale_pitch_patch.json`, KOJEK EAR-CONFIRMED
  2026-07-02): the final 白く微睡みながら (L38) is sung IN FULL over 231.3–236.5 s
  (~14189F) — the song's vocal ends there. MMS forced-align collapsed the line
  across the 33 s outro, and the stem's 246.9–251.1 s "second phrase" that two
  earlier passes kept (first to 253.3 s trusting RMVPE, then to 251.1 s trusting
  RMVPE+pYIN agreement) is accompaniment/hum bleed — it fooled BOTH trackers at
  up to −5 dB. The recut anchors the 8 chars to the L38 note plateaus
  (60|72, 70, 69, 67, 65, 65) following the L16 template (93.68–98.38, same
  melody sung in 4.7 s: 白=しろ, く=70, 微睡=まどろ on the 69 plateau, み=67,
  な/が on the first 65, ら=65 sustain). The first occurrence (L16) is untouched.
  Lesson recorded: for outro cutoffs, tracker agreement is NOT sufficient
  evidence — only lyrics + ear are truth.
- **Cursor glide**: `render_portrait._wipe_q` now sweeps the bar cursor across rests
  at constant real-time speed instead of teleporting to the next note (the blank gap
  stays). General renderer fix, not song-specific.
- **Mora-primary cleanup v2** (2026-07-02, `scripts/note_cleanup.py`, ear-confirmed
  examples): the first cleanup grouped by surface char despite its name. That missed
  `静（しず）`: し and ず shared one group, so ず stayed 67→69. v2 expands the reading
  into true morae, partitions multi-mora kanji by pitch coherence, and defaults nearby
  variants (≤3 st) to one dominant pitch per mora. Wide melisma remains only with
  supported plateaus. It also catches locally isolated +12 errors (129.36 s 70→58),
  drops `波に横たえながら`'s p68 spill (168.56–169.67), and treats lyrics as truth:
  an orphan outside all mora windows drops even if RMVPE and pYIN both follow the
  accompaniment. Final bars end at 236.5 s with ら's 65 sustain (ear-confirmed).
  Sidecar grid: `outputs/whale/portrait_grid.mora-primary.json`.
- **Mora-primary dominance is tracker-weighted** (v3): raw duration picked く's
  69 drift (0.42 s slide) over its sung 70 attack (0.39 s, both trackers pinned,
  L16 template agrees) by 0.03 s. Dominant plateau = duration × tracker support
  when sidecars are armed; also flips そ @227.4 and さ @219.9 to their supported
  attack pitches. Two more local-island octave fixes surfaced (61.55/64.28 65→53,
  same signature as the ear-confirmed ら @129.36 70→58): ear-check pending.
- **Slow song → narrower bars**: portrait grid built with `--quarters-per-row 12`
  (default 8). Whale's sparse long notes filled whole rows at 8 q; 12 q shrinks
  every bar by a third. Per-song render parameter, set in the grid command.
