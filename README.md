# karaoke-jp

JOYSOUND-style Japanese karaoke video generator. **Personal use only — do not upload outputs.**

End-to-end: a YouTube URL or local audio in, a 1080p60 MP4 out with discrete
pitch bars, per-character lyric wipe, furigana ruby, and an optional
background image / video.

See [spec.md](spec.md) for the full design, [CLAUDE.md](CLAUDE.md) for the
working agreement, and [MEMORY.md](MEMORY.md) for decision history + landmines.

## Status (M1–M4 wired)

Two songs validated end-to-end: tuki. 「零-zero-」 (Official Audio bg) and
結束バンド「ギターと孤独と蒼い惑星」 (Lyric Video MV bg). Polish items
(M5 multi-volume mix, M6 batch fan-out, M7 Yomikata + per-song gikun
overrides) are deferred — see `spec.md` §5.

## Pipeline

```
YouTube URL
  └─ download ─► source.wav, background.mp4         (M0, yt-dlp)
       └─ separate ─► vocals.wav, instrumental.wav  (M1, melband-roformer-infer)
            └─ melody  ─► melody.mid                (M2, RMVPE f0 + local note segmentation;
                                                     SOME fallback)
            └─ tokenize + asr + align ─► aligned.json
                                                    (M3, fugashi + faster-whisper
                                                     + kana-aware NW)
                 ├─ midi_markers ─► melody_markers.mid (M4 prep)
                 ├─ export_lrc   ─► karaoke.lrc       (M4 prep)
                 └─ render       ─► karaoke.mp4       (M4, headless MID2BAR fork)
```

Each stage is a Snakemake rule. Cache-friendly: changing code only re-runs
downstream of what changed.

## Score-first melody (piano-only cover)

If you already have a trustworthy score, do **not** ask RMVPE / BasicPitch to
guess the melody from polyphonic piano audio. Use the score as pitch ground
truth and let audio decide timing only.

```bash
# 1. Install the DTW alignment extra in the main venv
~/venvs/karaoke-jp/bin/pip install -e '.[score]'

# 2. Export the score as MIDI
#    For true pitch-perfect output, export the melody staff / melody-only MIDI.
#    A full piano MIDI can still use --top-voice, but that is best-effort.

# 3. Align score MIDI to the piano recording
~/venvs/karaoke-jp/bin/karaoke-jp score-melody \
  songs/<song-id>/source.wav \
  --score-midi songs/<song-id>/score.mid \
  -o outputs/<song-id>/melody.mid
```

Useful flags:

- `--top-voice` (default): best-effort highest note per onset, useful when the
  exported score MIDI still contains both hands.
- `--all-notes`: keep the full score MIDI after DTW timing alignment.
- `--tempo 93`: force the tempo metadata embedded in `melody.mid`.

## One-command end-to-end (after setup)

```bash
# 1. Fetch a song
~/venvs/karaoke-jp/bin/python scripts/download_song.py \
  'https://youtu.be/<id>' -o songs/<song-id>/

# 2. Hand-write or scrape lyrics into songs/<song-id>/lyrics.txt
#    (see ~/.claude/projects/.../memory/jpop_download_lyrics.md for sources)

# 3. Run the whole pipeline
~/venvs/karaoke-jp/bin/snakemake -j 1 outputs/<song-id>/karaoke.mp4
```

Output: `outputs/<song-id>/karaoke.mp4`.

If the source is a "Lyric Video" upload (e.g. Aniplex anime tie-in
releases) where lyrics are burned into the video, pass `--no-video` to
`download_song.py` and provide a still image (`background.png`) instead —
otherwise the burned text will collide with our own lyric layer.

## Layout

```
karaoke-jp/
├── src/karaoke_jp/         # main package (separate, melody, ruby, align,
│                             lrc_export, midi_markers, cli)
├── scripts/                # one-off CLIs the venv-segregated stages call
├── Snakefile               # 8 rules: separate -> melody -> {tokenize,
│                             asr -> align} -> {export_lrc, midi_markers}
│                             -> render
├── songs/<song-id>/        # source.wav, lyrics.txt, source.md,
│                             background.{mp4,png} (gitignored audio/video)
├── outputs/<song-id>/      # all intermediates (gitignored)
├── third_party/            # external repos (NOT vendored — see README)
├── spec.md  CLAUDE.md  MEMORY.md
└── pyproject.toml  README.md  Snakefile
```

## Three runtime venvs (intentional)

| venv | owns |
|---|---|
| `~/venvs/karaoke-jp/`         | M0 download, M1 separate, M4 render-prep, click CLI |
| `~/venvs/karaoke-jp-melody/`  | M2 SOME inference (legacy librosa<0.10) |
| `~/venvs/karaoke-jp-lyrics/`  | M3 ASR + tokenize + align (faster-whisper + cuBLAS shim) |
| `~/venvs/karaoke-jp-render/`  | M4 MID2BAR-Player (Pygame + OpenCV) |

Setup commands for all four — including `third_party/SOME` clone and
checkpoint download — live in [`third_party/README.md`](third_party/README.md).
