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
            └─ melody  ─► melody.mid                (M2, SOME / parselmouth f0)
            └─ tokenize + asr + align ─► aligned.json
                                                    (M3, fugashi + faster-whisper
                                                     + kana-aware NW)
                 ├─ midi_markers ─► melody_markers.mid (M4 prep)
                 ├─ export_lrc   ─► karaoke.lrc       (M4 prep)
                 └─ render       ─► karaoke.mp4       (M4, headless MID2BAR fork)
```

Each stage is a Snakemake rule. Cache-friendly: changing code only re-runs
downstream of what changed.

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
