# karaoke-jp

JOYSOUND-style Japanese karaoke video generator. **Personal use only — do not upload outputs.**

See [spec.md](spec.md) for the full design, [CLAUDE.md](CLAUDE.md) for the working agreement, and [MEMORY.md](MEMORY.md) for decision history.

## Quick start (development, GPU box)

```bash
# Bootstrap the dev env (uses uv if available, falls back to pip)
python3 -m venv ~/venvs/karaoke-jp
source ~/venvs/karaoke-jp/bin/activate
pip install -e '.[separation,render]'

# Run the CLI to see available commands
karaoke-jp --help
```

## Pipeline overview

```
input audio
  └─ separate ─► vocals.wav, instrumental.wav   (M1, melband-roformer-infer)
       └─ melody  ─► melody.mid                 (M2, SOME / parselmouth f0)
       └─ tokenize + asr + align ─► aligned.json (M3, fugashi + faster-whisper
                                                  + kana-aware NW)
            ├─ midi_markers ─► melody_markers.mid (M4 prep)
            ├─ export_lrc   ─► karaoke.lrc       (M4 prep)
            └─ render       ─► karaoke.mp4       (M4, headless MID2BAR fork)
```

Each stage is one Snakemake rule. Cache-friendly: re-running with new code
only re-runs downstream of what changed.

## Status (M1-M4 wired)

End-to-end pipeline working on the first test song (tuki. - 零-zero-).
1080p60 MP4 with discrete pitch bars + per-character lyric wipe + furigana
ruby + JOYSOUND-style background. Run the whole thing in one command:

```bash
~/venvs/karaoke-jp/bin/snakemake -j 1 outputs/<song>/karaoke.mp4
```

Three runtime venvs (see CLAUDE.md "環境隔離"); checkpoint downloads + setup
in `third_party/README.md`.

Polish items (M5/M6/M7) are deferred — see `spec.md` §5.
