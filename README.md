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
       └─ melody  ─► melody.mid                 (M2, RMVPE + SOME)
       └─ lyrics  ─► ruby.lrc                   (M3, whisper + SOFA + fugashi)
            └─ render ─► karaoke.mp4            (M4, MID2BAR-Player fork)
```

Each stage is one Snakemake rule. Cache-friendly: re-running with new code
only re-runs downstream of what changed.

## Status

M1 (vocal separation) — in progress. Higher milestones not yet wired.
