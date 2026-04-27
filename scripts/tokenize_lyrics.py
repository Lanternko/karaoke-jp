"""CLI helper to tokenize lyrics.txt -> tokens.json."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running without `pip install -e .` in the lyrics venv.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import click

from karaoke_jp.ruby import annotate_lyrics, dump_json


@click.command()
@click.argument("lyrics_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "-o", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--override", "override_path", type=click.Path(dir_okay=False), default=None)
def main(lyrics_path: str, out_path: str, override_path: str | None) -> None:
    lines = annotate_lyrics(lyrics_path, override_path=override_path)
    dump_json(lines, out_path)
    total_tokens = sum(len(ln.tokens) for ln in lines)
    ruby_tokens = sum(1 for ln in lines for t in ln.tokens if t.reading and not t.kana_only)
    print(f"{len(lines)} lines, {total_tokens} tokens, {ruby_tokens} with ruby -> {out_path}")


if __name__ == "__main__":
    main()
