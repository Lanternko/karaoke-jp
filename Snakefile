# karaoke-jp Snakemake pipeline.
#
# Usage:
#   snakemake --rerun-triggers params input code -j 1 outputs/<song>/karaoke.mp4
#
# At M1, only the `separate` rule is wired. Higher rules become real as
# milestones land.

from pathlib import Path

SONGS_DIR = Path("songs")
OUT_DIR = Path("outputs")

# Discover song IDs by scanning songs/<id>/source.* present.
SONG_IDS = sorted(
    p.parent.name
    for p in SONGS_DIR.glob("*/source.*")
    if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}
)


def source_for(song_id):
    """Return the first matching source file for the given song id."""
    for ext in (".wav", ".mp3", ".m4a", ".flac"):
        p = SONGS_DIR / song_id / f"source{ext}"
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"No source audio for {song_id}")


rule all:
    input:
        expand(str(OUT_DIR / "{song}" / "melody.mid"), song=SONG_IDS),


rule separate:
    """M1: Mel-Band-RoFormer vocal separation."""
    input:
        audio=lambda wc: source_for(wc.song),
    output:
        vocals=str(OUT_DIR / "{song}" / "vocals.wav"),
        instrumental=str(OUT_DIR / "{song}" / "instrumental.wav"),
    params:
        out_dir=lambda wc: str(OUT_DIR / wc.song),
    # `:q` shell-quotes path placeholders so song ids / source paths
    # containing spaces (or anything else the shell might split on) survive.
    shell:
        "karaoke-jp separate {input.audio:q} -o {params.out_dir:q}"


rule melody:
    """M2: vocals -> melody MIDI via SOME."""
    input:
        vocals=str(OUT_DIR / "{song}" / "vocals.wav"),
    output:
        midi=str(OUT_DIR / "{song}" / "melody.mid"),
    shell:
        "karaoke-jp melody {input.vocals:q} -o {output.midi:q}"
