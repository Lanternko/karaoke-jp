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
        expand(str(OUT_DIR / "{song}" / "karaoke.mp4"), song=SONG_IDS),


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


# M3 stages run in the lyrics venv (faster-whisper + fugashi). The wrappers
# already expect that venv on PATH; rules call the scripts directly.
LYRICS_PY = "~/venvs/karaoke-jp-lyrics/bin/python"
LYRICS_LD = (
    "$HOME/venvs/karaoke-jp-lyrics/lib/python3.12/site-packages/nvidia/cublas/lib:"
    "$HOME/venvs/karaoke-jp-lyrics/lib/python3.12/site-packages/nvidia/cudnn/lib"
)


rule tokenize:
    """M3a: lyrics.txt -> tokens.json (fugashi + UniDic readings)."""
    input:
        lyrics=str(SONGS_DIR / "{song}" / "lyrics.txt"),
    output:
        tokens=str(OUT_DIR / "{song}" / "tokens.json"),
    shell:
        f"{LYRICS_PY} scripts/tokenize_lyrics.py {{input.lyrics:q}} -o {{output.tokens:q}}"


rule asr:
    """M3b: vocals -> ASR JSON (faster-whisper, prompted with lyrics head)."""
    input:
        vocals=str(OUT_DIR / "{song}" / "vocals.wav"),
        lyrics=str(SONGS_DIR / "{song}" / "lyrics.txt"),
    output:
        asr=str(OUT_DIR / "{song}" / "asr.json"),
    shell:
        f"LD_LIBRARY_PATH={LYRICS_LD}:$LD_LIBRARY_PATH "
        f"{LYRICS_PY} scripts/run_asr.py {{input.vocals:q}} -o {{output.asr:q}} "
        f"--lyrics {{input.lyrics:q}}"


rule align:
    """M3c: ASR + tokens -> aligned.json + ruby.lrc (kana-aware NW)."""
    input:
        asr=str(OUT_DIR / "{song}" / "asr.json"),
        tokens=str(OUT_DIR / "{song}" / "tokens.json"),
    output:
        aligned=str(OUT_DIR / "{song}" / "aligned.json"),
        lrc=str(OUT_DIR / "{song}" / "ruby.lrc"),
    shell:
        f"{LYRICS_PY} scripts/align_lyrics.py "
        f"--asr {{input.asr:q}} --tokens {{input.tokens:q}} "
        f"--aligned-out {{output.aligned:q}} --lrc-out {{output.lrc:q}}"


# M4 stages: lrc/marker prep run in the main venv (mido lives there), and
# the actual render runs in the dedicated MID2BAR venv (Pygame + opencv).
MAIN_PY = "~/venvs/karaoke-jp/bin/python"
RENDER_PY = "~/venvs/karaoke-jp-render/bin/python"


rule export_lrc:
    """M4a: aligned.json -> MID2BAR-flavored .lrc (centiseconds + @RubyN=)."""
    input:
        aligned=str(OUT_DIR / "{song}" / "aligned.json"),
    output:
        lrc=str(OUT_DIR / "{song}" / "karaoke.lrc"),
    shell:
        f"{MAIN_PY} scripts/export_lrc.py {{input.aligned:q}} -o {{output.lrc:q}}"


rule midi_markers:
    """M4b: melody.mid + aligned.json -> melody_markers.mid (page boundaries)."""
    input:
        midi=str(OUT_DIR / "{song}" / "melody.mid"),
        aligned=str(OUT_DIR / "{song}" / "aligned.json"),
    output:
        midi=str(OUT_DIR / "{song}" / "melody_markers.mid"),
    shell:
        f"{MAIN_PY} scripts/add_midi_markers.py "
        f"--midi {{input.midi:q}} --aligned {{input.aligned:q}} --out {{output.midi:q}}"


rule render:
    """M4c: headless MID2BAR render -> karaoke.mp4 (1080p60, h264 + aac)."""
    input:
        instrumental=str(OUT_DIR / "{song}" / "instrumental.wav"),
        midi=str(OUT_DIR / "{song}" / "melody_markers.mid"),
        lrc=str(OUT_DIR / "{song}" / "karaoke.lrc"),
    output:
        mp4=str(OUT_DIR / "{song}" / "karaoke.mp4"),
    shell:
        f"SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy "
        f"{RENDER_PY} scripts/render_mp4.py "
        f"--audio {{input.instrumental:q}} --midi {{input.midi:q}} "
        f"--lrc {{input.lrc:q}} --out {{output.mp4:q}}"
