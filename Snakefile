# karaoke-jp Snakemake pipeline (M1-M4 wired).
#
# Usage:
#   snakemake --rerun-triggers params input code -j 1 outputs/<song>/karaoke.mp4
#
# DAG: separate -> melody -> {tokenize, asr -> align} -> {midi_markers,
#      export_lrc} -> render. Each stage runs in its own venv (see
#      CLAUDE.md "single source of truth"). Rules invoke the venv binary
#      by absolute path so reproducibility does not depend on the caller's
#      PATH or which venv they activated first.

import glob
import os
from pathlib import Path

SONGS_DIR = Path("songs")
OUT_DIR = Path("outputs")

# Set MELODY_BACKEND=cectc to use direct CTC+CE note transcription
# (Wang & Jang TASLP 2023). Default rmvpe = frame-level F0 + heuristic
# segmentation. cectc skips segment_f0_to_notes entirely; preferred for
# low-register vocals where RMVPE octave-halving bites.
MELODY_BACKEND = os.environ.get("MELODY_BACKEND", "rmvpe")

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


# Per-stage venv binaries (CLAUDE.md "single source of truth"). Pinning by
# absolute path avoids "depends on caller's active venv" surprises that
# bare-name commands like `karaoke-jp` or `python` have.
KARAOKE_BIN = str(Path.home() / "venvs" / "karaoke-jp" / "bin" / "karaoke-jp")
MAIN_PY = str(Path.home() / "venvs" / "karaoke-jp" / "bin" / "python")
LYRICS_PY = str(Path.home() / "venvs" / "karaoke-jp-lyrics" / "bin" / "python")
RENDER_PY = str(Path.home() / "venvs" / "karaoke-jp-render" / "bin" / "python")
LRC_BLOCK_SIZE = 2  # 2 phrases per lyric block → MID2BAR alternates row 2/3 (上下)
QUARTERS_PER_PAGE = 8  # bar-display fixed scale: 8 quarter notes per page → ≈5s @ 96 BPM
VOCAL_RATIO = 0.30  # guide-vocal level in mixed.wav
MID2BAR_APP_SETTINGS = str(Path("config") / "mid2bar_settings.json")


def _nvidia_lib(venv_dir: Path, component: str) -> str:
    """Return the absolute path to the nvidia/<component>/lib directory in
    the lyrics venv, glob-resolving the python3.X minor version."""
    pattern = str(venv_dir / "lib" / "python*" / "site-packages" / "nvidia" / component / "lib")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else ""


_LYRICS_VENV = Path.home() / "venvs" / "karaoke-jp-lyrics"
LYRICS_LD = ":".join(
    p for p in (_nvidia_lib(_LYRICS_VENV, "cublas"), _nvidia_lib(_LYRICS_VENV, "cudnn")) if p
)


rule separate:
    """M1: Mel-Band-RoFormer vocal separation (Kim FT2 Bleedless)."""
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
        f"{KARAOKE_BIN} separate {{input.audio:q}} -o {{params.out_dir:q}}"
        f" --model melband-roformer-kimmel-ft2-bleedless"


rule melody:
    """M2: vocals -> melody MIDI. Backend chosen by MELODY_BACKEND env var
    (default rmvpe; set to cectc for direct note transcription)."""
    input:
        vocals=str(OUT_DIR / "{song}" / "vocals.wav"),
        instrumental=str(OUT_DIR / "{song}" / "instrumental.wav"),
    output:
        midi=str(OUT_DIR / "{song}" / "melody.mid"),
    params:
        backend=MELODY_BACKEND,
    shell:
        f"{KARAOKE_BIN} melody {{input.vocals:q}} -o {{output.midi:q}} "
        f"--backend {{params.backend}} --instrumental {{input.instrumental:q}}"


rule quantize_melody:
    """M2b: snap note durations to {8th, quarter, half} of the beat grid
    estimated from instrumental.wav. Pitch + onset preserved verbatim;
    only the offset is moved to the closest musical-unit duration. Also
    writes a sidecar `<midi>.bpm.txt` so downstream rules can use the BPM."""
    input:
        midi=str(OUT_DIR / "{song}" / "melody.mid"),
        instrumental=str(OUT_DIR / "{song}" / "instrumental.wav"),
    output:
        midi=str(OUT_DIR / "{song}" / "melody_quantized.mid"),
        bpm=str(OUT_DIR / "{song}" / "melody_quantized.mid.bpm.txt"),
    shell:
        f"{Path.home() / 'venvs' / 'karaoke-jp-melody' / 'bin' / 'python'} "
        f"scripts/quantize_durations.py "
        f"--midi {{input.midi:q}} --instrumental {{input.instrumental:q}} "
        f"--out {{output.midi:q}}"


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
    resources:
        gpu=1,
    shell:
        f"LD_LIBRARY_PATH={LYRICS_LD}:${{{{LD_LIBRARY_PATH:-}}}} "
        f"{LYRICS_PY} scripts/run_asr.py {{input.vocals:q}} -o {{output.asr:q}} "
        f"--lyrics {{input.lyrics:q}}"


rule align:
    """M3c: ASR + tokens -> aligned.json + ruby.lrc (kana-aware NW).

    aligned.json at this stage uses Whisper word-level timestamps.  The
    following midi_timing rule replaces those with MIDI note onsets, which
    are syllable-accurate for singing.
    """
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


rule midi_timing:
    """M3d: Replace Whisper char timing with MIDI note onsets.

    SOME's melody.mid captures the actual note onset of every sung mora,
    giving much tighter timing than Whisper word timestamps.  The result is
    written to aligned_midi.json; downstream rules (export_lrc, midi_markers)
    consume that file so the final karaoke.lrc has syllable-accurate wipe.
    Lines with no notes in their Whisper window fall back to Whisper timing.
    """
    input:
        aligned=str(OUT_DIR / "{song}" / "aligned.json"),
        midi=str(OUT_DIR / "{song}" / "melody_quantized.mid"),
    output:
        aligned_midi=str(OUT_DIR / "{song}" / "aligned_midi.json"),
    shell:
        f"{MAIN_PY} scripts/midi_timing.py "
        f"--midi {{input.midi:q}} --aligned {{input.aligned:q}} "
        f"--out {{output.aligned_midi:q}}"


# M4 stages: lrc/marker prep run in the main venv (mido lives there); the
# actual render runs in the dedicated MID2BAR venv (Pygame + opencv).
# (MAIN_PY / RENDER_PY were defined alongside the other binaries above.)


rule export_lrc:
    """M4a: aligned_midi.json -> MID2BAR-flavored .lrc (centiseconds + @RubyN=)."""
    input:
        aligned=str(OUT_DIR / "{song}" / "aligned_midi.json"),
    output:
        lrc=str(OUT_DIR / "{song}" / "karaoke.lrc"),
    shell:
        f"{MAIN_PY} scripts/export_lrc.py {{input.aligned:q}} -o {{output.lrc:q}} "
        f"--block-size {LRC_BLOCK_SIZE}"


rule midi_markers:
    """M4b: melody_quantized.mid -> melody_markers.mid (page boundaries
    every QUARTERS_PER_PAGE quarter notes for fixed pixels-per-quarter)."""
    input:
        midi=str(OUT_DIR / "{song}" / "melody_quantized.mid"),
        bpm=str(OUT_DIR / "{song}" / "melody_quantized.mid.bpm.txt"),
    output:
        midi=str(OUT_DIR / "{song}" / "melody_markers.mid"),
    shell:
        f"{MAIN_PY} scripts/add_midi_markers.py "
        f"--midi {{input.midi:q}} --out {{output.midi:q}} "
        f"--mode beat --bpm-file {{input.bpm:q}} "
        f"--quarters-per-page {QUARTERS_PER_PAGE}"


rule mix:
    """M5: Blend instrumental + vocals at 20 % vocal ratio.

    Produces mixed.wav which is fed to the renderer so the guide vocal is
    audible at low volume while the melody bar still shows the karaoke pitch.
    Set VOCAL_RATIO=0 in the shell invocation to build a pure-instrumental
    version instead.
    """
    input:
        instrumental=str(OUT_DIR / "{song}" / "instrumental.wav"),
        vocals=str(OUT_DIR / "{song}" / "vocals.wav"),
    output:
        mixed=str(OUT_DIR / "{song}" / "mixed.wav"),
    shell:
        f"{MAIN_PY} scripts/mix_audio.py "
        f"--instrumental {{input.instrumental:q}} --vocals {{input.vocals:q}} "
        f"--out {{output.mixed:q}} --vocal-ratio {VOCAL_RATIO}"


def _background_arg(wc):
    """Return ``--background <path>`` if a background asset exists for this
    song, else empty. Searched in
    ``songs/<song>/background.{mp4,webm,mov,mkv,png,jpg,jpeg,webp}``.

    render_mp4.py always re-encodes the bg through ffmpeg before passing to
    MID2BAR, so source codec / container does not matter (avoids OpenCV's
    AV1 decode gap and silently-corrupt cv2.VideoCapture failures)."""
    import shlex
    for ext in ("mp4", "webm", "mov", "mkv", "png", "jpg", "jpeg", "webp"):
        p = SONGS_DIR / wc.song / f"background.{ext}"
        if p.exists():
            return f"--background {shlex.quote(str(p))}"
    return ""


rule render:
    """M4c/M5: headless MID2BAR render -> karaoke.mp4 (1080p60, h264 + aac).

    Uses mixed.wav (instrumental + 20 % guide vocal) as the audio track.
    Auto-detects ``songs/<song>/background.{mp4,webm,png,jpg,jpeg}`` and
    feeds it as the rendered backdrop; if absent, the bundled MID2BAR blue
    gradient is used.
    """
    input:
        audio=str(OUT_DIR / "{song}" / "mixed.wav"),
        midi=str(OUT_DIR / "{song}" / "melody_markers.mid"),
        lrc=str(OUT_DIR / "{song}" / "karaoke.lrc"),
    output:
        mp4=str(OUT_DIR / "{song}" / "karaoke.mp4"),
    params:
        bg=_background_arg,
        app_settings=MID2BAR_APP_SETTINGS,
    shell:
        f"SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy "
        f"{RENDER_PY} scripts/render_mp4.py "
        f"--audio {{input.audio:q}} --midi {{input.midi:q}} "
        f"--lrc {{input.lrc:q}} --out {{output.mp4:q}} "
        f"--app-settings {{params.app_settings:q}} {{params.bg}}"
