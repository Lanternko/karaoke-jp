# karaoke-jp Snakemake pipeline (M1-M4 wired).
#
# Usage:
#   snakemake --rerun-triggers mtime -j 1 outputs/<song>/karaoke.mp4
#
# (mtime-only: outputs/ has files produced outside Snakemake so provenance
# hashes are missing; default trigger set would force a full rebuild on
# every invocation. See CLAUDE.md "做事方式".)
#
# DAG: separate -> {melody, rmvpe_f0, pyin_f0} -> {tokenize, asr -> align}
#      -> midi_markers -> fix_octaves -> render (+ export_lrc, mix branches).
#      Each stage runs in its own venv (see
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

# Lyrics TIMING source (canonical mms since 2026-06-12; Kojek ear-accepted on
# night-dancer + chidori, line gold weighted start MAE 0.090 -> 0.069):
#   mms     = CTC forced alignment of the known mora sequence against the
#             separated vocals (NextFire mms-300m karaoke-ja fine-tune).
#             No ASR in the timing loop — tokens.json + vocals.wav suffice.
#   classic = Whisper ASR -> kana NW align -> mora->note vs melody MIDI.
#             Kept switchable for A/B and for mms regressions (haru-hikage
#             long-interlude re-entry family — see MEMORY.md PoC entry).
TIMING_SOURCE = os.environ.get("TIMING_SOURCE", "mms")
if TIMING_SOURCE not in {"mms", "classic"}:
    raise ValueError(f"TIMING_SOURCE must be mms|classic, got {TIMING_SOURCE!r}")

try:
    VOCAL_RATIO = float(os.environ.get("VOCAL_RATIO", "0.30"))
except ValueError as exc:
    raise ValueError("VOCAL_RATIO must be a float in [0, 1].") from exc
if not (0.0 <= VOCAL_RATIO <= 1.0):
    raise ValueError(f"VOCAL_RATIO must be in [0, 1], got {VOCAL_RATIO}")

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
MELODY_PY = str(Path.home() / "venvs" / "karaoke-jp-melody" / "bin" / "python")
LYRICS_PY = str(Path.home() / "venvs" / "karaoke-jp-lyrics" / "bin" / "python")
RENDER_PY = str(Path.home() / "venvs" / "karaoke-jp-render" / "bin" / "python")
ALIGN_PY = str(Path.home() / "venvs" / "karaoke-jp-align" / "bin" / "python")
RMVPE_CKPT = str(Path("third_party") / "SOME" / "pretrained" / "rmvpe" / "model.pt")
LRC_BLOCK_SIZE = 2  # 2 phrases per lyric block → MID2BAR alternates row 2/3 (上下)
QUARTERS_PER_PAGE = 10  # bar-display fixed scale: 10 quarter notes per page → narrower pitch bars
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


rule rmvpe_f0:
    """M2c: vocals -> RMVPE F0 cache (.npz). Primary estimator for the
    consensus octave fix (fix_octaves). Same separated vocals as melody."""
    input:
        vocals=str(OUT_DIR / "{song}" / "vocals.wav"),
    output:
        npz=str(OUT_DIR / "{song}" / "rmvpe_f0.npz"),
    resources:
        gpu=1,
    shell:
        f"{MELODY_PY} scripts/extract_rmvpe_f0.py "
        f"--wav {{input.vocals:q}} --model {RMVPE_CKPT} --out {{output.npz:q}}"


rule pyin_f0:
    """M2d: vocals -> pYIN F0 cache (.npz). Second estimator; vetoes a shift
    when it disagrees with RMVPE (late fusion). CPU-only (librosa)."""
    input:
        vocals=str(OUT_DIR / "{song}" / "vocals.wav"),
    output:
        npz=str(OUT_DIR / "{song}" / "pyin_f0.npz"),
    shell:
        f"{MELODY_PY} scripts/extract_pyin_f0.py "
        f"--wav {{input.vocals:q}} --out {{output.npz:q}}"


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


def _override_flag(song):
    p = Path("overrides") / f"{song}.json"
    return f"--override {p}" if p.exists() else ""

rule tokenize:
    """M3a: lyrics.txt -> tokens.json (fugashi + UniDic readings)."""
    input:
        lyrics=str(SONGS_DIR / "{song}" / "lyrics.txt"),
    output:
        tokens=str(OUT_DIR / "{song}" / "tokens.json"),
    params:
        override=lambda wc: _override_flag(wc.song),
    shell:
        f"{LYRICS_PY} scripts/tokenize_lyrics.py {{input.lyrics:q}} -o {{output.tokens:q}} {{params.override}}"


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


if TIMING_SOURCE == "mms":
    rule mms_align:
        """M3d (canonical): CTC forced alignment as the lyrics timing source.

        The known mora sequence (tokens.json, reading-corrected via
        overrides/) is romanized and force-aligned against the separated
        vocals — timing comes from frame-level acoustic posteriors under the
        lyric constraint (survey §3.5). Line-final particles own their actual
        phones, ad-libs are absorbed by CTC blank, onsets sit at consonant
        starts (the event the Audacity line gold marks). Raw CTC offsets are
        peaky-early by design; line_end_repair below recovers them (FZZ-style
        post-processing). Kojek ear-accepted 2026-06-12 (night-dancer +
        chidori; "舊的小瑕疵也變平滑").
        """
        input:
            tokens=str(OUT_DIR / "{song}" / "tokens.json"),
            vocals=str(OUT_DIR / "{song}" / "vocals.wav"),
        output:
            aligned_midi=str(OUT_DIR / "{song}" / "aligned_midi.raw.json"),
        resources:
            gpu=1,
        shell:
            f"{ALIGN_PY} scripts/forced_align_mms.py "
            f"--vocals {{input.vocals:q}} --tokens {{input.tokens:q}} "
            f"--out {{output.aligned_midi:q}}"
else:
    rule midi_timing:
        """M3d (classic): Replace Whisper char timing with MIDI note onsets.

        SOME's melody.mid captures the actual note onset of every sung mora,
        giving much tighter timing than Whisper word timestamps.  The result is
        written to aligned_midi.raw.json; line_end_repair then produces the
        aligned_midi.json that downstream rules (export_lrc, midi_markers)
        consume, so the final karaoke.lrc has syllable-accurate wipe.
        Lines with no notes in their Whisper window fall back to Whisper timing.

        The first-mora gate + absorb-trailing flags are the single cross-song
        boundary config validated on the chidori/haru-hikage/tuki-zero line gold
        (start MAE 0.11-0.18s) and ear-checked on byoushin (2026-06-11, Kojek:
        line starts/ends must be automatic — see docs/handoff-2026-06-11.md #4).
        """
        input:
            aligned=str(OUT_DIR / "{song}" / "aligned.json"),
            midi=str(OUT_DIR / "{song}" / "melody_quantized.mid"),
        output:
            aligned_midi=str(OUT_DIR / "{song}" / "aligned_midi.raw.json"),
        shell:
            f"{MAIN_PY} scripts/midi_timing.py "
            f"--midi {{input.midi:q}} --aligned {{input.aligned:q}} "
            f"--out {{output.aligned_midi:q}} "
            f"--first-mora-min-delay 0.05 --first-mora-gate-prev-gap 0.75 "
            f"--first-mora-gate-lead-tolerance 0.08 --absorb-trailing-notes"


rule line_end_repair:
    """M3e: RMS-based line-end sustain capture (same validated config).

    Extends a line's last char to cover the sung sustain tail the melody
    MIDI under-reports; guards against eating into the next line.
    """
    input:
        aligned=str(OUT_DIR / "{song}" / "aligned_midi.raw.json"),
        vocals=str(OUT_DIR / "{song}" / "vocals.wav"),
    output:
        aligned_midi=str(OUT_DIR / "{song}" / "aligned_midi.json"),
    shell:
        f"{MAIN_PY} scripts/line_end_repair.py "
        f"--aligned {{input.aligned:q}} --vocals {{input.vocals:q}} "
        f"--tail-top-db 26 --next-guard 0.25 --tail-gap 0.12 "
        f"-o {{output.aligned_midi:q}}"


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
    """M4b: melody_quantized.mid -> melody_markers.mid.

    Page boundaries stay on a fixed beat grid, while note events are filtered
    to lyric windows so false melody detections in instrumental gaps do not
    render pitch bars when no lyrics are present.
    """
    input:
        midi=str(OUT_DIR / "{song}" / "melody_quantized.mid"),
        bpm=str(OUT_DIR / "{song}" / "melody_quantized.mid.bpm.txt"),
        aligned=str(OUT_DIR / "{song}" / "aligned_midi.json"),
    output:
        midi=str(OUT_DIR / "{song}" / "melody_markers.mid"),
    shell:
        f"{MAIN_PY} scripts/add_midi_markers.py "
        f"--midi {{input.midi:q}} --out {{output.midi:q}} "
        f"--mode beat --bpm-file {{input.bpm:q}} "
        f"--quarters-per-page {QUARTERS_PER_PAGE} "
        f"--aligned {{input.aligned:q}}"


rule fix_octaves:
    """M4b2: melody_markers.mid -> melody_markers.octavefix.mid.

    Canonical late-fusion octave repair (unguarded consensus + same-pitch
    merge, 2026-06-06): shift only notes RMVPE *and* pYIN agree are a full
    octave off, then merge fragments. Rewrites only the note track, so the
    page-marker track stays intact for the renderer. Runs after midi_markers
    (post note-window gating); midi_timing is upstream and untouched.
    """
    input:
        midi=str(OUT_DIR / "{song}" / "melody_markers.mid"),
        rmvpe=str(OUT_DIR / "{song}" / "rmvpe_f0.npz"),
        pyin=str(OUT_DIR / "{song}" / "pyin_f0.npz"),
    output:
        midi=str(OUT_DIR / "{song}" / "melody_markers.octavefix.mid"),
    shell:
        f"{MAIN_PY} scripts/fix_pitch_octaves.py {{input.midi:q}} "
        f"-o {{output.midi:q}} "
        f"--rmvpe-f0 {{input.rmvpe:q}} --pyin-f0 {{input.pyin:q}}"


rule basicpitch:
    """Score-chain input: BasicPitch note events from the separated vocals.

    Second acoustic opinion for bp_hybrid_relabel (fixes RMVPE semitone
    undershoot on sustained chorus notes). Opt-in: only built when a
    score_chain target is requested."""
    input:
        vocals=str(OUT_DIR / "{song}" / "vocals.wav"),
    output:
        csv=str(OUT_DIR / "{song}" / "basic_pitch" / "vocals_basic_pitch.csv"),
    shell:
        f"{MAIN_PY} scripts/run_basicpitch.py "
        f"--vocals {{input.vocals:q}} "
        f"--out-dir {str(OUT_DIR)}/{{wildcards.song}}/basic_pitch"


rule score_chain:
    """Canonical audio-only score chain (validated 2026-06-10 on Chidori
    humangold + byoushin cross-reference; flags pinned inside
    scripts/run_score_chain.py):

        octavefix -> bp_hybrid_relabel(d1/s0.02/r1.2)
                  -> score_note_postfix(refine/extend/shakuri/fill)
                  -> add_midi_markers

    OPT-IN target (melody_markers.scorefix.mid). The render rule still
    consumes melody_markers.octavefix.mid by default because tuki-zero /
    bocchi-guitar were never evaluated under this chain; flip a song's render
    input only after checking its output. Note: the refit-q70 mora step used
    in the Chidori experiments is still a manual sidecar — this rule applies
    the validated bp+postfix stages to the standard chain's octavefix MIDI."""
    input:
        midi=str(OUT_DIR / "{song}" / "melody_markers.octavefix.mid"),
        bp=str(OUT_DIR / "{song}" / "basic_pitch" / "vocals_basic_pitch.csv"),
        f0=str(OUT_DIR / "{song}" / "rmvpe_f0.npz"),
        aligned=str(OUT_DIR / "{song}" / "aligned_midi.json"),
        bpm=str(OUT_DIR / "{song}" / "melody_quantized.mid.bpm.txt"),
    output:
        midi=str(OUT_DIR / "{song}" / "melody_markers.scorefix.mid"),
    shell:
        f"{MAIN_PY} scripts/run_score_chain.py "
        f"--midi {{input.midi:q}} --basicpitch-csv {{input.bp:q}} "
        f"--f0 {{input.f0:q}} --aligned {{input.aligned:q}} "
        f"--bpm-file {{input.bpm:q}} --quarters-per-page {QUARTERS_PER_PAGE} "
        f"--out {{output.midi:q}}"


rule game_chain:
    """GAME-backbone melody (validated 2026-06-10 on Chidori humangold +
    byoushin; config pinned in scripts/run_game_chain.py):

        GAME large (-l ja) -> postfix(extend) -> melody_union(fallback =
        score_chain output) -> add_midi_markers

    Chidori numbers: exact .728 / note-level 72.8% vs the classic chain's
    .651 / 60.6%. OPT-IN target (melody_markers.gamescore.mid); flip a
    song's render input only after a visual check."""
    input:
        vocals=str(OUT_DIR / "{song}" / "vocals.wav"),
        fallback=str(OUT_DIR / "{song}" / "melody_markers.scorefix.mid"),
        f0=str(OUT_DIR / "{song}" / "rmvpe_f0.npz"),
        aligned=str(OUT_DIR / "{song}" / "aligned_midi.json"),
        bpm=str(OUT_DIR / "{song}" / "melody_quantized.mid.bpm.txt"),
    output:
        midi=str(OUT_DIR / "{song}" / "melody_markers.gamescore.mid"),
    shell:
        f"{MAIN_PY} scripts/run_game_chain.py "
        f"--vocals {{input.vocals:q}} --fallback-midi {{input.fallback:q}} "
        f"--f0 {{input.f0:q}} --aligned {{input.aligned:q}} "
        f"--bpm-file {{input.bpm:q}} --quarters-per-page {QUARTERS_PER_PAGE} "
        f"--out {{output.midi:q}}"


rule mix:
    """M5: Blend instrumental + vocals at a configurable vocal ratio.

    Produces mixed.wav which is fed to the renderer so the guide vocal stays
    audible while the melody bar still shows the karaoke pitch. Set
    VOCAL_RATIO=0 in the shell invocation to build a pure-instrumental
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

    Uses mixed.wav (instrumental + 30 % guide vocal) as the audio track.
    Auto-detects ``songs/<song>/background.{mp4,webm,png,jpg,jpeg}`` and
    feeds it as the rendered backdrop; if absent, the bundled MID2BAR blue
    gradient is used.
    """
    input:
        audio=str(OUT_DIR / "{song}" / "mixed.wav"),
        midi=str(OUT_DIR / "{song}" / "melody_markers.octavefix.mid"),
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
