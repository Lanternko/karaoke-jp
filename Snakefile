# karaoke-jp Snakemake pipeline (M1-M4 wired).
#
# Usage:
#   snakemake --rerun-triggers mtime -j 1 outputs/<song>/karaoke.mp4
#
# (mtime-only: outputs/ has files produced outside Snakemake so provenance
# hashes are missing; default trigger set would force a full rebuild on
# every invocation.)
#
# DAG: separate -> {melody, rmvpe_f0, pyin_f0} -> {tokenize, asr -> align}
#      -> midi_markers -> fix_octaves -> render (+ export_lrc, mix branches).
#      Each stage runs in its own venv. Rules invoke the venv binary
#      by absolute path so reproducibility does not depend on the caller's
#      PATH or which venv they activated first.

import glob
import os
import shlex
from pathlib import Path

SONGS_DIR = Path("songs")
OUT_DIR = Path("outputs")

# Set MELODY_BACKEND=cectc to use direct CTC+CE note transcription
# (Wang & Jang TASLP 2023). Default rmvpe = frame-level F0 + heuristic
# segmentation. cectc skips segment_f0_to_notes entirely; preferred for
# low-register vocals where RMVPE octave-halving bites.
MELODY_BACKEND = os.environ.get("MELODY_BACKEND", "rmvpe")

# Lyrics TIMING source (default mms):
#   mms     = CTC forced alignment of the known mora sequence against the
#             separated vocals (NextFire mms-300m karaoke-ja aligner).
#             No ASR in the timing loop — tokens.json + vocals.wav suffice.
#   classic = Whisper ASR -> kana NW align -> mora->note vs melody MIDI.
TIMING_SOURCE = os.environ.get("TIMING_SOURCE", "mms")
if TIMING_SOURCE not in {"mms", "classic"}:
    raise ValueError(f"TIMING_SOURCE must be mms|classic, got {TIMING_SOURCE!r}")

import json as _json


def _load_canonical_profile():
    """Single source of truth: render/timing/pitch params come from the
    canonical version profile in config/versions.json (override with
    KARAOKE_PROFILE=<profile id>).
    """
    try:
        data = _json.loads((Path("config") / "versions.json").read_text())
        return data["profiles"][os.environ.get("KARAOKE_PROFILE") or data["canonical"]]
    except (OSError, KeyError, ValueError):
        return {}


_PROFILE = _load_canonical_profile()
_RENDER = _PROFILE.get("render", {})

# Pitch/display chain for the render rule (default game):
#   game    = GAME union -> make_display_grid (warp sidecar) -> render
#             --time-warp + flat bar skin + songinfo HUD + per-song
#             pitch-patch / rms-segments.
#   classic = legacy melody_markers.octavefix.mid straight into render (no
#             warp / skin / HUD): run with PITCH_CHAIN=classic.
PITCH_CHAIN = os.environ.get("PITCH_CHAIN", "game")
if PITCH_CHAIN not in {"game", "classic"}:
    raise ValueError(f"PITCH_CHAIN must be game|classic, got {PITCH_CHAIN!r}")

GAME_LANGUAGE = _PROFILE.get("pitch_chain", {}).get("game_language", "ja")

try:
    VOCAL_RATIO = float(os.environ.get("VOCAL_RATIO", _RENDER.get("vocal_ratio", 0.30)))
except (TypeError, ValueError) as exc:
    raise ValueError("VOCAL_RATIO must be a float in [0, 1].") from exc
if not (0.0 <= VOCAL_RATIO <= 1.0):
    raise ValueError(f"VOCAL_RATIO must be in [0, 1], got {VOCAL_RATIO}")

# Discover song IDs by scanning songs/<id>/source.* present. Skip `_`-prefixed
# stubs (e.g. songs/_smoketest, which has no lyrics.txt) so a broad target like
# `rule all` or a rule-name --forcerun doesn't choke the whole DAG on them.
SONG_IDS = sorted(
    p.parent.name
    for p in SONGS_DIR.glob("*/source.*")
    if p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac"}
    and not p.parent.name.startswith("_")
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


# Per-stage venv binaries. Pinning by
# absolute path avoids "depends on caller's active venv" surprises that
# bare-name commands like `karaoke-jp` or `python` have.
KARAOKE_BIN = str(Path.home() / "venvs" / "karaoke-jp" / "bin" / "karaoke-jp")
MAIN_PY = str(Path.home() / "venvs" / "karaoke-jp" / "bin" / "python")
MELODY_PY = str(Path.home() / "venvs" / "karaoke-jp-melody" / "bin" / "python")
LYRICS_PY = str(Path.home() / "venvs" / "karaoke-jp-lyrics" / "bin" / "python")
RENDER_PY = str(Path.home() / "venvs" / "karaoke-jp-render" / "bin" / "python")
ALIGN_PY = str(Path.home() / "venvs" / "karaoke-jp-align" / "bin" / "python")
RMVPE_CKPT = str(Path("third_party") / "SOME" / "pretrained" / "rmvpe" / "model.pt")
LRC_BLOCK_SIZE = int(_RENDER.get("lrc_block_size", 2))  # phrases/block → MID2BAR row 2/3 alternation
# Classic-chain marker page width (add_midi_markers, integer). Distinct from the
# display-grid quarters_per_page (16, make_display_grid's own default) — do not
# wire this to the display_grid profile value; midi_markers needs an int.
QUARTERS_PER_PAGE = 10
MID2BAR_APP_SETTINGS = str(Path("config") / "mid2bar_settings.json")
ASSETS_FLAT = str(Path("render_assets") / "assets_flat.json")  # flat bar skin
# Lyric layer + technique HUD (profile "lyrics" / "techniques"; absent in
# a profile -> MID2BAR default lyric style, whole-line LRC, no techniques)
_LYRICS = _PROFILE.get("lyrics", {})
LRC_SETTINGS = _LYRICS.get("lrc_settings")
LYRIC_FLAGS = ((f"--lrc-settings {shlex.quote(LRC_SETTINGS)} " if LRC_SETTINGS else "")
               + ("--miter-stroke " if _LYRICS.get("miter_stroke") else ""))
PAGE_SPLIT = bool(_LYRICS.get("page_split"))
TECHNIQUES = bool(_PROFILE.get("techniques", {}).get("enabled"))
LIGATURES = bool(_PROFILE.get("skin", {}).get("ligatures"))  # chained ligature bars
LIGATURE_STYLE = _PROFILE.get("skin", {}).get("ligature_style", "stroke")


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


# --- optional per-song sidecars ---------------------------------------------
# Sidecar files (reading overrides, pitch patches, rms gating, backgrounds) are
# hand-written per song and may be absent, so they cannot be plain inputs. But
# passing the path *only* through `params` keeps it out of the DAG: under
# `--rerun-triggers mtime` editing the sidecar then does NOT retrigger the rule
# and the output silently keeps the old content (e.g. a new reading in
# overrides/<song>.json would leave tokens.json — and every downstream
# artifact — on the stale reading).
#
# Fix: declare the same path as an optional input (present -> [path],
# absent -> []) AND build the CLI flag from the same helper, so input and
# params can never disagree.


def _optional_input(path):
    """[str(path)] when the sidecar exists, [] otherwise (or path is None)."""
    if path is None:
        return []
    return [str(path)] if Path(path).exists() else []


def _optional_flag(flag, path):
    """``--flag <path>`` when the sidecar exists, '' otherwise. Always pair
    with _optional_input() on the same path."""
    if path is None or not Path(path).exists():
        return ""
    return f"{flag} {shlex.quote(str(path))}"


def _render_overrides_path(song):
    """Per-song facts about the source for the horizontal render
    ({"bg_mask": "x,y,w,h[;...]", "key": "Bb"}); see versions.json per_song."""
    return Path("overrides") / f"{song}_render.json"


def _render_override_flags(song):
    p = _render_overrides_path(song)
    if not p.exists():
        return ""
    ov = _json.loads(p.read_text(encoding="utf-8"))
    flags = []
    if ov.get("bg_mask"):
        flags.append(f"--bg-mask {shlex.quote(ov['bg_mask'])}")
    if ov.get("key"):
        flags.append(f"--key {shlex.quote(ov['key'])}")
    return " ".join(flags)


def _override_path(song):
    """Per-song reading overrides (flat {"漢字": "よみ"}) fed to fugashi."""
    return Path("overrides") / f"{song}.json"


rule tokenize:
    """M3a: lyrics.txt -> tokens.json (fugashi + UniDic readings)."""
    input:
        lyrics=str(SONGS_DIR / "{song}" / "lyrics.txt"),
        override=lambda wc: _optional_input(_override_path(wc.song)),
    output:
        tokens=str(OUT_DIR / "{song}" / "tokens.json"),
    params:
        override=lambda wc: _optional_flag("--override", _override_path(wc.song)),
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
        lyric constraint. Line-final particles own their actual
        phones, interior ad-libs are absorbed by CTC blank, onsets sit at
        consonant starts. Edge <star> absorbers (default on) stop tail ad-lib
        hum / bleed from dragging the last line's end to the song end.
        Raw CTC offsets are peaky-early by design; line_end_repair below
        recovers them.
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

        The first-mora gate + absorb-trailing flags are a single cross-song
        boundary config (no per-song tuning).
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

    Late-fusion octave repair (unguarded consensus + same-pitch
    merge): shift only notes RMVPE *and* pYIN agree are a full
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
    """Audio-only score chain (flags pinned inside
    scripts/run_score_chain.py):

        octavefix -> bp_hybrid_relabel(d1/s0.02/r1.2)
                  -> score_note_postfix(refine/extend/shakuri/fill)
                  -> add_midi_markers

    OPT-IN target (melody_markers.scorefix.mid); also the fallback input of
    game_chain. Applies the bp+postfix stages to the standard chain's
    octavefix MIDI."""
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


def _pitch_patch_path(song):
    """Hand-written per-song display fixes (melisma_split / drop_notes /
    relabel); the render applies them when present."""
    return Path("overrides") / f"{song}_pitch_patch.json"


def _rms_segments_path(song):
    """Hand-made RMS gating that lets make_display_grid drop
    instrumental-break bleed notes. Not produced by any rule."""
    return OUT_DIR / song / "rms_segments.json"


rule game_chain:
    """GAME-backbone melody (config pinned in scripts/run_game_chain.py):

        GAME large (-l ja) -> postfix(extend) -> melody_union(fallback =
        score_chain output) -> add_midi_markers

    Output: melody_markers.gamescore.mid (default render input)."""
    input:
        vocals=str(OUT_DIR / "{song}" / "vocals.wav"),
        fallback=str(OUT_DIR / "{song}" / "melody_markers.scorefix.mid"),
        f0=str(OUT_DIR / "{song}" / "rmvpe_f0.npz"),
        aligned=str(OUT_DIR / "{song}" / "aligned_midi.json"),
        bpm=str(OUT_DIR / "{song}" / "melody_quantized.mid.bpm.txt"),
        pitch_patch=lambda wc: _optional_input(_pitch_patch_path(wc.song)),
        rms_segments=lambda wc: _optional_input(_rms_segments_path(wc.song)),
    output:
        midi=str(OUT_DIR / "{song}" / "melody_markers.gamescore.mid"),
        warp=str(OUT_DIR / "{song}" / "melody_markers.gamescore.warp.json"),
        union=str(OUT_DIR / "{song}" / "melody_markers.gamescore.union.mid"),
        ligatures=([str(OUT_DIR / "{song}" / "melody_markers.gamescore.ligatures.json")]
                   if LIGATURES else []),
    params:
        # Per-song display fixes (melisma_split / drop_notes / relabel) +
        # instrumental-break rms gating, when the song provides them.
        pitch_patch=lambda wc: _optional_flag(
            "--pitch-patch", _pitch_patch_path(wc.song)
        ),
        rms_segments=lambda wc: _optional_flag(
            "--rms-segments", _rms_segments_path(wc.song)
        ),
    shell:
        f"{MAIN_PY} scripts/run_game_chain.py "
        f"--vocals {{input.vocals:q}} --fallback-midi {{input.fallback:q}} "
        f"--f0 {{input.f0:q}} --aligned {{input.aligned:q}} "
        f"--bpm-file {{input.bpm:q}} --language {GAME_LANGUAGE} "
        f"{{params.pitch_patch}} {{params.rms_segments}} "
        f"--out {{output.midi:q}}"


rule export_lrc_paged:
    """Horizontal LRC: lines cut where the display grid flips page
    (lyrics.page_split), so a lyric row never runs wider than its bars."""
    input:
        aligned=str(OUT_DIR / "{song}" / "aligned_midi.json"),
        midi=str(OUT_DIR / "{song}" / "melody_markers.gamescore.mid"),
        warp=str(OUT_DIR / "{song}" / "melody_markers.gamescore.warp.json"),
        render_ov=lambda wc: _optional_input(_render_overrides_path(wc.song)),
    output:
        lrc=str(OUT_DIR / "{song}" / "karaoke.paged.lrc"),
    params:
        breaks=lambda wc: _optional_flag("--breaks-from", _render_overrides_path(wc.song)),
    shell:
        f"{MAIN_PY} scripts/export_lrc.py {{input.aligned:q}} -o {{output.lrc:q}} "
        f"--block-size {LRC_BLOCK_SIZE} --page-grid {{input.midi:q}} "
        f"--time-warp {{input.warp:q}} {{params.breaks}}"


rule techniques:
    """Original singer's DAM-vocabulary techniques for the songinfo HUD."""
    input:
        f0=str(OUT_DIR / "{song}" / "rmvpe_f0.npz"),
        union=str(OUT_DIR / "{song}" / "melody_markers.gamescore.union.mid"),
        warp=str(OUT_DIR / "{song}" / "melody_markers.gamescore.warp.json"),
    output:
        json=str(OUT_DIR / "{song}" / "techniques.json"),
    shell:
        f"{MAIN_PY} scripts/detect_techniques.py --f0 {{input.f0:q}} "
        f"--midi {{input.union:q}} --time-warp {{input.warp:q}} --out {{output.json:q}}"


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


def _background_path(song):
    """First existing ``songs/<song>/background.{mp4,webm,mov,mkv,png,jpg,
    jpeg,webp}``, else None.

    render_mp4.py always re-encodes the bg through ffmpeg before passing to
    MID2BAR, so source codec / container does not matter (avoids OpenCV's
    AV1 decode gap and silently-corrupt cv2.VideoCapture failures)."""
    for ext in ("mp4", "webm", "mov", "mkv", "png", "jpg", "jpeg", "webp"):
        p = SONGS_DIR / song / f"background.{ext}"
        if p.exists():
            return p
    return None


if PITCH_CHAIN == "game":

    rule render:
        """M4c/M5: headless MID2BAR render -> karaoke.mp4
        (1080p60, h264 + aac) via the GAME display-grid chain.

        GAME-union grid MIDI + piecewise-linear
        time-warp sidecar + flat bar skin + songinfo HUD, over mixed.wav
        (vocal_ratio from versions.json). Per-song pitch-patch / rms-segments
        are applied upstream in rule game_chain.
        """
        input:
            audio=str(OUT_DIR / "{song}" / "mixed.wav"),
            midi=str(OUT_DIR / "{song}" / "melody_markers.gamescore.mid"),
            warp=str(OUT_DIR / "{song}" / "melody_markers.gamescore.warp.json"),
            lrc=str(OUT_DIR / "{song}" / ("karaoke.paged.lrc" if PAGE_SPLIT else "karaoke.lrc")),
            techniques=[str(OUT_DIR / "{song}" / "techniques.json")] if TECHNIQUES else [],
            ligatures=([str(OUT_DIR / "{song}" / "melody_markers.gamescore.ligatures.json")]
                       if LIGATURES else []),
            render_ov=lambda wc: _optional_input(_render_overrides_path(wc.song)),
            bg=lambda wc: _optional_input(_background_path(wc.song)),
        output:
            mp4=str(OUT_DIR / "{song}" / "karaoke.mp4"),
        params:
            bg=lambda wc: _optional_flag(
                "--background", _background_path(wc.song)
            ),
            techniques=lambda wc, input: " ".join(
                [f"--techniques {shlex.quote(t)}" for t in input.techniques]
                + [f"--ligatures {shlex.quote(t)} --ligature-style {LIGATURE_STYLE}"
                   for t in input.ligatures]),
            per_song=lambda wc: _render_override_flags(wc.song),
            app_settings=MID2BAR_APP_SETTINGS,
            assets=ASSETS_FLAT,
        shell:
            f"SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy "
            f"{RENDER_PY} scripts/render_mp4.py "
            f"--audio {{input.audio:q}} --midi {{input.midi:q}} "
            f"--lrc {{input.lrc:q}} --out {{output.mp4:q}} "
            f"--app-settings {{params.app_settings:q}} "
            f"--assets {{params.assets:q}} --hud songinfo "
            f"--time-warp {{input.warp:q}} {LYRIC_FLAGS}"
            f"{{params.techniques}} {{params.per_song}} {{params.bg}}"

else:

    rule render:
        """M4c/M5 CLASSIC (legacy, PITCH_CHAIN=classic): octavefix.mid straight
        into MID2BAR, no warp/skin/HUD. For songs not yet GAME-validated."""
        input:
            audio=str(OUT_DIR / "{song}" / "mixed.wav"),
            midi=str(OUT_DIR / "{song}" / "melody_markers.octavefix.mid"),
            lrc=str(OUT_DIR / "{song}" / "karaoke.lrc"),
            bg=lambda wc: _optional_input(_background_path(wc.song)),
        output:
            mp4=str(OUT_DIR / "{song}" / "karaoke.mp4"),
        params:
            bg=lambda wc: _optional_flag(
                "--background", _background_path(wc.song)
            ),
            app_settings=MID2BAR_APP_SETTINGS,
        shell:
            f"SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy "
            f"{RENDER_PY} scripts/render_mp4.py "
            f"--audio {{input.audio:q}} --midi {{input.midi:q}} "
            f"--lrc {{input.lrc:q}} --out {{output.mp4:q}} "
            f"--app-settings {{params.app_settings:q}} {{params.bg}}"
