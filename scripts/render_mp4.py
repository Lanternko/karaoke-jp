"""Headless render: drive third_party/MID2BAR-Player/app.py to MP4.

Run inside ``~/venvs/karaoke-jp-render/`` with these tweaks:

* ``SDL_VIDEODRIVER=dummy`` so Pygame opens an in-memory display surface
  rather than attempting an X11 / Wayland window.
* Inject a synthetic SPACE keydown so ``app.run()``'s wait-for-input loop
  exits immediately into the recorder branch.
* Monkey-patch ``tkinter.messagebox.showinfo`` so the recorder's
  "Recording finished" popup doesn't crash on a display-less box.
* Inject a synthetic ``pygame.QUIT`` event once the recorder reports it
  has written all frames, so ``app.run()`` returns cleanly instead of
  spinning forever.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Headless Pygame must be set BEFORE the first pygame import anywhere in the
# process, including indirect imports inside MID2BAR-Player.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import click

REPO_ROOT = Path(__file__).resolve().parents[1]
MID2BAR_DIR = REPO_ROOT / "third_party" / "MID2BAR-Player"


def _stub_audio_modules() -> None:
    """Stub sounddevice (PortAudio backend) since we never enable mic input.
    MID2BAR's fft.py imports it unconditionally at module load."""
    import sys
    import types

    fake_sd = types.ModuleType("sounddevice")

    class _FakeStream:
        def __init__(self, *a, **kw):
            pass
        def start(self): pass
        def stop(self): pass
        def close(self): pass

    fake_sd.InputStream = _FakeStream
    fake_sd.query_devices = lambda *a, **kw: []
    sys.modules["sounddevice"] = fake_sd


def _silence_messagebox() -> None:
    """Stub out tkinter.messagebox so MID2BAR's framerecorder import succeeds
    even when the system doesn't ship python3-tk, AND any showinfo call from
    the recorder turns into a print rather than a popup that blocks headless.
    """
    import sys
    import types

    def _noop(*args, **kwargs):
        title = args[0] if args else kwargs.get("title", "")
        msg = args[1] if len(args) > 1 else kwargs.get("message", "")
        print(f"[render] tk msgbox suppressed: {title}: {msg}", flush=True)

    # Build a minimal fake tkinter package tree.
    fake_tk = types.ModuleType("tkinter")
    fake_mb = types.ModuleType("tkinter.messagebox")
    fake_mb.showinfo = _noop
    fake_mb.showerror = _noop
    fake_mb.showwarning = _noop
    fake_tk.messagebox = fake_mb
    sys.modules["tkinter"] = fake_tk
    sys.modules["tkinter.messagebox"] = fake_mb


def _hook_recorder_termination() -> None:
    """When the recorder finishes its frame budget, post pygame.QUIT so
    app.run() exits its main loop instead of looping forever."""
    import pygame
    from framerecorder import PipeFrameRecorder

    original_push = PipeFrameRecorder.push_frame

    def push_frame_with_quit_on_done(self, surface):
        result = original_push(self, surface)
        # When the recorder hits its frame budget it sets is_recording=False;
        # at that point we want app.run() to exit so we synthesize QUIT.
        if not self.is_recording:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        return result

    PipeFrameRecorder.push_frame = push_frame_with_quit_on_done


def _press_space_after_init() -> None:
    """Queue a SPACE keydown so the wait-for-user loop in app.run() exits."""
    import pygame

    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))


def _zero_lag_time(app) -> None:
    """Force LAG_TIME = 0 so pitch-bar fill matches the lyric wipe.

    MID2BAR's bar-fill width is ``(current_time - LAG_TIME - note.start) /
    duration * width`` (app.py:1012-1017), but the lyric wipe uses raw
    current_time (app.py:1426-1433). The default LAG_TIME=0.3 is intended to
    compensate live mic latency; in offline render it just makes bars trail
    the wipe by 300 ms. SettingsSchema is a frozen dataclass, so go through
    object.__setattr__ to bypass the freeze.
    """
    object.__setattr__(app.s, "LAG_TIME", 0.0)


def _apply_time_warp(app, warp_path: str) -> None:
    """Bar-area-only time warp (display-grid mode).

    The display MIDI's notes/markers live on a standardized display timeline
    (fixed quarter width, fixed gaps). Wrap the bar-area draw methods so they
    see display time = interp(real time): the wipe still flips exactly when
    the singer flips; only the cursor speed varies (Kojek-approved). Lyrics,
    audio and the seekbar keep real time.
    """
    import json as _json

    import numpy as _np

    data = _json.loads(Path(warp_path).read_text())
    real = _np.asarray(data["real"], dtype=float)
    disp = _np.asarray(data["display"], dtype=float)

    def _wrap(name: str) -> None:
        orig = getattr(app, name)

        def wrapped(*args, **kwargs):
            if getattr(app, "_warp_active", False):
                return orig(*args, **kwargs)
            real_t = app.current_time
            app._warp_active = True
            app.current_time = float(_np.interp(real_t, real, disp))
            try:
                return orig(*args, **kwargs)
            finally:
                app.current_time = real_t
                app._warp_active = False

        setattr(app, name, wrapped)

    for name in ("draw_notes", "draw_now_bar"):
        _wrap(name)


def _disable_particles(app) -> None:
    """Suppress MID2BAR's sparkle/glitter particles.

    MID2BAR's ``update_particles()`` (app.py:1249) reassigns the three particle
    lists every frame to whatever ``_update_particle_list`` returns. So
    swapping the lists for a NullList only survives one frame — by frame 2
    the lists are vanilla lists again and the for-loop draws sparkles.
    Patch the actual update method instead: lambda always returns ``[]``,
    which clears the list AND skips every ``p.update()`` / ``p.draw()`` call.

    Sources of sparkle particles in app.py:
      - line 777, 1068: note-pass ``Particle`` (general kira-kira)
      - line 790, 1049: ``MicInputParticle`` (mic-input glow; we never have mic)
      - line 897-924, 1181-1207: ``bar_count_particles`` for stat counter pops
    """
    app._update_particle_list = lambda particle_list, screen: []


def _shrink_long_lines(
    threshold_chars: int = 13,
    *,
    safe_width_px: int = 1700,
    char_advance_factor: float = 1.30,
    min_font_size: int = 56,
    max_font_size: int = 78,
) -> None:
    """Monkey-patch lyrics.text_tools.draw_lyric_image_with_ruby so any line
    that won't fit at the default 100 px font is rendered with a per-line
    shrunk font.

    Sidesteps MID2BAR's lack of any auto-fit logic. We do NOT mutate the
    .lrc text (so chorus marker glyphs ＜ ＞ never appear on-screen) — we
    flip the module's ``IS_CHORUS`` global per line and additionally
    override ``LYRIC_CHORUS.FONT_SIZE`` to a fit-to-width value computed
    from the line's char count, then restore everything.

    Width math: each full-width JP char occupies ~ FONT_SIZE *
    char_advance_factor pixels including kerning. Pick the largest font
    that keeps n_chars * font * factor ≤ safe_width_px, clamped to
    [min_font_size, max_font_size]. Default safe_width_px = 1720 = the
    typical 1920 px screen minus 100 px margins on each side.
    """
    import lyrics.text_tools as tt  # noqa: E402

    orig = tt.draw_lyric_image_with_ruby

    def wrapper(data, settings, **kw):
        # Count actual chars, not list lengths (some entries are list-of-list).
        line_chars = sum(
            len(s) for u in data.get("lyrics", []) for s in u if s
        )

        prev_is_chorus = tt.IS_CHORUS
        # Save fields we may mutate so concurrent / sequential calls don't
        # leak state across lines.
        lc = settings.LYRIC_CHORUS
        rc = settings.RUBY_CHORUS
        prev_lc_font = lc.FONT_SIZE
        prev_rc_font = rc.FONT_SIZE

        if line_chars > threshold_chars:
            tt.IS_CHORUS = True
            target = int(safe_width_px / max(line_chars, 1) / char_advance_factor)
            target = max(min_font_size, min(max_font_size, target))
            # Proportional ruby (default ratio ~ RUBY/LYRIC = 30/80 = 0.375).
            ruby_ratio = prev_rc_font / max(prev_lc_font, 1)
            lc.FONT_SIZE = target
            rc.FONT_SIZE = max(18, int(round(target * ruby_ratio)))
        try:
            return orig(data=data, settings=settings, **kw)
        finally:
            tt.IS_CHORUS = prev_is_chorus
            lc.FONT_SIZE = prev_lc_font
            rc.FONT_SIZE = prev_rc_font

    tt.draw_lyric_image_with_ruby = wrapper


def _hide_minmax_columns(app) -> None:
    """Hide the BAR_COUNT min/max counter columns (and their animation popups).

    ``draw_bar_count`` (app.py:1500-1530) blits 3 fixed-position digits:
    normal, max, min — followed by a loop over up/down/long. Settings'
    BAR_COUNT_DICT controls position and color. Frozen dataclass blocks
    attribute reassignment, but the inner dict is mutable, so we replace
    the max/min dict entries with off-screen positions.
    """
    OFF = (-9999, -9999)
    # When app_settings/settings.json overrides these dicts, settings_loader's
    # _cast leaves the values as raw dicts instead of (frozen) dataclasses.
    # Support both shapes so the off-screen patch works either way.
    def _replace_pos(entry, pos, key="color"):
        if hasattr(entry, key):
            return type(entry)(**{**{f.name: getattr(entry, f.name) for f in __import__("dataclasses").fields(entry)}, "pos": pos})
        return {**entry, "pos": list(pos)}

    for k in ("max", "min"):
        bc = app.s.BAR_COUNT_DICT[k]
        app.s.BAR_COUNT_DICT[k] = _replace_pos(bc, OFF)
        an = app.s.BAR_PASSED_COUNT_ANIMATION_DICT[k]
        app.s.BAR_PASSED_COUNT_ANIMATION_DICT[k] = _replace_pos(an, OFF, key="colors")


def _hide_notes_without_visible_lyrics(app) -> None:
    """Only draw pitch bars while a lyric image is actually visible.

    MID2BAR pages are time windows, so upcoming notes can appear several
    seconds before the next lyric line starts. For karaoke exports that looks
    like pitch guidance in an instrumental/no-lyric gap. Reuse MID2BAR's own
    lyric timing model so the note layer follows the same visibility condition
    as draw_lyrics().
    """
    original_draw_notes = app.draw_notes

    def has_visible_lyrics() -> bool:
        for lyric in app.lyrics:
            for typ in app.lyrics_types:
                part = lyric.get(typ)
                if not part:
                    continue
                if part["start"] <= app.current_time < part["end"]:
                    for x_wipe in part.get("x_wipes", []):
                        if x_wipe[0] <= app.current_time < x_wipe[1]:
                            return True
        return False

    def draw_notes_when_lyrics_visible():
        if has_visible_lyrics():
            return original_draw_notes()
        return None

    app.draw_notes = draw_notes_when_lyrics_visible


@click.command()
@click.option("--audio", "audio_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="instrumental.wav for playback (audio mux into output mp4).")
@click.option("--midi", "midi_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="melody.mid WITH page markers (run add_midi_markers first).")
@click.option("--lrc", "lrc_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="MID2BAR-flavored .lrc (centiseconds + @RubyN= header).")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True,
              help="output karaoke.mp4 path.")
@click.option("--background", "bg_path", type=click.Path(exists=True, dir_okay=False),
              default=None,
              help="Background image (.png/.jpg) or video (.mp4/.webm). PNG/JPG "
                   "is auto-converted to a 5s looping mp4 via ffmpeg. Default: "
                   "MID2BAR's bundled blue gradient.")
@click.option("--lrc-settings", "lrc_settings_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="path to lyrics_settings/settings_default.json (default: bundled).")
@click.option("--app-settings", "app_settings_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="path to app_settings/settings.json (default: bundled).")
@click.option("--assets", "assets_json_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="path to app_settings/assets.json (default: bundled).")
@click.option("--time-warp", "time_warp_path", type=click.Path(exists=True, dir_okay=False),
              default=None,
              help="JSON {real:[..], display:[..]} from make_display_grid.py. The "
              "bar MIDI then lives on a display timeline; the bar area (notes, "
              "wipe, cursor) sees piecewise-linearly warped time while audio, "
              "lyrics and seekbar stay on real time.")
def main(
    audio_path: str,
    midi_path: str,
    lrc_path: str,
    out_path: str,
    bg_path: str | None,
    lrc_settings_path: str | None,
    app_settings_path: str | None,
    assets_json_path: str | None,
    time_warp_path: str | None,
) -> None:
    # Defaults inside the bundled MID2BAR-Player tree. Resolve to absolute
    # paths BEFORE the os.chdir below — otherwise relative override paths
    # (e.g. `--app-settings my_settings.json` from the user's CWD) would
    # silently re-anchor to MID2BAR_DIR after the chdir and stop resolving.
    if time_warp_path is not None:
        time_warp_path = str(Path(time_warp_path).resolve())
    lrc_settings_path = str(Path(
        lrc_settings_path or (MID2BAR_DIR / "lyrics_settings" / "settings_default.json")
    ).resolve())
    app_settings_path = str(Path(
        app_settings_path or (MID2BAR_DIR / "app_settings" / "settings.json")
    ).resolve())
    assets_json_path = str(Path(
        assets_json_path or (MID2BAR_DIR / "app_settings" / "assets.json")
    ).resolve())

    # MID2BAR-Player imports its modules by bare names ('import lrc' etc),
    # so we cd into its tree and prepend it to sys.path. Output / input
    # paths must already be absolute by this point.
    audio_abs = str(Path(audio_path).resolve())
    midi_abs = str(Path(midi_path).resolve())
    lrc_abs = str(Path(lrc_path).resolve())
    out_abs = str(Path(out_path).resolve())
    Path(out_abs).parent.mkdir(parents=True, exist_ok=True)

    # Background normalization: yt-dlp often hands back AV1-encoded mp4 that
    # OpenCV's bundled cv2.VideoCapture cannot decode (no software AV1
    # support), so we always re-encode to h264 + yuv420p + 1920x1080 before
    # handing to MID2BAR. Static images are fed to ffmpeg with -loop 1
    # for a 5s mp4 that MID2BAR's loop-back will keep restarting.
    bg_abs = None
    if bg_path:
        import subprocess
        src = Path(bg_path).resolve()
        ext = src.suffix.lower()
        converted = Path(out_abs).parent / "_background.mp4"
        if ext in {".mp4", ".webm", ".mov", ".mkv"}:
            cmd = [
                "ffmpeg", "-y", "-i", str(src),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
                       "pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                str(converted),
            ]
        elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
            cmd = [
                "ffmpeg", "-y", "-loop", "1", "-i", str(src),
                "-t", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
                       "pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                str(converted),
            ]
        else:
            raise click.UsageError(
                f"Unsupported background format: {ext}. "
                "Use .mp4/.webm/.mov/.mkv or .png/.jpg/.jpeg/.webp."
            )
        print(f"[render] normalizing bg -> {converted}", flush=True)
        subprocess.run(cmd, check=True, capture_output=True)
        bg_abs = str(converted)

    sys.path.insert(0, str(MID2BAR_DIR))
    os.chdir(MID2BAR_DIR)
    # MID2BAR's tools.resource_path uses dirname(sys.argv[0]) as the asset
    # base. Spoof argv[0] so it resolves against MID2BAR_DIR, not our
    # scripts/ dir.
    sys.argv[0] = str(MID2BAR_DIR / "main.py")

    # MID2BAR caches per-line text PNGs under
    # `./lyrics_images/<lrc_basename>/` plus `./lyrics_images/<lrc_basename>.json`.
    # If two songs share an LRC basename (e.g. both produce `karaoke.lrc`)
    # the cache from song A is silently reused for song B. Wipe the cache
    # tree for our lrc basename before each render so the renderer
    # regenerates from the actual LRC content.
    import shutil
    lrc_stem = Path(lrc_abs).stem
    cache_dir = Path("lyrics_images") / lrc_stem
    cache_json = Path("lyrics_images") / f"{lrc_stem}.json"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    if cache_json.exists():
        cache_json.unlink()

    _silence_messagebox()
    _stub_audio_modules()
    _hook_recorder_termination()

    # Force MID2BAR's recorder to write to OUR path. The recorder defaults
    # to ./recordings/<timestamp>.mp4 if out_path is None, so we monkey-
    # patch start() to thread our path through.
    from framerecorder import PipeFrameRecorder
    original_start = PipeFrameRecorder.start
    def patched_start(self, *args, **kwargs):
        kwargs["out_path"] = out_abs
        return original_start(self, *args, **kwargs)
    PipeFrameRecorder.start = patched_start

    # MID2BAR's settings_schema uses frozen dataclasses (BarCountEntry,
    # AnimationEntry) for dict entries, but app.py accesses them with ["key"]
    # subscript notation.  Add __getitem__ so attribute access works either way.
    # This must be patched before app.py is imported (the draw methods reference
    # the class at call time, so late-binding is fine).
    import settings_schema as _ss
    for _cls in (_ss.BarCountEntry, _ss.AnimationEntry):
        if not hasattr(_cls, "__getitem__"):
            _cls.__getitem__ = lambda self, key: getattr(self, key)  # type: ignore[assignment]

    # Per-line font shrink for long lyric lines. Must hook into
    # lyrics.text_tools BEFORE Mid2barPlayerApp(..) constructs (the
    # constructor runs lrc.load_lyrics → draw_lyric_image_with_ruby
    # one line at a time and caches each PNG to disk).
    _shrink_long_lines(threshold_chars=14)

    from app import Mid2barPlayerApp

    app = Mid2barPlayerApp(
        audio_path=audio_abs,
        mid_path=midi_abs,
        lrc_path=lrc_abs,
        lrc_settings_path=lrc_settings_path,
        # If the user provided a background mp4 (or PNG converted above),
        # loop it; otherwise leave blank for MID2BAR's bundled blue gradient.
        video_paths=[bg_abs] if bg_abs else [],
        video_fixed_fps=0,
        video_shuffle=False,
        # MID2BAR's __init__ does `os.path.exists(splash_image)` without a
        # None-check, so pass empty strings to safely skip those assets.
        credit_text="",
        splash_image="",
        title_image="",
        enable_mic_input=False,
        mic_input_channel=0,
        record=True,
        settings_json_path=app_settings_path,
        assets_json_path=assets_json_path,
    )

    _disable_particles(app)
    _hide_minmax_columns(app)
    if time_warp_path is None:
        _hide_notes_without_visible_lyrics(app)
    else:
        # display-grid MIDIs are pre-gated and live on a warped timeline;
        # comparing their note times against real LRC times would be wrong
        _apply_time_warp(app, time_warp_path)
    _zero_lag_time(app)
    _press_space_after_init()
    app.run()
    print(f"[render] mp4 written to {out_abs}", flush=True)


if __name__ == "__main__":
    main()
