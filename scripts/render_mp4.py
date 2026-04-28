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


@click.command()
@click.option("--audio", "audio_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="instrumental.wav for playback (audio mux into output mp4).")
@click.option("--midi", "midi_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="melody.mid WITH page markers (run add_midi_markers first).")
@click.option("--lrc", "lrc_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="MID2BAR-flavored .lrc (centiseconds + @RubyN= header).")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True,
              help="output karaoke.mp4 path.")
@click.option("--lrc-settings", "lrc_settings_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="path to lyrics_settings/settings_default.json (default: bundled).")
@click.option("--app-settings", "app_settings_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="path to app_settings/settings.json (default: bundled).")
@click.option("--assets", "assets_json_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="path to app_settings/assets.json (default: bundled).")
def main(
    audio_path: str,
    midi_path: str,
    lrc_path: str,
    out_path: str,
    lrc_settings_path: str | None,
    app_settings_path: str | None,
    assets_json_path: str | None,
) -> None:
    # Defaults inside the bundled MID2BAR-Player tree. Resolve to absolute
    # paths BEFORE the os.chdir below — otherwise relative override paths
    # (e.g. `--app-settings my_settings.json` from the user's CWD) would
    # silently re-anchor to MID2BAR_DIR after the chdir and stop resolving.
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

    sys.path.insert(0, str(MID2BAR_DIR))
    os.chdir(MID2BAR_DIR)
    # MID2BAR's tools.resource_path uses dirname(sys.argv[0]) as the asset
    # base. Spoof argv[0] so it resolves against MID2BAR_DIR, not our
    # scripts/ dir.
    sys.argv[0] = str(MID2BAR_DIR / "main.py")

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

    from app import Mid2barPlayerApp

    app = Mid2barPlayerApp(
        audio_path=audio_abs,
        mid_path=midi_abs,
        lrc_path=lrc_abs,
        lrc_settings_path=lrc_settings_path,
        video_paths=[],  # no background video; MID2BAR's bundled samples
                          # would otherwise loop a colored shape.
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

    _press_space_after_init()
    app.run()
    print(f"[render] mp4 written to {out_abs}", flush=True)


if __name__ == "__main__":
    main()
