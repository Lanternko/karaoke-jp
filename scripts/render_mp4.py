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
    the singer flips; only the cursor speed varies. Lyrics,
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


def _install_miter_stroke(miter_limit: float = 2.0) -> None:
    """尖角描邊: PIL's stroke uses FreeType round joins, which bloat into soft
    rounded blobs at karaoke widths. Swap ImageDraw.text (only while MID2BAR
    draws a lyric line) for a FreeType Stroker with MITER joins; the glyph body
    still comes from PIL's own mask, so fill pixels are bit-identical."""
    import freetype
    import numpy as _np
    from PIL import Image as _Image, ImageDraw as _ImageDraw
    import lyrics.text_tools as tt

    cache: dict = {}

    def stroke_mask(path, size, ch, width):
        key = (path, size, ch, width)
        if key not in cache:
            face = freetype.Face(path)
            face.set_pixel_sizes(0, size)
            face.load_char(ch, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_BITMAP)
            glyph = face.glyph.get_glyph()
            st = freetype.Stroker()
            st.set(int(width * 64), freetype.FT_STROKER_LINECAP_ROUND,
                   freetype.FT_STROKER_LINEJOIN_MITER, int(miter_limit * 0x10000))
            glyph.stroke(st, True)
            bg = glyph.to_bitmap(freetype.FT_RENDER_MODE_NORMAL, freetype.Vector(0, 0), True)
            bm = bg.bitmap
            arr = (_np.array(bm.buffer, dtype=_np.uint8).reshape(bm.rows, bm.width)
                   if bm.rows else _np.zeros((0, 0), _np.uint8))
            cache[key] = (arr, bg.left, bg.top)
        return cache[key]

    def over(img, cov, x, y, color):
        # ImageDraw's RGBA semantics for an opaque ink == source-over with coverage as alpha
        h, w = cov.shape
        x0, y0 = max(0, -x), max(0, -y)
        x1, y1 = min(w, img.width - x), min(h, img.height - y)
        if x1 <= x0 or y1 <= y0:
            return
        layer = _np.empty((y1 - y0, x1 - x0, 4), _np.uint8)
        layer[..., :3] = color[:3]
        a = color[3] if len(color) > 3 else 255
        layer[..., 3] = cov[y0:y1, x0:x1].astype(_np.uint16) * a // 255
        img.alpha_composite(_Image.fromarray(layer, "RGBA"), dest=(x + x0, y + y0))

    orig_text = _ImageDraw.ImageDraw.text

    def text(self, xy, t, fill=None, font=None, *args, stroke_width=0, stroke_fill=None, **kw):
        img = self._image
        if (not stroke_width or stroke_fill is None or img.mode != "RGBA" or args or kw
                or not isinstance(t, str) or not hasattr(font, "path")):
            return orig_text(self, xy, t, fill, font, *args,
                             stroke_width=stroke_width, stroke_fill=stroke_fill, **kw)
        x, y = xy
        ascent = font.getmetrics()[0]
        for ch in t:
            xi, yi = int(round(x)), int(round(y))
            if not ch.isspace():
                m, left, top = stroke_mask(font.path, font.size, ch, stroke_width)
                if m.size:
                    over(img, m, xi + left, yi + ascent - top, stroke_fill)
            mask, off = font.getmask2(ch, "L")
            if mask.size[0] and mask.size[1]:
                mk = _np.frombuffer(bytes(mask), _np.uint8).reshape(mask.size[1], mask.size[0])
                over(img, mk, xi + off[0], yi + off[1], fill)
            x += font.getlength(ch)

    orig = tt.draw_lyric_image_with_ruby

    def wrapper(*a, **kw):
        _ImageDraw.ImageDraw.text = text
        try:
            return orig(*a, **kw)
        finally:
            _ImageDraw.ImageDraw.text = orig_text

    tt.draw_lyric_image_with_ruby = wrapper


def _install_lyric_halo(spec: str) -> None:
    """二重フチ: add a dark outer ring around the per-line lyric/ruby PNGs.

    MID2BAR's PIL draw has one stroke only, so a light stroke (the sung-state
    white rim) vanishes on bright backgrounds. ``spec`` = "#RRGGBB:LYRIC_W:RUBY_W".

    The lrc-settings STROKE_WIDTH must hold the TOTAL (inner + halo) width —
    display_tools derives the lyric/ruby row clips from it, so the halo stays
    inside the clip. While drawing we subtract the halo so the inner stroke is
    STROKE_WIDTH - halo, then grow the halo from the rendered alpha by grey
    dilation (one pass over the whole line ⇒ a neighbour's ring never cuts into
    the previous glyph, unlike per-char stroke drawing). Same colour on both
    wipe states, so the image_1-over-image_2 edge never shows a seam.
    """
    import cv2
    import numpy as _np
    from PIL import Image as _Image
    import lyrics.text_tools as tt  # noqa: E402

    try:
        col, lyric_w, ruby_w = spec.split(":")
        rgb = tuple(int(col.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        lyric_w, ruby_w = int(lyric_w), int(ruby_w)
    except ValueError:
        raise click.UsageError(f"--lyric-halo must be '#RRGGBB:LYRIC_W:RUBY_W': got {spec!r}")

    def _ring(alpha, w):
        if w <= 0:
            return _np.zeros_like(alpha)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * w + 1, 2 * w + 1))
        return cv2.dilate(alpha, k)

    orig = tt.draw_lyric_image_with_ruby

    def wrapper(data, settings, **kw):
        cats = [(settings.LYRIC, lyric_w), (settings.LYRIC_CHORUS, lyric_w),
                (settings.RUBY, ruby_w), (settings.RUBY_CHORUS, ruby_w)]
        prev = [c.STROKE_WIDTH for c, _ in cats]
        for c, w in cats:
            c.STROKE_WIDTH = max(0, c.STROKE_WIDTH - w)
        try:
            out = orig(data=data, settings=settings, **kw)
        finally:
            for (c, _), p in zip(cats, prev):
                c.STROKE_WIDTH = p
        # Ruby band sits above the lyric band; split halfway between the
        # ruby row's bottom clip and the lyric row's top clip.
        ruby_bottom = settings.GENERAL.Y_RUBY + settings.RUBY.FONT_SIZE + settings.RUBY.STROKE_WIDTH
        split = (ruby_bottom + settings.GENERAL.Y_LYRIC - settings.LYRIC.STROKE_WIDTH) // 2
        for key in ("image_1", "image_2"):
            img = _Image.open(out[key]).convert("RGBA")
            arr = _np.array(img)
            a = arr[..., 3]
            halo_a = _np.vstack([_ring(a, ruby_w)[:split], _ring(a, lyric_w)[split:]])
            base = _np.zeros_like(arr)
            base[..., :3] = rgb
            base[..., 3] = halo_a
            merged = _Image.alpha_composite(_Image.fromarray(base), img)
            merged.save(out[key], "PNG")
        return out

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


# --- song-info HUD: static whole-song stats replacing the cumulative counters ---
_NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
_NOTE_NAMES_FLAT = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]


def _pitch_name(p: int, use_flats: bool = False) -> str:
    # MIDI 60 = C4 (Yamaha/JOYSOUND convention: octave = p // 12 - 1).
    names = _NOTE_NAMES_FLAT if use_flats else _NOTE_NAMES
    return f"{names[p % 12]}{p // 12 - 1}"


def _compute_song_stats(notes) -> dict:
    """Whole-song note statistics, computed once. Static for the whole video.

    ``notes`` is ``app.notes`` (display-timeline notes: pitch and order are the
    real ones, only time is warped). Mirrors MID2BAR's own ``long`` definition
    (>90th-percentile duration) but replaces its musically meaningless
    ``range / 2`` leap threshold with a fixed >= 7 semitones (perfect fifth).
    """
    import numpy as _np

    pitches = [n["pitch"] for n in notes]
    durs = [n["end"] - n["start"] for n in notes]
    if not pitches:
        return {"count": 0, "pmin": 60, "pmax": 60, "span": 0,
                "high": 0, "up": 0, "down": 0, "long": 0}
    pmin, pmax = min(pitches), max(pitches)
    p90 = float(_np.percentile(durs, 90)) if durs else 0.0
    up = sum(1 for i in range(len(notes) - 1)
             if notes[i + 1]["pitch"] - notes[i]["pitch"] >= 7)
    down = sum(1 for i in range(len(notes) - 1)
               if notes[i + 1]["pitch"] - notes[i]["pitch"] <= -7)
    return {
        "count": len(notes),
        "pmin": pmin, "pmax": pmax, "span": pmax - pmin,
        "high": sum(1 for p in pitches if p >= pmax - 2),
        "up": up, "down": down,
        "long": sum(1 for d in durs if d > p90),
    }


def _format_song_info(st: dict, use_flats: bool = False) -> str:
    return (
        f"音符 {st['count']}　"
        f"音域 {_pitch_name(st['pmin'], use_flats)}〜{_pitch_name(st['pmax'], use_flats)}"
        f"（{st['span']}半音）　"
        f"高音 {st['high']}　"
        f"跳躍 ↑{st['up']} ↓{st['down']}　"
        f"長音 {st['long']}"
    )


# Krumhansl-Schmuckler key profiles (duration-weighted pitch-class correlation).
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
# Major keys conventionally spelled with flats: F, B♭, E♭, A♭, D♭ (G♭/F♯ left sharp).
_FLAT_MAJOR_PCS = {5, 10, 3, 8, 1}


def _read_wav_mono(path: str):
    """Dependency-free WAV reader -> (mono float32 in [-1,1], sample_rate).

    The render venv has no soundfile/scipy and stdlib ``wave`` rejects float
    WAVs (raises on format 3). Our pipeline feeds PCM16 (the karaoke mix) AND
    float32 (separated stems), so parse the RIFF chunks directly and handle
    PCM 16/32-bit + IEEE float 32/64-bit. Returns None for anything else
    (24-bit, WAVE_FORMAT_EXTENSIBLE, compressed) so callers fall back. NOTE:
    feeding a float32 stem to the old PCM16-only reader silently fell back to
    melody-only key detection (the biased path) -- this reader fixes that.
    """
    import struct

    import numpy as _np

    try:
        raw = Path(path).read_bytes()
        if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
            return None
        fmt = data = None
        i = 12
        while i + 8 <= len(raw):
            cid, sz = raw[i:i + 4], struct.unpack_from("<I", raw, i + 4)[0]
            if cid == b"fmt ":
                fmt = struct.unpack_from("<HHIIHH", raw, i + 8)
            elif cid == b"data":
                data = raw[i + 8:i + 8 + sz]
            i += 8 + sz + (sz & 1)  # RIFF chunks are word-aligned
        if fmt is None or data is None:
            return None
        afmt, ch, rate, bits = fmt[0], fmt[1], fmt[2], fmt[5]
        if afmt == 1 and bits == 16:
            x = _np.frombuffer(data, dtype="<i2").astype(_np.float32) / 32768.0
        elif afmt == 1 and bits == 32:
            x = _np.frombuffer(data, dtype="<i4").astype(_np.float32) / 2147483648.0
        elif afmt == 3 and bits == 32:
            x = _np.frombuffer(data, dtype="<f4").astype(_np.float32)
        elif afmt == 3 and bits == 64:
            x = _np.frombuffer(data, dtype="<f8").astype(_np.float32)
        else:
            return None
        if ch > 1:
            x = x[:len(x) - (len(x) % ch)].reshape(-1, ch).mean(1)
        return x, rate
    except Exception:
        return None


def _audio_pcp(path: str):
    """Peak-based pitch-class profile of a WAV (None if unreadable).

    Per frame only local spectral maxima vote (top-10, 80-2000 Hz), which keeps
    percussion/overtone smear out of the chroma -- a naive all-bin STFT chroma
    of a full mix ranks garbage keys. A circular-mean
    tuning estimate is subtracted before pitch-class rounding.
    """
    import numpy as _np

    rd = _read_wav_mono(path)
    if rd is None:
        return None
    x, fr = rd
    N, hop = 8192, 4096
    win = _np.hanning(N)
    freqs = _np.fft.rfftfreq(N, 1 / fr)
    bidx = _np.where((freqs >= 80) & (freqs <= 2000))[0]
    peaks = []
    for i in range(0, len(x) - N, hop):
        m = _np.abs(_np.fft.rfft(x[i:i + N] * win))[bidx]
        cand = _np.where((m[1:-1] > m[:-2]) & (m[1:-1] > m[2:]))[0] + 1
        if not len(cand):
            continue
        for j in cand[_np.argsort(m[cand])[-10:]]:
            peaks.append((69 + 12 * _np.log2(freqs[bidx[j]] / 440.0), m[j]))
    if not peaks:
        return None
    arr = _np.asarray(peaks)
    ang = _np.exp(2j * _np.pi * (arr[:, 0] % 1.0))
    tune = float(_np.angle(_np.average(ang, weights=arr[:, 1])) / (2 * _np.pi))
    pcp = _np.zeros(12)
    _np.add.at(pcp, _np.round(arr[:, 0] - tune).astype(int) % 12,
               _np.sqrt(arr[:, 1]))
    return pcp / pcp.sum()


_PC_NAME = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _name_to_pc(name: str):
    """'Eb' / 'E♭' / 'F#' / 'F♯' -> pitch-class int (None if unparseable)."""
    name = name.strip().replace("♭", "b").replace("♯", "#")
    if not name or name[0] not in _PC_NAME:
        return None
    pc = _PC_NAME[name[0]]
    for ch in name[1:]:
        pc += 1 if ch == "#" else -1 if ch == "b" else 0
    return pc % 12


def _key_display(tonic: int, mode: str):
    """(tonic_pc, 'major'|'minor') -> (display name, use_flats) in our spelling."""
    rel_major = tonic if mode == "major" else (tonic + 3) % 12
    flats = rel_major in _FLAT_MAJOR_PCS
    names = _NOTE_NAMES_FLAT if flats else _NOTE_NAMES
    return names[tonic] + ("m" if mode == "minor" else ""), flats


def _detect_key_essentia(path: str):
    """Essentia KeyExtractor (HPCP + key template) -- the primary detector.

    _detect_key falls back to
    peak-PCP/melody when essentia is unavailable. We re-spell from the pitch
    class via _key_display so the HUD sharp/flat convention stays consistent.
    Returns None on any failure (missing package, unreadable audio).
    """
    try:
        import essentia
        essentia.log.warningActive = False  # silence per-call "No network" spam
        essentia.log.infoActive = False
        import essentia.standard as _es
        key, scale, strength = _es.KeyExtractor()(
            _es.MonoLoader(filename=path, sampleRate=44100)())
    except Exception:
        return None
    pc = _name_to_pc(key)
    if pc is None:
        return None
    mode = "minor" if scale.startswith("min") else "major"
    name, use_flats = _key_display(pc, mode)
    return {"name": name, "use_flats": use_flats, "corr": float(strength),
            "source": "essentia", "top": [(name, float(strength))]}


def _detect_key(notes, audio_path: str | None = None) -> dict:
    """Key guess: Essentia KeyExtractor primary, peak-PCP K-S fallback.

    Both read the render audio (instrumental-dominated mix): the accompaniment
    harmony carries the key, the vocal does not (melody-only K-S falls for the
    dominant). If essentia is unavailable we fall back to the in-house
    peak-PCP, then to duration-weighted melody pitch classes.
    """
    import numpy as _np

    if audio_path:
        res = _detect_key_essentia(audio_path)
        if res is not None:
            return res

    w = _audio_pcp(audio_path) if audio_path else None
    source = "harmony" if w is not None else "melody"
    if w is None:
        w = _np.zeros(12)
        for n in notes:
            w[n["pitch"] % 12] += n["end"] - n["start"]
    if not _np.any(w) or _np.std(w) == 0:
        # No usable evidence (silent/unreadable audio AND no notes): a flat
        # profile makes every corrcoef NaN and argmax picks C by accident.
        return {"name": "?", "use_flats": False, "corr": 0.0,
                "source": "none", "top": []}
    scored = []
    for mode, profile in (("major", _KS_MAJOR), ("minor", _KS_MINOR)):
        prof = _np.asarray(profile, float)
        for tonic in range(12):
            r = float(_np.corrcoef(_np.roll(prof, tonic), w)[0, 1])
            scored.append((r, tonic, mode))
    scored.sort(reverse=True)
    r, tonic, mode = scored[0]
    name, use_flats = _key_display(tonic, mode)
    return {
        "name": name,
        "use_flats": use_flats,
        "corr": r,
        "source": source,
        "top": [(_key_display(t, m)[0], rr) for rr, t, m in scored[:3]],
    }


def _minimal_icon(kind: str, color, size: int = 34):
    """Flat HUD glyphs in the category colors, supersampled 4x for clean edges.

    Label text and abstract arrows/dashes read ambiguously ("↗" as a trend,
    "—" as a minus sign). The counters are about pitch bars, so the glyphs are drawn in
    the pitch-bar language itself:
      note = eighth note; high = one bar pressed against a dashed ceiling with
      dim bars below; up/down = a low and a high bar joined by a jump arrow;
      long = one long bar with a span bracket over it; range = ascending
      level bars (replaces the baked rainbow double-arrow by the range gauge).
    Technique glyphs (--techniques) draw the pitch contour of each DAM
    technique: shakuri = curve scooping up into a bar, fall = bar then curve
    dropping away, kobushi = sharp up-and-back bump, vibrato = sine wave.
    """
    import pygame

    W = size * 4
    surf = pygame.Surface((W, W), pygame.SRCALPHA)
    c = (*color, 255)
    dim = (*color, 110)
    lw = max(2, int(W * 0.11))
    bh = int(W * 0.20)  # pitch-bar thickness

    def line(a, b, col=c, w=lw):
        pygame.draw.line(surf, col, a, b, w)
        pygame.draw.circle(surf, col, (int(a[0]), int(a[1])), w // 2)
        pygame.draw.circle(surf, col, (int(b[0]), int(b[1])), w // 2)

    def bar(x0, x1, cy, col=c):
        pygame.draw.rect(
            surf, col, pygame.Rect(int(W * x0), int(W * cy - bh / 2),
                                   int(W * (x1 - x0)), bh),
            border_radius=bh // 2)

    def jump(x0, y0, x1, y1):
        # shaft + open arrowhead at (x1, y1)
        import math
        line((W * x0, W * y0), (W * x1, W * y1))
        ang = math.atan2(y1 - y0, x1 - x0)
        for d in (2.5, -2.5):
            hx = x1 - 0.20 * math.cos(ang + d * 0.25)
            hy = y1 - 0.20 * math.sin(ang + d * 0.25)
            line((W * x1, W * y1), (W * hx, W * hy))

    if kind == "note":
        pygame.draw.ellipse(surf, c, pygame.Rect(W * .14, W * .60, W * .42, W * .30))
        sx = int(W * .56 - lw)
        pygame.draw.rect(surf, c, pygame.Rect(sx, W * .10, lw, W * .66))
        pygame.draw.polygon(surf, c, [(sx, W * .10), (W * .86, W * .34),
                                      (W * .86, W * .50), (sx, W * .30)])
    elif kind == "high":
        for i in range(3):  # dashed ceiling = the song's top notes
            x = .06 + i * .34
            line((W * x, W * .10), (W * (x + .20), W * .10), w=max(2, lw // 2))
        bar(.24, .94, .34)
        bar(.06, .56, .80, dim)
    elif kind in ("up", "down"):
        lo, hi = (.86, .20) if kind == "up" else (.20, .86)
        bar(.04, .40, lo)
        bar(.60, .96, hi)
        sgn = -1 if kind == "up" else 1
        jump(.30, lo + sgn * .16, .70, hi - sgn * .18)
    elif kind == "long":
        bar(.04, .96, .66)
        y = .30
        line((W * .10, W * y), (W * .90, W * y), w=max(2, lw // 2))
        line((W * .08, W * (y - .10)), (W * .08, W * (y + .10)), w=max(2, lw // 2))
        line((W * .92, W * (y - .10)), (W * .92, W * (y + .10)), w=max(2, lw // 2))
    elif kind == "shakuri":  # scoop up into a held bar
        bar(.52, .96, .26)
        pts = [(W * (.08 + .46 * u), W * (.86 - .60 * (1 - (1 - u) ** 2)))
               for u in [i / 12 for i in range(13)]]
        for a_, b_ in zip(pts, pts[1:]):
            line(a_, b_)
    elif kind == "fall":  # held bar, then the pitch drops away
        bar(.04, .48, .26)
        pts = [(W * (.46 + .46 * u), W * (.26 + .60 * u * u))
               for u in [i / 12 for i in range(13)]]
        for a_, b_ in zip(pts, pts[1:]):
            line(a_, b_)
    elif kind == "kobushi":  # quick up-and-back turn inside a held note
        import math
        pts = [(W * (.06 + .88 * u),
                W * (.66 - .44 * math.exp(-((u - .5) / .12) ** 2)))
               for u in [i / 24 for i in range(25)]]
        for a_, b_ in zip(pts, pts[1:]):
            line(a_, b_)
    elif kind == "vibrato":  # periodic wave
        import math
        pts = [(W * (.06 + .88 * u), W * (.50 - .22 * math.sin(u * 2.5 * 2 * math.pi)))
               for u in [i / 40 for i in range(41)]]
        for a_, b_ in zip(pts, pts[1:]):
            line(a_, b_)
    elif kind == "range":
        bw = int(lw * 2.0)
        base = W * .82
        for x, h in ((.24, .26), (.50, .46), (.76, .66)):
            pygame.draw.rect(
                surf, c,
                pygame.Rect(int(W * x - bw / 2), int(base - W * h), bw, int(W * h)),
                border_radius=bw // 2)
    return pygame.transform.smoothscale(surf, (size, size))


def _layout_counter_pill(font, final_counts, *, icon_w: int,
                         pad: int = 22, icon_gap: int = 10, group_gap: int = 44,
                         height: int = 56):
    """Fixed per-group geometry for the counter pill.

    Each group is [icon][icon_gap][number slot], where the number slot is as
    wide as that category's *final* count, so the layout never shifts while
    counts grow and every group ends where its widest number ends. Groups are
    separated by the same group_gap with a thin divider centred in it, so
    spacing is uniform regardless of how many digits each count has.
    Returns (pill_surface, [(icon_x, num_x)], [divider_x]) relative to pill.
    """
    import pygame

    slots = [font.size(str(n))[0] for n in final_counts]
    xs, dividers = [], []
    x = pad
    for i, slot in enumerate(slots):
        xs.append((x, x + icon_w + icon_gap))
        x += icon_w + icon_gap + slot
        if i + 1 < len(slots):
            dividers.append(x + group_gap // 2)
            x += group_gap
    width = x + pad
    pill = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.rect(pill, (0, 0, 0, 150), pill.get_rect(), border_radius=14)
    for dx in dividers:
        pygame.draw.line(pill, (255, 255, 255, 40), (dx, 14), (dx, height - 14), 2)
    return pill, xs


def _erase_baked_hud_strip(app) -> None:
    """Clear MID2BAR's baked counter chrome from the project_front/back skins.

    Both are ``convert_alpha()`` SRCALPHA surfaces (tools.load_image). Measured
    extents on the 1920x1080 flat theme:
      * project_front dots/labels/icons: x0-1097, y348-382
      * project_back  dark counter slots: x20-1240, y~325-414
    Keepers (NOT cleared): the range gauge slot (project_back x1400-1900) and
    its rainbow arrow icon (project_front x1416-1459) -- both at x>=1400 -- plus
    the corner gray boxes and the pitch-bar bg panel, all at y<=319. Clearing
    x0-1300 below y325 stays clear of every keeper.
    """
    import pygame

    for surf, rect in (
        (app.assets.project_front, pygame.Rect(0, 338, 1300, 56)),
        (app.assets.project_back, pygame.Rect(0, 325, 1300, 95)),
    ):
        assert surf.get_flags() & pygame.SRCALPHA, (
            "HUD skin surface is not SRCALPHA; an RGBA fill would paint opaque "
            "black instead of clearing to transparent")
        surf.fill((0, 0, 0, 0), rect)


def _install_song_info_panel(app, time_warp_path: str | None = None,
                             audio_path: str | None = None,
                             techniques_path: str | None = None,
                             key_override: str | None = None) -> None:
    """Icon-only counters that tick as each event scrolls past the now-bar.

    No words, cross-language. Left pill = five
    [icon count] groups -- ♪ all notes, ▲ high notes (within 2 semitones of the
    song top), ↑/↓ leaps (>= 7 semitones, MID2BAR icons), long notes (> p90
    duration, MID2BAR icon). Counts increase the moment the note's tail crosses
    the cursor (offline render can't score singing; these are chart events, not
    judgments). Range info moves onto the gauge: min/max note names at its
    ends, semitone span top-right, K-S key guess top-left.

    draw_bar_count is NOT wrapped by _apply_time_warp, but app.notes lives on
    the display timeline when a warp is active -- so we map current_time
    real->display ourselves before comparing against note ends. Interludes are
    compressed out of the display timeline, so counters freeze there.
    """
    import json as _json

    import numpy as _np
    import pygame

    _erase_baked_hud_strip(app)

    notes = sorted(app.notes, key=lambda n: (n["start"], n["pitch"]))
    st = _compute_song_stats(notes)
    if key_override:
        # a user-supplied key beats every detector
        minor = key_override.endswith("m")
        pc = _name_to_pc(key_override[:-1] if minor else key_override)
        if pc is None:
            raise click.UsageError(f"--key: unknown key {key_override!r} (e.g. Bb, F#m)")
        name, flats = _key_display(pc, "minor" if minor else "major")
        key = {"name": name, "use_flats": flats, "source": "override", "top": [(name, 1.0)]}
    else:
        key = _detect_key(notes, audio_path)
    use_flats = key["use_flats"]
    print(f"[hud] {_format_song_info(st, use_flats)}", flush=True)
    print(f"[hud] key guess ({key['source']}): "
          + "  ".join(f"{n} r={r:.3f}" for n, r in key["top"]), flush=True)

    if time_warp_path:
        data = _json.loads(Path(time_warp_path).read_text())
        w_real = _np.asarray(data["real"], float)
        w_disp = _np.asarray(data["display"], float)
    else:
        w_real = w_disp = None

    durs = [n["end"] - n["start"] for n in notes]
    p90 = float(_np.percentile(durs, 90)) if durs else 0.0
    pmax = st["pmax"]

    def ends_where(pred):
        return _np.sort(_np.asarray(
            [n["end"] for i, n in enumerate(notes) if pred(i, n)], float))

    all_ends = ends_where(lambda i, n: True)
    cat_ends = [
        all_ends,
        ends_where(lambda i, n: n["pitch"] >= pmax - 2),
        ends_where(lambda i, n: i + 1 < len(notes)
                   and notes[i + 1]["pitch"] - n["pitch"] >= 7),
        ends_where(lambda i, n: i + 1 < len(notes)
                   and notes[i + 1]["pitch"] - n["pitch"] <= -7),
        ends_where(lambda i, n: n["end"] - n["start"] > p90),
    ]

    # Colors follow BAR_COUNT_DICT's palette so the theme stays coherent.
    kinds = ("note", "high", "up", "down", "long")
    colors = [(235, 235, 240), (255, 239, 85), (230, 89, 37),
              (45, 153, 232), (101, 227, 94)]
    if techniques_path:
        # Original singer's techniques (detect_techniques.py) in DAM 精密採点's
        # vocabulary and colour convention: しゃくり orange, こぶし blue,
        # フォール purple, ビブラート green. Event times are real-time, so
        # they're compared on the display timeline via their t_display.
        tech = _json.loads(Path(techniques_path).read_text())["events"]
        kinds = ("note", "shakuri", "kobushi", "fall", "vibrato")
        colors = [(235, 235, 240), (245, 140, 40), (50, 130, 240),
                  (170, 95, 230), (80, 205, 95)]
        cat_ends = [all_ends] + [
            _np.sort(_np.asarray([e["t_display"] for e in tech
                                  if e["type"] == k], float))
            for k in kinds[1:]]
    font_num = app.bar_count_font
    ICON = 40
    icons = [_minimal_icon(k, col, ICON) for k, col in zip(kinds, colors)]

    if techniques_path:
        _install_technique_marks(app, tech, dict(zip(kinds, colors)))

    X0, CY = 32, 365
    pill, slot_xs = _layout_counter_pill(
        font_num, [len(e) for e in cat_ends], icon_w=ICON)
    pill_pos = (X0, CY - pill.get_height() // 2)

    # The glossy flame/arrow PNGs MID2BAR stamps on note bars (draw_notes,
    # app.py:827) are the old visual language; blank them in this mode. The
    # jump/long info is already in the HUD counts and visible bar geometry.
    blank = pygame.Surface((1, 1), pygame.SRCALPHA)
    app.assets.icons = {k: blank for k in app.assets.icons}

    # Static range labels on the gauge's dark slot (x>=1400 was left unerased).
    small_font = pygame.font.Font(app.s.BAR_COUNT_FONT, 20)
    gx, gy = app.s.RANGE_GAUGE_POS
    gw, gh = app.s.RANGE_GAUGE_W, app.s.RANGE_GAUGE_H
    lab = (215, 215, 222)
    lo_s = small_font.render(_pitch_name(st["pmin"], use_flats), True, lab)
    hi_s = small_font.render(_pitch_name(st["pmax"], use_flats), True, lab)
    span_s = small_font.render(str(st["span"]), True, lab)
    key_s = small_font.render(key["name"], True, (160, 160, 172))
    # Swap the baked rainbow double-arrow (project_front x1416-1459, y350-382)
    # for a minimal ascending-bars glyph at the same spot.
    app.assets.project_front.fill((0, 0, 0, 0), pygame.Rect(1404, 332, 72, 60))
    range_icon = _minimal_icon("range", lab, 36)
    static_blits = [
        (range_icon, range_icon.get_rect(centerx=1438, centery=gy + gh // 2)),
        (lo_s, lo_s.get_rect(left=gx, top=gy + gh + 2)),
        (hi_s, hi_s.get_rect(right=gx + gw, top=gy + gh + 2)),
        (span_s, span_s.get_rect(right=gx + gw, bottom=gy - 2)),
        (key_s, key_s.get_rect(left=gx, bottom=gy - 2)),
    ]

    def _draw_song_info():
        t = app.current_time
        if w_real is not None:
            t = float(_np.interp(t, w_real, w_disp))
        app.screen.blit(pill, pill_pos)
        px, _ = pill_pos
        for icon, ends, color, (ix, nx) in zip(icons, cat_ends, colors, slot_xs):
            app.screen.blit(icon, icon.get_rect(left=px + ix, centery=CY))
            cnt = int(_np.searchsorted(ends, t, side="right"))
            num = font_num.render(str(cnt), True, color)
            app.screen.blit(num, num.get_rect(left=px + nx, centery=CY))
        for surf, rect in static_blits:
            app.screen.blit(surf, rect)

    app.draw_bar_count = _draw_song_info


def _expand_font_paths(settings_path: str) -> str:
    """Expand `~` in the lyric settings' FONT_PATH fields (MID2BAR opens them
    verbatim), so tracked configs can say `~/.local/share/fonts/...`. Returns
    the original path when nothing needs expanding, else a temp copy."""
    import json as _json
    import tempfile

    data = _json.loads(Path(settings_path).read_text(encoding="utf-8"))
    changed = False
    for section in data.values():
        fp = section.get("FONT_PATH") if isinstance(section, dict) else None
        if isinstance(fp, str) and fp.startswith("~"):
            section["FONT_PATH"] = os.path.expanduser(fp)
            changed = True
    if not changed:
        return settings_path
    fd, tmp = tempfile.mkstemp(prefix="lrc_settings_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        _json.dump(data, f, ensure_ascii=False, indent=2)
    return tmp


LIGATURE_STYLES = ("stroke", "ribbon", "ribbon2", "brush", "thread", "chain", "chain_wide")


def _install_ligatures(app, ligatures_path: str, style: str = "stroke") -> None:
    """EXPERIMENTAL 連筆: join the bars of one melisma mora, like joined-up
    (行書) handwriting, so a sustained vowel that moves through several pitches
    reads as ONE sung syllable instead of separate attacks. Pairs come from
    make_display_grid --out-ligatures (display-time starts of consecutive
    different-pitch notes on the same mora). Wraps draw_notes, so install
    BEFORE _apply_time_warp (display-time compare) and before the technique
    marks (which then draw on top).

    Styles (all but "stroke" are 4x-supersampled, drawn UNDER the bars so the
    bar edges stay crisp, and turn yellow exactly where the cursor has passed):
      stroke  straight slanted bar-to-bar stroke (first prototype)
      ribbon  full-height S-curve band: the bars flow into each other
      brush   S-curve that pinches in the middle (筆鋒: thick-thin-thick)
      thread  hairline S-curve between the bar ends (藕斷絲連)
      chain   the whole melisma is redrawn as ONE shape (its bars are hidden):
              flat at each pitch, smootherstep between them, constant vertical
              thickness (so steep turns thin out like a brush), bar-like
              rounded corners only at the two ends — no seams, no cap bumps
      chain_wide  chain with a longer, softer transition
    """
    import json as _json

    import numpy as _np
    import pygame
    from PIL import Image as _Image
    from PIL import ImageDraw as _ImageDraw

    pairs = _json.loads(Path(ligatures_path).read_text())["pairs"]
    starts_a = _np.asarray([a for a, _b in pairs], float)
    UNSUNG, SUNG = (255, 255, 255), (255, 224, 40)  # flat skin back / passed
    SS = 4  # supersampling factor
    original_draw_notes = app.draw_notes
    cache: dict = {}

    def page_alpha(page, t):
        a_in = (t - page["fade_in_time"]) / app.s.FADE_TIME
        a_out = (page["fade_out_time"] - t) / app.s.FADE_TIME
        return max(0.0, min(1.0, a_in, a_out))

    def find(notes, t0):
        best = min(notes, key=lambda n: abs(n["start"] - t0))
        return best if abs(best["start"] - t0) < 0.01 else None

    def stroke(a, b, color, alpha):
        th = max(4.0, a["height"] * 0.42)
        ia = min(a["width"] * 0.35, a["height"] * 1.2)
        ib = min(b["width"] * 0.35, b["height"] * 1.2)
        x0, y0 = a["x_start"] + a["width"] - ia, a["y"]
        x1, y1 = b["x_start"] + ib, b["y"]
        pad = int(th) + 2
        left, top = int(min(x0, x1)) - pad, int(min(y0, y1)) - pad
        w, h = int(abs(x1 - x0)) + 2 * pad, int(abs(y1 - y0)) + 2 * pad
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        p0, p1 = _np.array([x0 - left, y0 - top]), _np.array([x1 - left, y1 - top])
        d = p1 - p0
        n = _np.array([-d[1], d[0]]) / max(float(_np.hypot(*d)), 1e-6) * th / 2
        poly = [tuple(p0 + n), tuple(p1 + n), tuple(p1 - n), tuple(p0 - n)]
        col = (*color, 255)
        pygame.draw.polygon(surf, col, poly)
        pygame.draw.aalines(surf, col, True, poly)
        for c in (p0, p1):
            pygame.draw.circle(surf, col, (int(c[0]), int(c[1])), int(th / 2))
        surf.set_alpha(int(255 * alpha))
        app.screen.blit(surf, (left, top))

    def mask_for(a, b):
        """Coverage mask (float 0..1) + top-left of the S-curve sweep."""
        h = a["height"]
        ends = {"ribbon": 0.5, "ribbon2": 0.9, "brush": 0.5, "thread": 0.25}[style]
        x0 = a["x_start"] + a["width"] - min(h * ends, a["width"] * 0.45)
        x1 = b["x_start"] + min(h * ends, b["width"] * 0.45)
        ya, yb = a["y"], b["y"]
        key = (round(x0, 1), round(x1, 1), round(ya, 1), round(yb, 1), round(h, 1))
        if key in cache:
            return cache[key]
        pad = h / 2 + 2
        left, top = int(min(x0, x1) - pad), int(min(ya, yb) - pad)
        w, hh = int(abs(x1 - x0) + 2 * pad) + 2, int(abs(yb - ya) + 2 * pad) + 2
        img = _Image.new("L", (w * SS, hh * SS), 0)
        dr = _ImageDraw.Draw(img)
        dx = x1 - x0
        n = max(16, int(_np.hypot(dx, yb - ya) * SS / 2))
        for u in _np.linspace(0.0, 1.0, n):
            # cubic Bezier with horizontal tangents at both ends = S-curve
            m = 1 - u
            cx = m**3 * x0 + 3 * m * m * u * (x0 + dx / 2) + 3 * m * u * u * (x1 - dx / 2) + u**3 * x1
            cy = (m**3 + 3 * m * m * u) * ya + (3 * m * u * u + u**3) * yb
            if style in ("ribbon", "ribbon2"):
                r = h / 2
            elif style == "brush":
                r = h / 2 * (1 - 0.62 * _np.sin(_np.pi * u))
            else:  # thread
                r = h / 2 * (0.5 - 0.32 * _np.sin(_np.pi * u))
            px, py, rr = (cx - left) * SS, (cy - top) * SS, r * SS
            dr.ellipse((px - rr, py - rr, px + rr, py + rr), fill=255)
        img = img.resize((w, hh), _Image.BOX)
        cov = _np.asarray(img, _np.float32) / 255.0
        cache[key] = (cov, left, top)
        return cache[key]

    def sweep(a, b, cursor_x, alpha):
        cov, left, top = mask_for(a, b)
        hh, w = cov.shape
        rgba = _np.empty((hh, w, 4), _np.uint8)
        xs = _np.arange(w) + left + 0.5
        sung = (xs < cursor_x)[None, :]
        for k in range(3):
            rgba[..., k] = _np.where(sung, SUNG[k], UNSUNG[k])
        rgba[..., 3] = (cov * 255 * alpha).astype(_np.uint8)
        surf = pygame.image.frombuffer(rgba.tobytes(), (w, hh), "RGBA")
        app.screen.blit(surf, (left, top))


    # ---- chain styles: one continuous shape per melisma ----
    chains: list[list[float]] = []
    for ta, tb in pairs:
        if chains and abs(chains[-1][-1] - ta) < 1e-6:
            chains[-1].append(tb)
        else:
            chains.append([ta, tb])
    chain_starts = _np.asarray([c[0] for c in chains], float)
    HIDDEN = -4242  # asset channel with transparent bars; not auto-played
    hidden_ready = []

    def ensure_hidden(ch):
        if hidden_ready:
            return
        src = app.assets.bars[ch]
        app.assets.bars[HIDDEN] = {
            typ: {k: pygame.Surface(img.get_size(), pygame.SRCALPHA) for k, img in d.items()}
            for typ, d in src.items()}
        hidden_ready.append(True)

    def chain_mask(notes):
        h = notes[0]["height"]
        k = 1.4 if style == "chain_wide" else 0.8
        xs0, xs1 = notes[0]["x_start"], notes[-1]["x_start"] + notes[-1]["width"]
        key = ("chain", style, round(xs0, 1), round(xs1, 1), tuple(round(n["y"], 1) for n in notes), round(h, 1))
        if key in cache:
            return cache[key]
        left, top = int(xs0) - 2, int(min(n["y"] for n in notes) - h / 2) - 2
        w = int(xs1 - left) + 3
        hh = int(max(n["y"] for n in notes) + h / 2 - top) + 3
        X = left + (_np.arange(w * SS) + 0.5) / SS
        Y = top + (_np.arange(hh * SS) + 0.5) / SS
        yc = _np.full_like(X, notes[0]["y"])
        for a, b in zip(notes, notes[1:]):
            xb = (a["x_start"] + a["width"] + b["x_start"]) / 2
            L = min(k * h, a["width"] * 0.45, b["width"] * 0.45)
            u = _np.clip((X - (xb - L)) / (2 * L), 0.0, 1.0)
            yc += (b["y"] - a["y"]) * u**3 * (u * (u * 6 - 15) + 10)  # smootherstep
        half = _np.full_like(X, h / 2)
        r = 0.15 * h  # the flat skin's corner radius
        for edge, d in ((xs0, X - xs0), (xs1, xs1 - X)):
            m = d < r
            half[m] = h / 2 - r + _np.sqrt(_np.clip(r * r - (r - d[m]) ** 2, 0, None))
        inside = (_np.abs(Y[:, None] - yc[None, :]) <= half[None, :]) & (X >= xs0)[None, :] & (X <= xs1)[None, :]
        cov = inside.reshape(hh, SS, w, SS).mean(axis=(1, 3)).astype(_np.float32)
        cache[key] = (cov, left, top)
        return cache[key]

    def blit_cov(cov, left, top, cursor_x, alpha, target):
        hh, w = cov.shape
        rgba = _np.empty((hh, w, 4), _np.uint8)
        sung = ((_np.arange(w) + left + 0.5) < cursor_x)[None, :]
        for k in range(3):
            rgba[..., k] = _np.where(sung, SUNG[k], UNSUNG[k])
        rgba[..., 3] = (cov * 255 * alpha).astype(_np.uint8)
        target.blit(pygame.image.frombuffer(rgba.tobytes(), (w, hh), "RGBA"), (left, top))

    def draw_chains():
        t = app.current_time
        hidden = []
        todo = []
        for pi in app.get_current_page_i():
            page = app.pages[pi]
            pa = page_alpha(page, t)
            if not page["notes"]:
                continue
            lo = int(_np.searchsorted(chain_starts, page["start_time"] - 0.01))
            hi = int(_np.searchsorted(chain_starts, page["end_time"] + 0.01))
            cursor_x = (app.s.BAR_AREA_LEFT + (t - page["start_time"])
                        / (page["end_time"] - page["start_time"]) * app.s.BAR_AREA_WIDTH)
            for c in chains[lo:hi]:
                notes = [find(page["notes"], x) for x in c]
                if any(n is None for n in notes):
                    continue
                ensure_hidden(notes[0]["channel"])
                for n in notes:
                    hidden.append((n, n["channel"]))
                    n["channel"] = HIDDEN
                if pa > 0:
                    todo.append((notes, cursor_x, pa))
        try:
            result = original_draw_notes()
        finally:
            for n, ch in hidden:
                n["channel"] = ch
        for notes, cursor_x, pa in todo:
            cov, left, top = chain_mask(notes)
            blit_cov(cov, left, top, cursor_x, pa, app.screen)
        return result

    def draw_ligatures():
        t = app.current_time
        for pi in app.get_current_page_i():
            page = app.pages[pi]
            pa = page_alpha(page, t)
            if pa <= 0 or not page["notes"]:
                continue
            cursor_x = (app.s.BAR_AREA_LEFT + (t - page["start_time"])
                        / (page["end_time"] - page["start_time"]) * app.s.BAR_AREA_WIDTH)
            lo = int(_np.searchsorted(starts_a, page["start_time"] - 0.01))
            hi = int(_np.searchsorted(starts_a, page["end_time"] + 0.01))
            for ta, tb in pairs[lo:hi]:
                a, b = find(page["notes"], ta), find(page["notes"], tb)
                if a is None or b is None:
                    continue
                if style == "stroke":
                    sung = t >= (a["end"] + b["start"]) / 2
                    stroke(a, b, SUNG if sung else UNSUNG, pa)
                else:
                    sweep(a, b, cursor_x, pa)

    def draw_notes_with_ligatures():
        if style.startswith("chain"):
            return draw_chains()
        if style in ("stroke", "ribbon2"):  # on top of the bars
            result = original_draw_notes()
            draw_ligatures()
            return result
        draw_ligatures()
        return original_draw_notes()

    app.draw_notes = draw_notes_with_ligatures


def _install_technique_marks(app, events, colors) -> None:
    """Stamp each technique on the bar where the original singer did it.

    DAM 精密採点 (II onward) drops the technique icon at the detection point
    on the guide bar; we do the same on MID2BAR's pages. Wraps draw_notes, so
    it must be installed BEFORE _apply_time_warp: inside the warp wrapper
    app.current_time is display time, matching the events' t_display. A mark
    appears the moment the cursor reaches it (pops in over 0.15 s) and then
    fades out with its page, using MID2BAR's own page alpha formula.
    """
    import numpy as _np
    import pygame

    SIZE, POP = 30, 0.15
    glyphs = {}
    for kind, col in colors.items():
        if kind == "note":
            continue
        badge = pygame.Surface((SIZE + 8, SIZE + 8), pygame.SRCALPHA)
        pygame.draw.rect(badge, (10, 10, 16, 190), badge.get_rect(),
                         border_radius=9)
        badge.blit(_minimal_icon(kind, col, SIZE), (4, 4))
        glyphs[kind] = badge
    evs = sorted((e for e in events if e["type"] in glyphs),
                 key=lambda e: e["t_display"])
    ev_t = _np.asarray([e["t_display"] for e in evs], float)
    original_draw_notes = app.draw_notes

    def page_alpha(page, t):
        a_in = (t - page["fade_in_time"]) / app.s.FADE_TIME
        a_out = (page["fade_out_time"] - t) / app.s.FADE_TIME
        return max(0.0, min(1.0, a_in, a_out))

    def pick_bar(bars, te, kind):
        """Bar the event belongs to. A scoop belongs to the bar it rises
        into (first bar ending after te); a fall to the bar it leaves (last
        bar starting at/before te); turns/vibrato to the bar under te."""
        eps = 0.08
        if kind == "shakuri":
            cand = [n for n in bars if n["end"] > te + eps] or bars[-1:]
            return min(cand, key=lambda n: n["start"])
        if kind == "fall":
            cand = [n for n in bars if n["start"] <= te + eps] or bars[:1]
            return max(cand, key=lambda n: n["start"])
        return min(bars, key=lambda n: (
            0 if n["start"] <= te < n["end"]
            else min(abs(te - n["start"]), abs(te - n["end"]))))

    def draw_notes_with_marks():
        result = original_draw_notes()
        t = app.current_time
        for pi in app.get_current_page_i():
            page = app.pages[pi]
            pa = page_alpha(page, t)
            if pa <= 0 or not page["notes"]:
                continue
            p0, p1 = page["start_time"], page["end_time"]
            lo = int(_np.searchsorted(ev_t, p0, side="left"))
            hi = int(_np.searchsorted(ev_t, min(p1, t), side="right"))
            placed = []
            for e in evs[lo:hi]:
                te = e["t_display"]
                bar = pick_bar(page["notes"], te, e["type"])
                x = (app.s.BAR_AREA_LEFT
                     + (te - p0) / (p1 - p0) * app.s.BAR_AREA_WIDTH)
                # display-grid edges drift a little from the real-time notes
                # the detector used; keep the mark on its own bar
                x = min(max(x, bar["x_start"] + SIZE / 2),
                        max(bar["x_start"] + SIZE / 2, bar["x_end"] - SIZE / 2))
                y = bar["y"] - bar["height"] / 2 - SIZE / 2 - 8
                # nudge sideways so two marks on one spot don't overlap
                while any(abs(x - px) < SIZE + 6 and abs(y - py) < 4
                          for px, py in placed):
                    x += SIZE + 6
                placed.append((x, y))
                g = glyphs[e["type"]]
                pop = min(1.0, (t - te) / POP)
                if pop < 1.0:
                    sc = 1.0 + 0.5 * (1 - pop)
                    g = pygame.transform.smoothscale(
                        g, (int(g.get_width() * sc), int(g.get_height() * sc)))
                g = g.copy()
                g.set_alpha(int(255 * pa * max(pop, 0.2)))
                app.screen.blit(g, g.get_rect(center=(int(x), int(y))))
        return result

    app.draw_notes = draw_notes_with_marks


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
@click.option("--bg-mask", "bg_mask", default=None,
              help="Black-out boxes burned into the background during "
                   "normalization, as \"x,y,w,h\" in 1920x1080 output pixels "
                   "(\";\" separates several). Used to cover subtitles already "
                   "burned into a source MV so they don't clash with our lyric "
                   "layer. Applied after the scale/pad, so coordinates are "
                   "output-frame coordinates.")
@click.option("--lrc-settings", "lrc_settings_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="path to lyrics_settings/settings_default.json (default: bundled).")
@click.option("--lyric-halo", "lyric_halo", default=None,
              help="Dark outer ring (二重フチ) around lyric/ruby text, as "
                   "\"#RRGGBB:LYRIC_W:RUBY_W\". The --lrc-settings STROKE_WIDTH "
                   "values must be the total (inner + halo) widths, e.g. "
                   "config/mid2bar_lyrics_p1.json with \"#16131F:4:2\".")
@click.option("--ligatures", "ligatures_path", type=click.Path(exists=True, dir_okay=False),
              default=None, help="EXPERIMENTAL: ligatures JSON from make_display_grid "
                   "--out-ligatures; joins one mora's melisma bars with a stroke.")
@click.option("--ligature-style", "ligature_style", type=click.Choice(list(LIGATURE_STYLES)),
              default="stroke", show_default=True, help="EXPERIMENTAL 連筆 drawing style (see _install_ligatures).")
@click.option("--stills", "stills", default=None,
              help="Debug: comma-separated REAL times; save PNG frames to <out>_stills/ instead of encoding a video.")
@click.option("--key", "key_override", default=None,
              help="--hud songinfo only: hand-verified song key (e.g. Bb, F#m) "
                   "shown instead of the detector's guess.")
@click.option("--miter-stroke/--round-stroke", "miter_stroke", default=False,
              help="Draw lyric/ruby outlines with sharp MITER joins (DAM-like) "
                   "instead of PIL's round joins.")
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
@click.option("--techniques", "techniques_path", type=click.Path(exists=True, dir_okay=False),
              default=None,
              help="--hud songinfo only: techniques JSON from detect_techniques.py. "
              "Replaces the high/leap/long chart counters with the original "
              "singer's しゃくり/こぶし/フォール/ビブラート counts (DAM colours).")
@click.option("--hud", type=click.Choice(["legacy", "songinfo", "none"]),
              default="legacy", show_default=True,
              help="top counter strip: legacy=MID2BAR cumulative counters "
              "(min/max hidden, current behaviour), songinfo=static whole-song "
              "stats pill, none=blank. The range gauge is unaffected.")
def main(
    audio_path: str,
    midi_path: str,
    lrc_path: str,
    out_path: str,
    bg_path: str | None,
    bg_mask: str | None,
    lrc_settings_path: str | None,
    lyric_halo: str | None,
    miter_stroke: bool,
    key_override: str | None,
    ligatures_path: str | None,
    ligature_style: str,
    stills: str | None,
    app_settings_path: str | None,
    assets_json_path: str | None,
    time_warp_path: str | None,
    techniques_path: str | None,
    hud: str,
) -> None:
    # Defaults inside the bundled MID2BAR-Player tree. Resolve to absolute
    # paths BEFORE the os.chdir below — otherwise relative override paths
    # (e.g. `--app-settings my_settings.json` from the user's CWD) would
    # silently re-anchor to MID2BAR_DIR after the chdir and stop resolving.
    if time_warp_path is not None:
        time_warp_path = str(Path(time_warp_path).resolve())
    if techniques_path is not None:
        techniques_path = str(Path(techniques_path).resolve())
    if ligatures_path is not None:
        ligatures_path = str(Path(ligatures_path).resolve())
    lrc_settings_path = _expand_font_paths(str(Path(
        lrc_settings_path or (MID2BAR_DIR / "lyrics_settings" / "settings_default.json")
    ).resolve()))
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
        # --bg-mask boxes are appended AFTER scale/pad, so they are expressed
        # in final 1920x1080 coordinates regardless of the source resolution.
        vf = ("scale=1920:1080:force_original_aspect_ratio=decrease,"
              "pad=1920:1080:(ow-iw)/2:(oh-ih)/2")
        # "x,y,w,h" = solid black for the whole video (letterbox subtitle
        # band); "x,y,w,h@t0-t1" = heavy blur only between t0..t1 s (captions
        # burned mid-picture, where a black slab would punch a hole in the MV)
        blurs = []
        for box in filter(None, (bg_mask or "").split(";")):
            geom, _, span = box.partition("@")
            try:
                x, y, w, h = (int(float(n)) for n in geom.split(","))
                t0, t1 = (float(n) for n in span.split("-")) if span else (None, None)
            except ValueError:
                raise click.UsageError(
                    f"--bg-mask box must be 'x,y,w,h[@t0-t1]': got {box!r}")
            if span:
                blurs.append((x, y, w, h, t0, t1))
            else:
                vf += f",drawbox=x={x}:y={y}:w={w}:h={h}:color=black:t=fill"
        for k, (x, y, w, h, t0, t1) in enumerate(blurs):
            vf += (f"[m{k}];[m{k}]split[a{k}][b{k}];"
                   f"[b{k}]crop={w}:{h}:{x}:{y},gblur=sigma=14:steps=3[c{k}];"
                   f"[a{k}][c{k}]overlay={x}:{y}:enable='between(t,{t0},{t1})'")
        if ext in {".mp4", ".webm", ".mov", ".mkv"}:
            cmd = [
                "ffmpeg", "-y", "-i", str(src),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
                "-vf", vf,
                str(converted),
            ]
        elif ext in {".png", ".jpg", ".jpeg", ".webp"}:
            cmd = [
                "ffmpeg", "-y", "-loop", "1", "-i", str(src),
                "-t", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-vf", vf,
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
    if miter_stroke:
        _install_miter_stroke()
    if lyric_halo:
        _install_lyric_halo(lyric_halo)

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
    if ligatures_path:
        _install_ligatures(app, ligatures_path, ligature_style)
    if hud == "songinfo":
        _install_song_info_panel(app, time_warp_path, audio_abs, techniques_path, key_override)
    elif hud == "none":
        _erase_baked_hud_strip(app)
        app.draw_bar_count = lambda: None
    else:
        _hide_minmax_columns(app)
    if time_warp_path is None:
        _hide_notes_without_visible_lyrics(app)
    else:
        # display-grid MIDIs are pre-gated and live on a warped timeline;
        # comparing their note times against real LRC times would be wrong
        _apply_time_warp(app, time_warp_path)
    _zero_lag_time(app)
    if stills:
        import pygame
        stills_dir = Path(out_abs).with_suffix("")
        stills_dir = stills_dir.parent / f"{stills_dir.name}_stills"
        stills_dir.mkdir(parents=True, exist_ok=True)
        for tok in stills.split(","):
            app.current_time = float(tok)
            app.draw()
            pygame.image.save(app.screen, str(stills_dir / f"{float(tok):.2f}.png"))
        print(f"[render] stills written to {stills_dir}", flush=True)
        return
    _press_space_after_init()
    app.run()
    print(f"[render] mp4 written to {out_abs}", flush=True)


if __name__ == "__main__":
    main()
