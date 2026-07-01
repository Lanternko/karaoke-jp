# render-portrait

Render a portrait (9:16) karaoke video. Layout top→bottom: bars A / bars B
/ MV (centered, sides cropped) / lyric A / lyric B.

## Usage

`/render-portrait <song-id>` — generate portrait grid + render 1080×1920 MP4.

## Instructions

1. **Check prerequisites** in `outputs/<song-id>/`:
   - `melody_markers.gamescore.mms.union.mid` — the REAL-TIME union melody.
     Written by `run_game_chain.py` since 2026-06-12; older songs only have
     the display MIDI, so re-run the chain (see `/render-song` step 4) to
     get it. Do NOT feed the display MIDI (`.gamescore.mms.mid`) without
     `--warp`: its times include page leads/count-ins and wipe against the
     wrong clock.
   - `aligned_midi.json`, `melody_quantized.mid.bpm.txt`, `mixed.wav`
   - MV background: `songs/<song-id>/background.mp4` or
     `outputs/<song-id>/_background.mp4`. A song without an MV is a poor
     portrait demo (per Kojek: use night-dancer over byoushin until its MV
     is fetched).

2. **Generate portrait grid** (main venv):
   ```bash
   source ~/venvs/karaoke-jp/bin/activate
   python scripts/make_portrait_grid.py \
     --midi outputs/<song>/melody_markers.gamescore.mms.union.mid \
     --bpm-file outputs/<song>/melody_quantized.mid.bpm.txt \
     --aligned outputs/<song>/aligned_midi.json \
     [--pitch-patch overrides/<song>_pitch_patch.json] \
     [--rms-segments outputs/<song>/rms_segments.json] \
     --vocals outputs/<song>/vocals.wav \
     --rmvpe-f0 outputs/<song>/rmvpe_f0.npz \
     --pyin-f0 outputs/<song>/pyin_f0.npz \
     --out outputs/<song>/portrait_grid.json
   ```

   `--vocals/--rmvpe-f0/--pyin-f0` (all three, or none) arm the mora-aware
   note cleanup (`scripts/note_cleanup.py`, whale survey 2026-07-02):
   readings are expanded to TRUE morae (not surface chars — 静=し+ず), and
   nearby pitch variants (≤3 semitones) default to one dominant score pitch
   per mora. Wide melisma is the exception and each plateau must earn acoustic
   support. Same-pitch shatters merge; local or globally-high RMVPE octave-up
   islands repitch −12. Tail fragments are cross-checked against stem energy +
   both trackers, while notes outside every lyric mora drop even when pitched
   (lyrics are truth; accompaniment can fool both trackers). Geometry-only
   merges still run without sidecars; destructive drops do not. Every action
   logs as `[note-cleanup]` — review it per song.

3. **Render portrait MP4** (render venv):
   ```bash
   source ~/venvs/karaoke-jp-render/bin/activate
   python scripts/render_portrait.py \
     --grid outputs/<song>/portrait_grid.json \
     --audio outputs/<song>/mixed.wav \
     --bg <background video> \
     --out outputs/<song>/karaoke_portrait_<VERSION>.mp4
   ```

4. **Report** output path and file size. ~5-10 min for a 4-min song
   (Pillow frame-by-frame at 60fps, composited over the MV by ffmpeg).

## Parameters

- `--quarters-per-row 8.0` — half the horizontal page (16.0)
- `--phrase-gap-units 0.75`, `--phrase-split-gap 0.8` — phrase flow within a row
- `--gap-units`, `--breath-gap`, `--breath-units`, `--min-note` — same as horizontal grid
- `--lead-max 8.0` — max preview lead before a line starts (interludes blank the rows)
- `--linger 4.0` — how long a finished line may outlive its last note
- `--warp <warp.json>` — only for legacy display MIDIs without a union sibling
- `--backing <backing.json>` — call-response / antiphonal BACKING VOCALS only (e.g.
  muroshi's 〈…〉 echo lines), rendered as a dim bottom strip tied to each lead line.
  **Never feed romaji here.** Romaji (`romaji.txt`) is a reading-verification answer-key
  that is NEVER printed on the video (see `/add-song` step 2–3); it corrects furigana via
  `romaji_overrides.py`, nothing more. Songs without call-response omit `--backing`.

## Hand corrections (`--pitch-patch overrides/<song>_pitch_patch.json`)

A JSON list of display-only fixes for cases the automated chain can't nail — it
never edits `aligned_midi.json`. Entry types:

- `{"at": t, "pitch": p, "start": s, "end": e}` — insert/replace a pitch-bar note.
- `{"drop_notes": [lo, hi]}` — remove phantom notes in a window (separation bleed /
  interlude runs with no lyric).
- `{"melisma_split": …}` — split one long note into repeated attacks.
- `{"lyric_retime": [lo, hi], "to": [t0, t1]}` — a lyric line stretched across an
  interlude; linearly RESCALES its chars onto the true span. Keeps internal ratios —
  use when the shape is right but the window is wrong.
- `{"lyric_recut": <line_time_start>, "chars": [[s, e], …]}` — set each char's wipe
  window EXPLICITLY (one [start, end] per display char). For alignment COLLAPSE — a
  long sparse outro where one mora absorbs seconds and its neighbours get ~0 s and
  flash by ("漏字/沒聽到"). `lyric_retime` can't fix that (a rescale keeps the
  collapse); recut replaces the timing. Ground the windows in the vocals' voiced
  segments (RMVPE F0 `f0 > 0`). Reference: Whale L38 白く微睡みながら (33 s outro,
  two phrases split by a 10 s instrumental). RMVPE/pYIN can both follow
  accompaniment bleed, so the final cutoff still requires ear confirmation.
  Always mark
  `"note": "HAND CORRECTION … NEEDS EAR-CONFIRM"`.

## Architecture

Two INDEPENDENT line systems, each alternating display rows A/B:

- **Bar lines** — snake packing: phrases fill a row until the 8q budget
  runs out, then jump to the next row. NOT tied to lyric line boundaries.
- **Lyric lines** — one aligned_midi.json line each (JOYSOUND subtitle
  style), auto-fit font, ruby above kanji, per-char partial wipe.

Every element wipes by its own real_start/real_end against the playback
clock. Flip rule (JOYSOUND): a line leaves its row only when the NEXT line
(other row) starts being sung — long sustains stay put; in interludes the
row clears after `--linger` and the next line previews ≤ `--lead-max`
before its start. A white vertical cursor marks the active bar row; across a
rest it GLIDES from the previous note's right edge to the next note's left edge
at constant real-time speed (over the pause's own duration) instead of
teleporting — the blank gap stays on screen, only the cursor sweeps it.

Note gating: lyric windows ∩ (RMS voiced ∪ MMS char evidence). The char
windows rescue softly-sung passages the RMS VAD misses (night-dancer's
Tu-tu-lu hook) while interludes — which have no chars — still drop
separation-bleed ghost notes. (The horizontal display grid still gates by
RMS only; carry this fix into v15.)

Output: `portrait_grid.json` (`bar_lines` + `lyric_lines`) +
`karaoke_portrait_*.mp4`. Renderer: Pillow overlay frames piped to ffmpeg;
MV composited under the overlay by filter_complex (no MID2BAR dependency).
