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
     --out outputs/<song>/portrait_grid.json
   ```

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
before its start. A white vertical cursor marks the active bar row.

Note gating: lyric windows ∩ (RMS voiced ∪ MMS char evidence). The char
windows rescue softly-sung passages the RMS VAD misses (night-dancer's
Tu-tu-lu hook) while interludes — which have no chars — still drop
separation-bleed ghost notes. (The horizontal display grid still gates by
RMS only; carry this fix into v15.)

Output: `portrait_grid.json` (`bar_lines` + `lyric_lines`) +
`karaoke_portrait_*.mp4`. Renderer: Pillow overlay frames piped to ffmpeg;
MV composited under the overlay by filter_complex (no MID2BAR dependency).
