# render-song

Render a karaoke video for a song using the canonical version profile.

## Usage

`/render-song <song-id>` — full pipeline: GAME chain → display grid → LRC → render MP4.

## Instructions

1. **Read the canonical version** from `config/versions.json`. The `"canonical"` key points to the active profile (e.g., `"v14"`). All parameters below come from that profile — do NOT hardcode values.

2. **Verify prerequisites exist** in `outputs/<song-id>/`:
   - `vocals.wav` (from separation)
   - `instrumental.wav`
   - `aligned_midi.json` (from MMS/classic timing + line_end_repair)
   - `melody_markers.scorefix.mid` (classic chain fallback)
   - `rmvpe_f0.npz`
   - `melody_quantized.mid.bpm.txt`
   
   If missing, tell the user which prerequisite is absent and suggest running the Snakemake pipeline first:
   ```
   snakemake --rerun-triggers mtime -j1 outputs/<song>/aligned_midi.json outputs/<song>/melody_markers.scorefix.mid
   ```

3. **Check for per-song overrides**:
   - `overrides/<song-id>_pitch_patch.json` → pass as `--pitch-patch`
   - `outputs/<song-id>/rms_segments.json` → pass as `--rms-segments`

4. **Run the GAME chain** (`scripts/run_game_chain.py`), which internally does:
   GAME extract → postfix → melody_union → make_display_grid.
   
   Read the version profile's `pitch_chain.game_seg_threshold` — it must match `GAME_SEG_THRESHOLD` in `run_game_chain.py`. If they differ, update the script constant.

   ```bash
   source ~/venvs/karaoke-jp/bin/activate
   python scripts/run_game_chain.py \
     --vocals outputs/<song>/vocals.wav \
     --fallback-midi outputs/<song>/melody_markers.scorefix.mid \
     --f0 outputs/<song>/rmvpe_f0.npz \
     --aligned outputs/<song>/aligned_midi.json \
     --bpm-file outputs/<song>/melody_quantized.mid.bpm.txt \
     --language <pitch_chain.game_language> \
     [--pitch-patch overrides/<song>_pitch_patch.json] \
     [--rms-segments outputs/<song>/rms_segments.json] \
     --out outputs/<song>/melody_markers.gamescore.mms.mid
   ```

5. **Export LRC** (for the wipe layer):
   ```bash
   python scripts/export_lrc.py \
     outputs/<song>/aligned_midi.json \
     -o outputs/<song>/karaoke.mms.lrc \
     --block-size <render.lrc_block_size>
   ```

6. **Mix audio**:
   ```bash
   python scripts/mix_audio.py \
     --instrumental outputs/<song>/instrumental.wav \
     --vocals outputs/<song>/vocals.wav \
     --out outputs/<song>/mixed.wav \
     --vocal-ratio <render.vocal_ratio>
   ```

7. **Render MP4** — use the render venv:
   ```bash
   SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
   ~/venvs/karaoke-jp-render/bin/python scripts/render_mp4.py \
     --audio outputs/<song>/mixed.wav \
     --midi outputs/<song>/melody_markers.gamescore.mms.mid \
     --lrc outputs/<song>/karaoke.mms.lrc \
     --out outputs/<song>/karaoke_<VERSION>_full.mp4 \
     --app-settings config/mid2bar_settings.json \
     --assets render_assets/assets_flat.json \
     --hud songinfo \
     --time-warp outputs/<song>/melody_markers.gamescore.mms.warp.json \
     [--background songs/<song>/background.*]
   ```
   
   Replace `<VERSION>` with the canonical version ID (e.g., `v14`).
   `--assets` is the v14 flat bar skin (see MEMORY.md "flat bar skin") —
   omitting it silently falls back to MID2BAR's glossy pink default.
   `--hud songinfo` is the v14 info HUD (icon counters + range/key gauge,
   harmony-based key detection); `--hud legacy` restores MID2BAR's old
   cumulative counters, `--hud none` blanks the strip.

8. **Report** the output path and file size. Remind user to ear-test before promoting.

## Version switching

All version-specific parameters come from `config/versions.json → profiles[canonical]`. When the canonical version changes (e.g., v14 → v15), this skill automatically uses the new profile's parameters. No skill edits needed.

If `run_game_chain.py` has hardcoded constants (like `GAME_SEG_THRESHOLD`) that disagree with the version profile, update the script to match — the version profile is the source of truth.
