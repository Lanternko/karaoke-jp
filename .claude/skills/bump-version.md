# bump-version

Create a new display/pipeline version profile and optionally promote it to canonical.

## Usage

`/bump-version` — interactive: review current canonical, create next version with changes.

## Instructions

1. **Read current state** from `config/versions.json`:
   - Show the current canonical version and its key parameters
   - Show a diff summary of what the user wants to change

2. **Create the new profile**: Copy the current canonical profile, apply the user's requested changes, and add it to `config/versions.json` under the next version key (e.g., `v14` → `v15`).

   Required fields for every profile:
   ```json
   {
     "description": "...",
     "date": "YYYY-MM-DD",
     "pitch_chain": { "backend", "game_language", "game_seg_threshold", "postfix_flags", "use_melody_union" },
     "timing": { "source", "line_end_repair": { "tail_top_db", "next_guard", "tail_gap" } },
     "display_grid": { "quarters_per_page", "gap_units", "phrase_gap_units", "lead_units", "phrase_split_gap", "min_note", "breath_gap", "breath_units", "count_in_quarters", "flip_delay", "note_window_margin", "note_tail_allowance" },
     "skin": { "type" },
     "render": { "vocal_ratio", "lrc_block_size" }
   }
   ```

3. **Update hardcoded constants** in scripts if any parameter changed:
   - `scripts/run_game_chain.py`: `GAME_SEG_THRESHOLD`, `POSTFIX_FLAGS`
   - `Snakefile`: `TIMING_SOURCE`, `VOCAL_RATIO`, `LRC_BLOCK_SIZE`, `QUARTERS_PER_PAGE`, line_end_repair flags
   - `scripts/make_display_grid.py`: CLI defaults (gap_units, breath_gap, etc.)

4. **Ask before promoting**: Don't change the `"canonical"` pointer until the user has ear-tested a render with the new version. Suggest:
   ```
   /render-song <song-id>  (renders using current canonical)
   ```
   Then manually render one song with the new version's parameters for A/B comparison.

5. **Promote**: Once the user confirms, update `"canonical"` in `config/versions.json` to the new version.

6. **Update CLAUDE.md** if the version change affects the project status description (e.g., new display grid version, new timing source).

## Version naming

Sequential integers: v14, v15, v16, ... Never skip numbers. The description field explains what changed.
