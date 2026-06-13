# add-song

Add a new song to the karaoke-jp pipeline.

## Usage

`/add-song <youtube-url-or-title>` — download, separate, and prepare a song directory.

## Instructions

1. **Download** using `scripts/download_song.py`:
   ```bash
   source ~/venvs/karaoke-jp/bin/activate
   python scripts/download_song.py "<url>" -o songs/
   ```
   This creates `songs/<song-id>/source.wav` (and optionally a background video).
   
   If user provides a title instead of URL, ask for the URL. Never guess YouTube URLs.

2. **Create lyrics file**: Ask the user to paste the Japanese lyrics. Write them to `songs/<song-id>/lyrics.txt`. Do NOT use ASR/LLM to generate lyrics — the user-provided text is the ground truth.

3. **Check for reading overrides**: If the song has unusual kanji readings (義訓, name readings, etc.), create `overrides/<song-id>.json` with reading corrections. Format:
   ```json
   {"readings": {"漢字": "よみかた"}}
   ```

4. **Run the Snakemake pipeline** up to `aligned_midi.json`:
   ```bash
   snakemake --rerun-triggers mtime -j1 \
     outputs/<song-id>/aligned_midi.json \
     outputs/<song-id>/melody_markers.scorefix.mid
   ```
   This executes: separate → {melody, rmvpe_f0, pyin_f0} → {tokenize, asr → align} → {mms_align → line_end_repair, melody → quantize → markers → octave_fix → score_chain}.

5. **Run render-song**: Once prerequisites are ready, invoke `/render-song <song-id>` to produce the karaoke MP4 using the canonical version profile.

6. **Remind user to ear-test** the output before considering it done. Per-song pitch patches (`overrides/<song-id>_pitch_patch.json`) may be needed after the first listen.

## Notes

- Song IDs from yt-dlp are title slugs. Verify the ID is filesystem-safe.
- `--no-video` if the YouTube upload is a "Lyric Video" (burned-in subtitles conflict with our lyric layer).
- NEVER upload outputs. 著作権法 30 条 private use only.
