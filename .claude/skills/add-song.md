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

2. **Create lyrics + romaji answer-key**:
   - Write the user's Japanese lyrics verbatim to `songs/<song-id>/lyrics.txt` (one sung
     phrase per line; blank lines separate stanzas and are ignored by line indexing).
     Do NOT use ASR/LLM to generate lyrics — the user-provided text is the ground truth.
   - If the user also supplies a **romanization**, write it to `songs/<song-id>/romaji.txt`,
     one line per non-blank lyrics.txt line (1:1). **Romaji is a reading-verification
     answer-key ONLY — it is NEVER rendered on the video.** It records what the singer
     actually sings, so it disambiguates gikun / rare readings (生僻字) that fugashi
     guesses wrong. (Do NOT feed it to the renderer's `--backing`; that flag is for
     call-response backing vocals — see `/render-portrait`.)

3. **Verify readings against the romaji answer-key** (this is the whole point of romaji):
   ```bash
   snakemake --rerun-triggers mtime -j1 outputs/<song-id>/tokens.json   # fugashi readings
   python scripts/romaji_overrides.py --tokens outputs/<song-id>/tokens.json \
     --romaji songs/<song-id>/romaji.txt --out overrides/<song-id>.json --dry-run
   ```
   Read every `+ 漢字: fugashi → romaji` candidate and **human-vet each one** — the
   romanizer is fuzzy at ん / particle (は・へ・を) / long-vowel boundaries and throws
   false positives (e.g. Whale flagged `自分 じぶん→じぶ`, which is wrong — reject it).
   `~ unpaired` lines merely failed to auto-anchor (repeats, ad-libs) and are not errors.
   Write ONLY vetted corrections, in the **flat** format the existing files use (NOT a
   `{"readings": …}` wrapper):
   ```json
   {"漢字": "よみ", "地球": "ほし"}
   ```
   Existing (human) entries always win over auto-extraction. If no romaji was supplied,
   hand-check unusual kanji (義訓, name readings) the same way. Re-run without `--dry-run`
   to write the vetted file (or edit it by hand); the pipeline then re-tokenizes with
   `--override`, so the corrected readings flow into BOTH the furigana display and the
   MMS forced-alignment (romanized tokens).

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
