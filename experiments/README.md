# Experiments Moved

The blind lyrics transcription benchmark has been split out to:

```text
/home/kojiek/side_projects/lyrics-transcription-benchmark/
```

`karaoke-jp` remains the known-lyrics karaoke generation project.

## Alignment Ablation Manifests

Local gold fixtures for the three current karaoke regression songs live in:

```text
data/alignment_gold/
```

The ablation runner uses those files by default for haru-hikage, tuki-zero, and
chidori.

`literature_alignment_manifest.example.json` is a schema example for evaluating
converted literature datasets with `scripts/run_alignment_ablation.py`.

It intentionally does not include raw benchmark data. Convert each dataset into
the project sidecar shape first:

```text
data/literature_alignment/<dataset>/aligned.json
data/literature_alignment/<dataset>/melody.mid
data/literature_alignment/<dataset>/vocals.wav
data/literature_alignment/<dataset>/gold.tsv
```

Until those files exist locally, the ablation runner reports the literature
entries as skipped rather than treating them as passed.
