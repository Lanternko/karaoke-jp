# alignment_gold (private)

The `*.gold.tsv` line-gold files carry full lyric text, so they live in the
private `karaoke-jp-gold` repo (`gold/alignment_gold/`), untracked here.
Only aggregate metrics are public (docs/alignment-benchmark.md).

Restore on a dev machine (needs private-repo access):

    git submodule update --init gold
    cp gold/alignment_gold/*.gold.tsv data/alignment_gold/
