"""Concatenate multiple ctc_patch.json files into one.

PoC tool. Each input patch covers a disjoint set of aligned-line indices
(bridge, chorus, etc.). Merging is line-index ordered concatenation; on
overlap (same line patched twice) the run aborts so the user resolves
manually rather than letting one patch silently overwrite another.

Usage
-----
    python scripts/merge_ctc_patches.py \\
        --patch outputs/haru-hikage/ctc_patch_bridge.json \\
        --patch outputs/haru-hikage/ctc_patch_chorus.json \\
        --out   outputs/haru-hikage/ctc_patch_poc.json
"""
from __future__ import annotations

import json
from pathlib import Path

import click


@click.command()
@click.option("--patch", "patch_paths", multiple=True, required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Input patch (repeatable).")
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False))
def main(patch_paths: tuple[str, ...], out_path: str) -> None:
    if len(patch_paths) < 2:
        raise click.ClickException("Need at least two --patch inputs to merge.")

    seen: dict[int, str] = {}
    merged_lines: list[dict] = []
    merged_word_spans: list[dict] = []
    audio_windows: list[list[float]] = []
    line_index_ranges: list[list[int]] = []
    backend = None

    for path in patch_paths:
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        if backend is None:
            backend = p.get("backend")
        elif backend != p.get("backend"):
            raise click.ClickException(
                f"backend mismatch: {backend} vs {p.get('backend')} in {path}"
            )
        for ln in p.get("lines", []):
            idx = ln["line_idx"]
            if idx in seen:
                raise click.ClickException(
                    f"line_idx={idx} appears in both {seen[idx]} and {path}"
                )
            seen[idx] = path
            merged_lines.append(ln)
        merged_word_spans.extend(p.get("word_spans", []))
        if "audio_window" in p:
            audio_windows.append(p["audio_window"])
        if "line_index_range" in p:
            line_index_ranges.append(p["line_index_range"])

    merged_lines.sort(key=lambda r: r["line_idx"])

    payload = {
        "backend": backend,
        "merged_from": [str(p) for p in patch_paths],
        "audio_windows": audio_windows,
        "line_index_ranges": line_index_ranges,
        "lines": merged_lines,
        "word_spans": merged_word_spans,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(
        f"[merge_ctc_patches] merged {len(patch_paths)} patches -> "
        f"{len(merged_lines)} lines (idx {sorted(seen)}) -> {out_path}"
    )


if __name__ == "__main__":
    main()
