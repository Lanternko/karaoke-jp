"""Post-hoc applier for the F0-voicing re-entry guard.

Thin CLI around ``forced_align_mms.f0_reentry_guard`` so the guard composes
with existing sidecars like the other repair stages (line_end_repair etc.):

    f0_reentry_guard.py --aligned aligned_midi.json \
        --f0 rmvpe_f0.npz -o aligned_midi.guarded.json

Rationale and the haru line-29 evidence are documented on the function.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forced_align_mms import f0_reentry_guard  # noqa: E402


@click.command()
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--f0", "f0_npz_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "-o", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--min-gap", default=4.0, show_default=True)
@click.option("--search", default=1.2, show_default=True)
@click.option("--min-fix", default=0.2, show_default=True)
def main(aligned_path: str, f0_npz_path: str, out_path: str,
         min_gap: float, search: float, min_fix: float) -> None:
    lines = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    moved = f0_reentry_guard(lines, f0_npz_path, min_gap=min_gap,
                             search=search, min_fix=min_fix)
    Path(out_path).write_text(json.dumps(lines, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    click.echo(f"[f0-guard] moved {moved} line start(s) -> {out_path}")


if __name__ == "__main__":
    main()
