#!/usr/bin/env python3
"""Holistic singing benchmark — unified table for note transcription + lyric alignment.

Reads the registry (holistic_registry.json) and any referenced eval result files,
assembles a single two-axis table, and outputs markdown / JSON / TSV.

    ~/venvs/karaoke-jp/bin/python benchmarks/holistic_benchmark.py
    ~/venvs/karaoke-jp/bin/python benchmarks/holistic_benchmark.py --format tsv
    ~/venvs/karaoke-jp/bin/python benchmarks/holistic_benchmark.py --format json --out holistic_results.json
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "holistic_registry.json"

GRAN_LABELS = {"phone": "phone", "word": "word", "line": "line"}
THR_LABELS = {"phone": "<=50ms", "word": "PCO@.3", "line": "<=250ms"}


@dataclass
class Row:
    dataset: str
    system: str
    condition: str
    con: float | None = None
    conp: float | None = None
    conpoff: float | None = None
    granularity: str | None = None
    mae: float | None = None
    median: float | None = None
    hit_pct: float | None = None
    threshold: float | None = None
    note: str = ""
    contaminated: bool = False


def parse_evaluate_py_txt(path: Path) -> dict[str, float]:
    text = path.read_text()
    m_con = re.search(r"COn\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    m_conp = re.search(r"COnP\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    m_conpoff = re.search(r"COnPOff\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
    return {
        "COn": float(m_con.group(3)) if m_con else 0,
        "COnP": float(m_conp.group(3)) if m_conp else 0,
        "COnPOff": float(m_conpoff.group(3)) if m_conpoff else 0,
    }


def parse_phone_boundary_json(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    boundary = data.get("boundary", {})
    return {
        "MAE": boundary.get("mae", 0),
        "median": boundary.get("median", 0),
        "hit_pct": boundary.get("within_50ms", 0),
    }


def load_entry(entry: dict) -> Row:
    row = Row(
        dataset=entry["dataset"],
        system=entry["system"],
        condition=entry.get("condition", ""),
        note=entry.get("note", ""),
        contaminated=entry.get("condition") == "contaminated",
    )

    override = entry.get("override", {})

    if entry["axis"] == "transcription":
        if override:
            row.con = override.get("COn")
            row.conp = override.get("COnP")
            row.conpoff = override.get("COnPOff")
        elif "eval_file" in entry:
            path = ROOT / entry["eval_file"]
            if path.exists():
                nums = parse_evaluate_py_txt(path)
                row.con = nums["COn"]
                row.conp = nums["COnP"]
                row.conpoff = nums["COnPOff"]

    elif entry["axis"] == "alignment":
        row.granularity = entry.get("granularity")
        row.threshold = entry.get("threshold")
        if override:
            row.mae = override.get("MAE")
            row.median = override.get("median")
            row.hit_pct = override.get("hit_pct")
        elif "eval_file" in entry:
            path = ROOT / entry["eval_file"]
            if path.exists() and entry.get("eval_format") == "phone_boundary_json":
                nums = parse_phone_boundary_json(path)
                row.mae = nums["MAE"]
                row.median = nums["median"]
                row.hit_pct = nums["hit_pct"]

    return row


def fmt_f(v: float | None, decimals: int = 3) -> str:
    if v is None:
        return "-"
    return f".{int(v * 10**decimals):0{decimals}d}" if v < 1 else f"{v:.{decimals}f}"


def fmt_s(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}s"


def fmt_pct(v: float | None) -> str:
    if v is None:
        return "-"
    return f"{v:.1%}"


def render_markdown(rows: list[Row], datasets: dict) -> str:
    lines = [
        "# Holistic Singing Benchmark",
        "",
        "| Dataset | Lang | System | Cond. "
        "| COn | COnP | COnPOff "
        "| Gran. | MAE | median | hit% | thr |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    prev_ds = None
    for r in rows:
        ds_info = datasets.get(r.dataset, {})
        ds_label = r.dataset if r.dataset != prev_ds else ""
        lang = ds_info.get("lang", "") if r.dataset != prev_ds else ""

        warn = " !" if r.contaminated else ""
        gran = GRAN_LABELS.get(r.granularity, "-") if r.granularity else "-"
        thr = THR_LABELS.get(r.granularity, "") if r.granularity else ""

        lines.append(
            f"| {ds_label} | {lang} | {r.system}{warn} | {r.condition} "
            f"| {fmt_f(r.con)} | {fmt_f(r.conp)} | {fmt_f(r.conpoff)} "
            f"| {gran} | {fmt_s(r.mae)} | {fmt_s(r.median)} | {fmt_pct(r.hit_pct)} | {thr} |"
        )
        prev_ds = r.dataset

    return "\n".join(lines)


def render_tsv(rows: list[Row]) -> str:
    header = "dataset\tsystem\tcondition\tCOn\tCOnP\tCOnPOff\tgranularity\tMAE\tmedian\thit_pct\tthreshold"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r.dataset}\t{r.system}\t{r.condition}"
            f"\t{r.con or ''}\t{r.conp or ''}\t{r.conpoff or ''}"
            f"\t{r.granularity or ''}\t{r.mae or ''}\t{r.median or ''}"
            f"\t{r.hit_pct or ''}\t{r.threshold or ''}"
        )
    return "\n".join(lines)


def render_json(rows: list[Row]) -> str:
    data = []
    for r in rows:
        entry: dict = {
            "dataset": r.dataset,
            "system": r.system,
            "condition": r.condition,
        }
        if r.con is not None:
            entry["transcription"] = {
                "COn": round(r.con, 4),
                "COnP": round(r.conp, 4) if r.conp else None,
                "COnPOff": round(r.conpoff, 4) if r.conpoff else None,
            }
        if r.mae is not None:
            entry["alignment"] = {
                "granularity": r.granularity,
                "MAE": round(r.mae, 4),
                "median": round(r.median, 4) if r.median else None,
                "hit_pct": round(r.hit_pct, 4) if r.hit_pct else None,
                "threshold": r.threshold,
            }
        if r.note:
            entry["note"] = r.note
        if r.contaminated:
            entry["contaminated"] = True
        data.append(entry)
    return json.dumps(data, ensure_ascii=False, indent=2)


@click.command()
@click.option("--registry", type=click.Path(exists=True), default=str(REGISTRY))
@click.option("--format", "fmt", type=click.Choice(["markdown", "tsv", "json"]), default="markdown")
@click.option("--out", type=click.Path(), default=None, help="Write to file instead of stdout.")
def main(registry: str, fmt: str, out: str | None) -> None:
    reg = json.loads(Path(registry).read_text())
    datasets = reg.get("datasets", {})
    entries = reg.get("entries", [])

    rows = [load_entry(e) for e in entries]

    ds_order = list(datasets.keys())
    rows.sort(key=lambda r: (
        ds_order.index(r.dataset) if r.dataset in ds_order else 999,
        0 if r.con is not None else 1,
        r.system,
    ))

    if fmt == "markdown":
        output = render_markdown(rows, datasets)
    elif fmt == "tsv":
        output = render_tsv(rows)
    else:
        output = render_json(rows)

    if out:
        Path(out).write_text(output, encoding="utf-8")
        click.echo(f"Written to {out}")
    else:
        click.echo(output)


if __name__ == "__main__":
    main()
