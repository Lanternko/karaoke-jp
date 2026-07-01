"""Evaluate karaoke alignment sidecars against Audacity gold labels.

The metrics intentionally mix research-style boundary errors with product
invariants:

* start/end MAE, median, P90, within 250/500 ms, signed bias;
* zero-duration sung chars, backward/overlap violations, short sung lines;
* a boundary stress subset for lines near butted/legato boundaries.

This keeps score-informed experiments honest: a global average can improve while
one line boundary regresses badly enough to look wrong in karaoke render.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import click


def _is_sung_char(ch: str) -> bool:
    if ch.isspace() or ch == "　":
        return False
    return unicodedata.category(ch)[0] not in {"P", "S"}


@dataclass(frozen=True)
class GoldRow:
    song_id: str
    line_idx: int
    text: str
    start: float
    end: float
    source: str = "human"   # human-priority gold tags lines human/machine; --human-only skips machine


@dataclass(frozen=True)
class EvalRow:
    song_id: str
    line_idx: int
    text: str
    gold_start: float
    pred_start: float
    signed_start_error: float
    gold_end: float
    pred_end: float
    signed_end_error: float
    pred_gap_prev: float | None
    gold_gap_prev: float | None
    stress_reason: str

    @property
    def abs_start_error(self) -> float:
        return abs(self.signed_start_error)

    @property
    def abs_end_error(self) -> float:
        return abs(self.signed_end_error)

    @property
    def is_stress(self) -> bool:
        return bool(self.stress_reason)


def read_gold(path: Path) -> list[GoldRow]:
    rows: list[GoldRow] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(
                GoldRow(
                    song_id=row["song_id"],
                    line_idx=int(row["line_idx"]),
                    text=row["text"],
                    start=float(row["gold_start"]),
                    end=float(row["gold_end"]),
                    source=(row.get("source") or "human"),
                )
            )
    return rows


def sung_chars(line: dict) -> list[dict]:
    out: list[dict] = []
    for tok in line.get("tokens", []):
        for ch in tok.get("chars") or []:
            if _is_sung_char(ch.get("char", "")):
                out.append(ch)
    return out


def line_span(line: dict) -> tuple[float, float]:
    sung = sung_chars(line)
    if sung:
        return float(sung[0]["start"]), float(sung[-1]["end"])
    return float(line["start"]), float(line["end"])


def percentile_nearest(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    return vals[math.ceil(len(vals) * q) - 1]


def metric_block(errors: list[float]) -> dict[str, float]:
    abs_errors = [abs(e) for e in errors]
    return {
        "mae": statistics.mean(abs_errors) if abs_errors else 0.0,
        "median": statistics.median(abs_errors) if abs_errors else 0.0,
        "p90": percentile_nearest(abs_errors, 0.9),
        "within_250ms": sum(e <= 0.25 for e in abs_errors) / len(abs_errors)
        if abs_errors
        else 0.0,
        "within_500ms": sum(e <= 0.50 for e in abs_errors) / len(abs_errors)
        if abs_errors
        else 0.0,
        "bias": statistics.mean(errors) if errors else 0.0,
    }


def span_iou(p_start: float, p_end: float, g_start: float, g_end: float) -> float:
    """Temporal IoU of two [start,end] spans. Catches 'whole line one beat late':
    nearest-onset MAE rates a line shifted by ~its own duration as a small error,
    but IoU collapses to ~0 -- this is the ownership-aware view the survey (D1)
    says is needed to see interior/line warp that MAE hides."""
    p_end = max(p_end, p_start)
    g_end = max(g_end, g_start)
    inter = max(0.0, min(p_end, g_end) - max(p_start, g_start))
    union = max(p_end, g_end) - min(p_start, g_start)
    return inter / union if union > 0 else 0.0


def iou_block(ious: list[float]) -> dict[str, float]:
    if not ious:
        return {"median": 0.0, "mean": 0.0, "below_0.5": 0.0, "below_0.3": 0.0}
    return {
        "median": statistics.median(ious),
        "mean": statistics.mean(ious),
        "below_0.5": sum(i < 0.5 for i in ious) / len(ious),
        "below_0.3": sum(i < 0.3 for i in ious) / len(ious),
    }


def invariants(lines: list[dict], *, short_line_threshold: float = 0.35) -> dict[str, int]:
    zero_duration_lines = 0
    zero_duration_chars = 0
    zero_duration_sung_chars = 0
    backward_sung_chars = 0
    overlap_sung_chars = 0
    short_sung_lines = 0
    prev_sung_end: float | None = None

    for line in lines:
        sung = []
        for tok in line.get("tokens", []):
            for ch in tok.get("chars") or []:
                start = float(ch["start"])
                end = float(ch["end"])
                if end <= start:
                    zero_duration_chars += 1
                    if _is_sung_char(ch.get("char", "")):
                        zero_duration_sung_chars += 1
                if _is_sung_char(ch.get("char", "")):
                    sung.append(ch)

        if not sung:
            continue

        start = float(sung[0]["start"])
        end = float(sung[-1]["end"])
        if end <= start:
            zero_duration_lines += 1
        if end - start < short_line_threshold:
            short_sung_lines += 1

        for prev, cur in zip(sung, sung[1:], strict=False):
            prev_start = float(prev["start"])
            prev_end = float(prev["end"])
            cur_start = float(cur["start"])
            if cur_start < prev_start:
                backward_sung_chars += 1
            if cur_start < prev_end:
                overlap_sung_chars += 1

        if prev_sung_end is not None and start < prev_sung_end:
            backward_sung_chars += 1
        prev_sung_end = end

    return {
        "zero_duration_lines": zero_duration_lines,
        "zero_duration_chars": zero_duration_chars,
        "zero_duration_sung_chars": zero_duration_sung_chars,
        "backward_sung_chars": backward_sung_chars,
        "overlap_sung_chars": overlap_sung_chars,
        "short_sung_lines": short_sung_lines,
    }


def evaluate(
    lines: list[dict],
    gold_rows: list[GoldRow],
    *,
    stress_error_threshold: float,
    stress_gap_threshold: float,
) -> tuple[dict, list[EvalRow]]:
    rows: list[EvalRow] = []
    prev_gold: GoldRow | None = None
    prev_pred_end: float | None = None

    for gold in gold_rows:
        if gold.line_idx >= len(lines):
            raise ValueError(f"gold line_idx={gold.line_idx} outside aligned length {len(lines)}")
        pred_start, pred_end = line_span(lines[gold.line_idx])
        pred_gap = None if prev_pred_end is None else pred_start - prev_pred_end
        gold_gap = None if prev_gold is None else gold.start - prev_gold.end

        start_error = pred_start - gold.start
        end_error = pred_end - gold.end
        reasons: list[str] = []
        if abs(start_error) >= stress_error_threshold:
            reasons.append("start_error")
        if abs(end_error) >= stress_error_threshold:
            reasons.append("end_error")
        if pred_gap is not None and pred_gap <= stress_gap_threshold:
            reasons.append("pred_butted")
        if gold_gap is not None and gold_gap <= stress_gap_threshold:
            reasons.append("gold_butted")

        rows.append(
            EvalRow(
                song_id=gold.song_id,
                line_idx=gold.line_idx,
                text=gold.text,
                gold_start=gold.start,
                pred_start=pred_start,
                signed_start_error=start_error,
                gold_end=gold.end,
                pred_end=pred_end,
                signed_end_error=end_error,
                pred_gap_prev=pred_gap,
                gold_gap_prev=gold_gap,
                stress_reason=",".join(reasons),
            )
        )
        prev_gold = gold
        prev_pred_end = pred_end

    line_ious = [
        span_iou(r.pred_start, r.pred_end, r.gold_start, r.gold_end) for r in rows
    ]
    summary = {
        "n": len(rows),
        "start": metric_block([r.signed_start_error for r in rows]),
        "end": metric_block([r.signed_end_error for r in rows]),
        "line_iou": iou_block(line_ious),
        "stress_n": sum(r.is_stress for r in rows),
        "invariants": invariants(lines),
    }
    return summary, rows


def write_rows(path: Path, rows: list[EvalRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "song_id",
                "line_idx",
                "text",
                "gold_start",
                "pred_start",
                "signed_start_error",
                "abs_start_error",
                "gold_end",
                "pred_end",
                "signed_end_error",
                "abs_end_error",
                "pred_gap_prev",
                "gold_gap_prev",
                "stress_reason",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.song_id,
                    row.line_idx,
                    row.text,
                    f"{row.gold_start:.3f}",
                    f"{row.pred_start:.3f}",
                    f"{row.signed_start_error:+.3f}",
                    f"{row.abs_start_error:.3f}",
                    f"{row.gold_end:.3f}",
                    f"{row.pred_end:.3f}",
                    f"{row.signed_end_error:+.3f}",
                    f"{row.abs_end_error:.3f}",
                    "" if row.pred_gap_prev is None else f"{row.pred_gap_prev:.3f}",
                    "" if row.gold_gap_prev is None else f"{row.gold_gap_prev:.3f}",
                    row.stress_reason,
                ]
            )


def print_summary(summary: dict) -> None:
    print(f"n={summary['n']}")
    for name in ["start", "end"]:
        block = summary[name]
        print(
            f"{name}_MAE={block['mae']:.3f}s "
            f"median={block['median']:.3f}s "
            f"P90={block['p90']:.3f}s "
            f"within_250ms={block['within_250ms']:.1%} "
            f"within_500ms={block['within_500ms']:.1%} "
            f"bias={block['bias']:+.3f}s"
        )
    iou = summary["line_iou"]
    print(
        f"line_IoU median={iou['median']:.3f} mean={iou['mean']:.3f} "
        f"below_0.5={iou['below_0.5']:.1%} below_0.3={iou['below_0.3']:.1%}"
    )
    inv = summary["invariants"]
    print(
        "invalids="
        f"zero_lines:{inv['zero_duration_lines']} "
        f"zero_chars:{inv['zero_duration_chars']} "
        f"zero_sung:{inv['zero_duration_sung_chars']} "
        f"backward:{inv['backward_sung_chars']} "
        f"overlap:{inv['overlap_sung_chars']} "
        f"short_lines:{inv['short_sung_lines']}"
    )
    print(f"stress_n={summary['stress_n']}")


@click.command()
@click.option("--gold", "gold_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--diff-out", type=click.Path(dir_okay=False), default=None)
@click.option("--stress-out", type=click.Path(dir_okay=False), default=None)
@click.option("--json-out", type=click.Path(dir_okay=False), default=None)
@click.option("--stress-error-threshold", default=0.5, show_default=True)
@click.option("--stress-gap-threshold", default=0.25, show_default=True)
@click.option("--human-only", is_flag=True,
              help="evaluate only source=human lines (skip machine-filled lines in a human-priority gold)")
def main(
    gold_path: str,
    aligned_path: str,
    diff_out: str | None,
    stress_out: str | None,
    json_out: str | None,
    stress_error_threshold: float,
    stress_gap_threshold: float,
    human_only: bool,
) -> None:
    lines = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    gold_rows = read_gold(Path(gold_path))
    if human_only:
        kept = [g for g in gold_rows if g.source != "machine"]
        print(f"[human-only] {len(kept)}/{len(gold_rows)} lines evaluated "
              f"(skipped {len(gold_rows) - len(kept)} machine-filled)")
        gold_rows = kept
    summary, rows = evaluate(
        lines,
        gold_rows,
        stress_error_threshold=stress_error_threshold,
        stress_gap_threshold=stress_gap_threshold,
    )
    print_summary(summary)

    if diff_out:
        write_rows(Path(diff_out), rows)
    if stress_out:
        write_rows(Path(stress_out), [r for r in rows if r.is_stress])
    if json_out:
        dest = Path(json_out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
