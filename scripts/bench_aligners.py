"""Cross-model lyric-alignment benchmark vs the human-priority gold.

Scores every available aligner prediction with ONE ruler: per-line start/end
boundary error against the human gold lines (source != "machine"), per song and
pooled. Self-contained on purpose -- it inlines the same line_span / sung-char /
span-IoU logic as scripts/eval_alignment.py so it can be committed and run on a
fresh checkout without depending on in-flight changes to that file. (Verified to
reproduce eval_alignment.evaluate() numbers to the millisecond.)

Run from repo root:
    ~/venvs/karaoke-jp/bin/python scripts/bench_aligners.py

Predictions live under outputs/<song>/ (gitignored); gold under
data/alignment_gold/ (lyrics are private -- only aggregate metrics are public).
"""
from __future__ import annotations
import csv, json, statistics, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SONGS = ["chidori", "haru-hikage", "tuki-zero"]
GOLD = {s: ROOT / f"data/alignment_gold/{s}.gold.tsv" for s in SONGS}

# model -> {song: prediction json}. haru MMS uses the fresh file; the
# outputs/haru-hikage/aligned_midi.json is the stale 10.9s disaster (see handoff).
MODELS: dict[str, dict[str, str]] = {
    "MMS-JA (canonical)": {
        "chidori": "outputs/chidori/aligned_midi.json",
        "haru-hikage": "tmp/haru_mms_fresh.json",
        "tuki-zero": "outputs/tuki-zero/aligned_midi.json",
    },
    "SOFA zero-shot": {s: f"outputs/{s}/aligned.sofa.json" for s in SONGS},
    "SOFA +island anchor": {s: f"outputs/{s}/aligned.sofa_islands.json" for s in SONGS},
    "classic (Whisper)": {"tuki-zero": "outputs/tuki-zero/aligned_whisper_backup.json"},
}


def is_sung(ch: str) -> bool:
    if ch.isspace() or ch == "　":
        return False
    return unicodedata.category(ch)[0] not in {"P", "S"}


def line_span(line: dict) -> tuple[float, float]:
    sung = [c for tok in line.get("tokens", []) for c in (tok.get("chars") or [])
            if is_sung(c.get("char", ""))]
    if sung:
        return float(sung[0]["start"]), float(sung[-1]["end"])
    return float(line["start"]), float(line["end"])


def span_iou(p0: float, p1: float, g0: float, g1: float) -> float:
    p1, g1 = max(p1, p0), max(g1, g0)
    inter = max(0.0, min(p1, g1) - max(p0, g0))
    union = max(p1, g1) - min(p0, g0)
    return inter / union if union > 0 else 0.0


def human_gold(song: str) -> list[dict]:
    with GOLD[song].open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [r for r in rows if (r.get("source") or "human") != "machine"]


def score(pred_path: Path, gold: list[dict]) -> dict | None:
    lines = json.loads(pred_path.read_text(encoding="utf-8"))
    s_err, e_err, ious = [], [], []
    for r in gold:
        li = int(r["line_idx"])
        if li >= len(lines):
            return None  # prediction shorter than gold -> wrong pairing
        ps, pe = line_span(lines[li])
        gs, ge = float(r["gold_start"]), float(r["gold_end"])
        s_err.append(abs(ps - gs)); e_err.append(abs(pe - ge))
        ious.append(span_iou(ps, pe, gs, ge))
    n = len(s_err)
    agg = lambda v: dict(mae=statistics.mean(v), med=statistics.median(v),
                         w250=sum(x <= .25 for x in v) / n, w500=sum(x <= .50 for x in v) / n)
    return dict(n=n, start=agg(s_err), end=agg(e_err), iou_med=statistics.median(ious),
                _s=s_err, _e=e_err, _iou=ious)


def main() -> None:
    pooled = {m: {"s": [], "e": [], "iou": []} for m in MODELS}
    print("=== Lyric alignment vs HUMAN gold (human-only), per song ===")
    print(f"{'model':22s} {'song':12s} {'n':>3s}  {'start_MAE':>9s} {'st_med':>7s} {'st<=250':>7s} "
          f"{'st<=500':>7s}  {'end_MAE':>8s} {'en<=250':>7s}  {'IoU_med':>7s}")
    for model, mp in MODELS.items():
        for song in SONGS:
            p = mp.get(song)
            if not p or not (ROOT / p).exists():
                continue
            r = score(ROOT / p, human_gold(song))
            if r is None:
                print(f"{model:22s} {song:12s}  SKIP (pred shorter than gold)"); continue
            st, en = r["start"], r["end"]
            print(f"{model:22s} {song:12s} {r['n']:>3d}  {st['mae']:>8.3f}s {st['med']:>6.3f}s "
                  f"{st['w250']:>7.0%} {st['w500']:>7.0%}  {en['mae']:>7.3f}s {en['w250']:>7.0%}  "
                  f"{r['iou_med']:>7.3f}")
            pooled[model]["s"] += r["_s"]; pooled[model]["e"] += r["_e"]; pooled[model]["iou"] += r["_iou"]

    print("\n=== Pooled across all covered human lines (micro-average) ===")
    print(f"{'model':22s} {'lines':>5s}  {'start_MAE':>9s} {'st_med':>7s} {'st<=250':>7s}  "
          f"{'end_MAE':>8s} {'en_med':>7s} {'en<=250':>7s}  {'IoU_med':>7s}  coverage")
    for model, d in pooled.items():
        n = len(d["s"])
        if not n:
            continue
        sm, smed = statistics.mean(d["s"]), statistics.median(d["s"])
        em, emed = statistics.mean(d["e"]), statistics.median(d["e"])
        s250 = sum(x <= .25 for x in d["s"]) / n
        e250 = sum(x <= .25 for x in d["e"]) / n
        cov = ",".join(s for s in SONGS if s in MODELS[model])
        print(f"{model:22s} {n:>5d}  {sm:>8.3f}s {smed:>6.3f}s {s250:>7.0%}  "
              f"{em:>7.3f}s {emed:>6.3f}s {e250:>7.0%}  {statistics.median(d['iou']):>7.3f}  [{cov}]")

    print("\nstart MAE = line-start boundary (the karaoke wipe-in cue, most perceptually critical).")


if __name__ == "__main__":
    main()
