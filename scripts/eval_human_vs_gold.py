"""Compare an MMS alignment prediction against OLD gold vs HUMAN markers.

Usage: python eval_human_vs_gold.py <markers.tsv> <gold.tsv> <aligned.json>

Builds human line gold (per gold line: nearest human line_start/line_end, independent, win 0.5s),
then reports start/end MAE etc. for the SAME lines & SAME prediction under both references.

Guards against pointing at the wrong song's files:
  * warns if markers' song_id != gold's song_id;
  * ABORTS if too few lines match (markers almost certainly belong to a different song).
"""
import sys, csv, json
import numpy as np

if len(sys.argv) != 4:
    sys.exit("usage: eval_human_vs_gold.py <markers.tsv> <gold.tsv> <aligned.json>")
markers_p, gold_p, aligned_p = sys.argv[1], sys.argv[2], sys.argv[3]

mk_rows = list(csv.DictReader(open(markers_p), delimiter="\t"))
ms = [(r["type"], float(r["time"])) for r in mk_rows]
H_start = np.array([t for k, t in ms if k == "line_start"])
H_end = np.array([t for k, t in ms if k == "line_end"])
gold = list(csv.DictReader(open(gold_p), delimiter="\t"))
pred = json.load(open(aligned_p))

# --- guard 1: song_id cross-check (markers song_id is often stale/default — warn, don't abort) ---
mk_song = (mk_rows[0].get("song_id", "") if mk_rows else "").strip()
gold_song = (gold[0].get("song_id", "") if gold else "").strip()
if mk_song and gold_song and mk_song != gold_song:
    print(f"⚠️  song_id MISMATCH: markers='{mk_song}' but gold='{gold_song}'. "
          f"(markers song_id is often left at the tool default — continuing, but check this is the right pairing.)")

def near(arr, x, win=0.5):
    if not len(arr):
        return None
    j = int(np.argmin(np.abs(arr - x)))
    return float(arr[j]) if abs(arr[j] - x) <= win else None

def iou(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0))
    uni = max(a1, b1) - min(a0, b0)
    return inter / uni if uni > 0 else 0.0

rows = []
for r in gold:
    li = int(r["line_idx"])
    if li >= len(pred):
        continue
    ps, pe = float(pred[li]["start"]), float(pred[li]["end"])
    gs, ge = float(r["gold_start"]), float(r["gold_end"])
    hs, he = near(H_start, gs), near(H_end, ge)
    if hs is None or he is None:          # only lines the human verified BOTH ends
        continue
    rows.append((ps, pe, gs, ge, hs, he, li))

# --- guard 2: too few matches => almost certainly the WRONG song's markers/gold pair ---
need = max(3, int(0.4 * len(gold)))
if len(rows) < need:
    sys.exit(f"⛔ only {len(rows)}/{len(gold)} gold lines got a human match (need >= {need}).\n"
             f"   The markers ({markers_p}) almost certainly belong to a DIFFERENT song than\n"
             f"   the gold ({gold_p}). Refusing to emit misleading numbers. Check the file pairing.")

def stats(name, errs, ious):
    e = np.array(errs); a = np.abs(e)
    print(f"  {name:5s} MAE={a.mean():.3f}s median={np.median(a):.3f}s P90={np.percentile(a,90):.3f}s "
          f"bias={e.mean():+.3f}s within250={(a<=.25).mean():.0%}  line_IoU_med={np.median(ious):.3f}")

print(f"lines compared (human-verified, both ends): {len(rows)}\n")
for which, si, gi in [("START", 0, 2), ("END", 1, 3)]:
    print(f"[{which}]")
    old_e = [r[si] - r[gi] for r in rows]; hum_e = [r[si] - r[gi + 2] for r in rows]
    old_iou = [iou(r[0], r[1], r[2], r[3]) for r in rows]
    hum_iou = [iou(r[0], r[1], r[4], r[5]) for r in rows]
    stats("OLD ", old_e, old_iou)
    stats("HUMAN", hum_e, hum_iou)
print("\n(pred fixed; only the reference gold differs. HUMAN<<OLD => old gold was the limiting factor.)")
