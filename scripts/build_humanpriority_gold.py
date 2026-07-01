"""Build a FULL-SONG human-priority alignment gold.

Line structure + text come from the machine alignment (one line per aligned line);
each line's timing is the HUMAN marker where the user marked it, else the machine value.
A `source` column tags human/machine so eval can exclude (or down-weight) machine lines.

Usage: python build_humanpriority_gold.py <markers.tsv> <aligned.json> <song_id> <out.tsv> [win]
"""
import sys, csv, json
import numpy as np

markers_p, aligned_p, song_id, out_p = sys.argv[1:5]
win = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7

ms = [(r["type"], float(r["time"])) for r in csv.DictReader(open(markers_p), delimiter="\t")]
H_start = np.array(sorted(t for k, t in ms if k == "line_start"))
H_end = np.array(sorted(t for k, t in ms if k == "line_end"))
pred = json.load(open(aligned_p))

def near(arr, x):
    if not len(arr):
        return None
    j = int(np.argmin(np.abs(arr - x)))
    return float(arr[j]) if abs(arr[j] - x) <= win else None

rows = []
n_human = 0
for i, L in enumerate(pred):
    ts, te, text = float(L["start"]), float(L["end"]), L.get("text", "")
    hs, he = near(H_start, ts), near(H_end, te)
    if hs is not None and he is not None and he > hs:
        gs, ge, src = hs, he, "human"
        n_human += 1
    else:
        gs, ge, src = ts, te, "machine"
    rows.append({"song_id": song_id, "line_idx": i, "text": text,
                 "gold_start": f"{gs:.3f}", "gold_end": f"{ge:.3f}", "source": src})

with open(out_p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["song_id", "line_idx", "text", "gold_start", "gold_end", "source"], delimiter="\t")
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"{song_id}: {len(rows)} lines, {n_human} human / {len(rows)-n_human} machine -> {out_p}")
