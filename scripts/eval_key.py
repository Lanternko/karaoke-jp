"""Key-detection A/B harness: our peak-PCP K-S vs Essentia (vs madmom if present).

Runs each {method} on each {input stem} and scores against a small hand gold
set with the MIREX weighted score (correct 1.0 / fifth 0.5 / relative 0.3 /
parallel 0.2). Built to answer: which INPUT (mix / accompaniment / vocals) and
which TOOL should the karaoke pipeline adopt? See docs/key-detection-survey.md.

Run in the keytest venv (has essentia + numpy + click for the peak-PCP import):
    ~/venvs/keytest/bin/python scripts/eval_key.py [song ...]

Inputs are normalized to mono 44.1k PCM16 first (our peak-PCP only reads PCM16;
the raw stems are float32 -> feeding them directly returns garbage). madmom is
optional: it needs numpy<2 / py<=3.11 and conflicts with essentia's numpy2 ABI,
so it stays a gated backend (skipped with a note when unimportable).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

# --- gold (extend as songs get hand/sheet-validated keys) ---
GOLD = {
    "chidori": ("Eb", "major"),  # professor + Synthesia-sheet validated (MEMORY.md)
}

INPUTS = {
    "source(mix)": "songs/{s}/source.wav",
    "mixed(ours)": "outputs/{s}/mixed.wav",
    "instrumental": "outputs/{s}/instrumental.wav",
    "vocals": "outputs/{s}/vocals.wav",
}

_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_key(name: str, scale: str | None = None) -> tuple[int, str] | None:
    """('E♭','major')/('Eb',None as 'Gm')/('F♯') -> (tonic_pc, 'major'|'minor')."""
    name = name.strip().replace("♭", "b").replace("♯", "#")
    if scale is None:  # peak-PCP style: trailing 'm' = minor, else major
        if name.endswith("m"):
            scale, name = "minor", name[:-1]
        else:
            scale = "major"
    if not name or name[0] not in _PC:
        return None
    pc = _PC[name[0]]
    for ch in name[1:]:
        if ch == "#":
            pc += 1
        elif ch == "b":
            pc -= 1
    return pc % 12, ("minor" if scale.startswith("min") else "major")


def mirex_score(pred, gold) -> float:
    if pred is None:
        return 0.0
    pp, pm = pred
    gp, gm = gold
    if pp == gp and pm == gm:
        return 1.0
    if pm == gm and (pp == (gp + 7) % 12 or gp == (pp + 7) % 12):
        return 0.5  # perfect fifth (dominant/subdominant), modes agree
    if pm != gm:
        if gm == "major" and pp == (gp + 9) % 12:
            return 0.3  # relative minor of a major gold
        if gm == "minor" and pp == (gp + 3) % 12:
            return 0.3  # relative major of a minor gold
        if pp == gp:
            return 0.2  # parallel (same tonic, other mode)
    return 0.0


# --- backends: name -> fn(wav_path) -> "Display Key" or None ---
def backend_peakpcp(path: str):
    from render_mp4 import _detect_key
    return _detect_key([], path)["name"]


def backend_essentia(path: str):
    import essentia.standard as es
    key, scale, _ = es.KeyExtractor()(es.MonoLoader(filename=path, sampleRate=44100)())
    return f"{key} {scale}"


def backend_madmom(path: str):
    from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
    return key_prediction_to_label(CNNKeyRecognitionProcessor()(path))


def available_backends() -> dict:
    out = {"peakpcp": backend_peakpcp}
    try:
        import essentia.standard  # noqa: F401
        out["essentia"] = backend_essentia
    except Exception:
        print("[eval] essentia unavailable — skipping", file=sys.stderr)
    try:
        import madmom  # noqa: F401
        out["madmom"] = backend_madmom
    except Exception:
        print("[eval] madmom unavailable (needs numpy<2 / py<=3.11) — skipping",
              file=sys.stderr)
    return out


def norm_wav(src: str, dst: Path) -> bool:
    """ffmpeg -> mono 44.1k PCM16 so every backend reads identical audio."""
    if not Path(src).is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "44100",
         "-c:a", "pcm_s16le", str(dst)],
        check=True, capture_output=True)
    return True


def parse_pred(method: str, raw: str):
    if raw is None:
        return None
    if method == "essentia":
        k, _, s = raw.partition(" ")
        return parse_key(k, s)
    if method == "madmom":  # "C major" / "A minor"
        k, _, s = raw.partition(" ")
        return parse_key(k, s)
    return parse_key(raw)  # peak-PCP display name


def main(argv):
    songs = argv or list(GOLD)
    backends = available_backends()
    methods = list(backends)
    print(f"methods: {methods}\nMIREX weighted: correct 1.0 / fifth 0.5 / "
          f"relative 0.3 / parallel 0.2\n")
    tmp = REPO / "tmp" / "keytest"
    agg = {m: [] for m in methods}
    for song in songs:
        gold = parse_key(*GOLD[song])
        print(f"### {song}  (gold {GOLD[song][0]} {GOLD[song][1]})")
        head = "  " + f"{'input':<14}" + "".join(f"{m:<22}" for m in methods)
        print(head)
        for label, tmpl in INPUTS.items():
            src = str(REPO / tmpl.format(s=song))
            dst = tmp / f"{song}_{label.split('(')[0]}.wav"
            if not norm_wav(src, dst):
                print(f"  {label:<14}" + "(missing)")
                continue
            cells = []
            for m in methods:
                try:
                    raw = backends[m](str(dst))
                    pred = parse_pred(m, raw)
                    sc = mirex_score(pred, gold)
                    agg[m].append(sc)
                    cells.append(f"{raw:<13} {sc:.2f}")
                except Exception as e:
                    cells.append(f"ERR {type(e).__name__}")
            print(f"  {label:<14}" + "".join(f"{c:<22}" for c in cells))
        print()
    if any(agg.values()):
        print("=== mean MIREX weighted (all song×input cells) ===")
        for m in methods:
            v = agg[m]
            print(f"  {m:<12} {sum(v)/len(v):.3f}  (n={len(v)})" if v else f"  {m}: n=0")


if __name__ == "__main__":
    main(sys.argv[1:])
