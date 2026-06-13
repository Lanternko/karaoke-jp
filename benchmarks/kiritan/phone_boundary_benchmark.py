"""Kiritan phoneme-boundary benchmark for forced aligners.

This is intentionally separate from the existing Kiritan note benchmark.
GAME/CE+CTC/ROSVOT emit notes and are scored with COn/COnP/COnPOff; MMS/SOFA
emit lyric/phoneme timing and should be scored against Kiritan's mono_label
phoneme boundaries.

Common protocol:

* source labels: Kiritan ``mono_label/*.lab``;
* input transcript: sung phones, excluding ``pau`` and ``br``;
* output/eval format: HTK-style ``start end phone`` with seconds;
* headline metrics: start/end/boundary MAE and BER@10/20/50ms.

The evaluator compares phones by index after dropping ignored labels. This keeps
MMS and SOFA on the same ground even though SOFA may insert SP/AP labels.
"""
from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import click

DEFAULT_DATASET = Path("/home/kojiek/side_projects/kiritan/kiritan_singing")
DEFAULT_SKIP_INPUT = {"pau", "br"}
DEFAULT_IGNORE_EVAL = {"", "pau", "br", "SP", "AP", "<SP>", "<AP>"}


@dataclass(frozen=True)
class PhoneInterval:
    start: float
    end: float
    phone: str


def read_kiritan_lab(path: Path, *, skip: set[str] | None = None) -> list[PhoneInterval]:
    out: list[PhoneInterval] = []
    skip = skip or set()
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        s, e, ph = parts
        if ph in skip:
            continue
        out.append(PhoneInterval(float(s), float(e), ph))
    return out


def read_htk_seconds(path: Path) -> list[PhoneInterval]:
    out: list[PhoneInterval] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        s, e, ph = parts
        # SOFA HTK exports use 100ns ticks; this benchmark's own HTK files use
        # decimal seconds. Accept both to make the evaluator interoperate.
        if "." in s or "." in e:
            start, end = float(s), float(e)
        else:
            start, end = int(s) / 1e7, int(e) / 1e7
        out.append(PhoneInterval(start, end, ph))
    return out


def write_htk_seconds(path: Path, intervals: list[PhoneInterval]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in intervals:
            f.write(f"{it.start:.6f} {it.end:.6f} {it.phone}\n")


def split_sung_phrases(intervals: list[PhoneInterval], *, skip: set[str]) -> list[list[PhoneInterval]]:
    phrases: list[list[PhoneInterval]] = []
    current: list[PhoneInterval] = []
    for it in intervals:
        if it.phone in skip:
            if current:
                phrases.append(current)
                current = []
            continue
        current.append(it)
    if current:
        phrases.append(current)
    return phrases


def percentile_nearest(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    return vals[math.ceil(len(vals) * q) - 1]


def summarize(errors: list[float]) -> dict[str, float]:
    abs_err = [abs(e) for e in errors]
    return {
        "mae": statistics.mean(abs_err) if abs_err else 0.0,
        "median": statistics.median(abs_err) if abs_err else 0.0,
        "p90": percentile_nearest(abs_err, 0.9),
        "bias": statistics.mean(errors) if errors else 0.0,
        "within_10ms": sum(e <= 0.010 for e in abs_err) / len(abs_err) if abs_err else 0.0,
        "within_20ms": sum(e <= 0.020 for e in abs_err) / len(abs_err) if abs_err else 0.0,
        "within_50ms": sum(e <= 0.050 for e in abs_err) / len(abs_err) if abs_err else 0.0,
        "ber_10ms": sum(e > 0.010 for e in abs_err) / len(abs_err) if abs_err else 0.0,
        "ber_20ms": sum(e > 0.020 for e in abs_err) / len(abs_err) if abs_err else 0.0,
        "ber_50ms": sum(e > 0.050 for e in abs_err) / len(abs_err) if abs_err else 0.0,
    }


def _filter(intervals: list[PhoneInterval], ignore: set[str]) -> list[PhoneInterval]:
    return [it for it in intervals if it.phone not in ignore]


def evaluate_dirs(pred_dir: Path, target_dir: Path, *, ignore: set[str]) -> dict:
    per_song = []
    all_start: list[float] = []
    all_end: list[float] = []
    mismatches = []
    for target_path in sorted(target_dir.glob("*.lab")):
        pred_path = pred_dir / target_path.name
        if not pred_path.exists():
            mismatches.append({"song": target_path.stem, "error": "missing_pred"})
            continue
        gold = _filter(read_htk_seconds(target_path), ignore)
        pred = _filter(read_htk_seconds(pred_path), ignore)
        if [x.phone for x in gold] != [x.phone for x in pred]:
            mismatches.append(
                {
                    "song": target_path.stem,
                    "error": "phone_sequence_mismatch",
                    "gold_n": len(gold),
                    "pred_n": len(pred),
                    "gold_head": [x.phone for x in gold[:20]],
                    "pred_head": [x.phone for x in pred[:20]],
                }
            )
            continue
        start_err = [p.start - g.start for p, g in zip(pred, gold)]
        end_err = [p.end - g.end for p, g in zip(pred, gold)]
        all_start.extend(start_err)
        all_end.extend(end_err)
        per_song.append(
            {
                "song": target_path.stem,
                "n": len(gold),
                "start": summarize(start_err),
                "end": summarize(end_err),
            }
        )
    boundary_err = all_start + all_end
    return {
        "songs": len(per_song),
        "phones": len(all_start),
        "start": summarize(all_start),
        "end": summarize(all_end),
        "boundary": summarize(boundary_err),
        "mismatches": mismatches,
        "per_song": per_song,
    }


def load_model(method: str, device: str):
    import torch  # noqa: F401

    if method == "mms_ja":
        from transformers import AutoProcessor, Wav2Vec2ForCTC

        model_id = "NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn"
        processor = AutoProcessor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
        vocab = processor.tokenizer.get_vocab()
        return model, vocab, processor.tokenizer.pad_token_id, vocab.get("|")

    import torchaudio

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).to(device).eval()
    vocab = bundle.get_dict(star=None)
    return model, vocab, 0, None


def _chunk_boundaries(wave, sr: int, chunk_s: float):
    n = wave.shape[-1]
    if n <= int(chunk_s * sr):
        return [(0, n)]
    bounds = [0]
    target = int(chunk_s * sr)
    win = int(0.02 * sr)
    while bounds[-1] + target < n:
        nominal = bounds[-1] + target
        lo = max(bounds[-1] + sr, nominal - 5 * sr)
        hi = min(n - sr, nominal + 5 * sr)
        seg = wave[lo:hi]
        frames = seg.unfold(0, win, win)
        rms = (frames ** 2).mean(dim=1)
        bounds.append(lo + int(rms.argmin().item()) * win)
    bounds.append(n)
    return list(zip(bounds[:-1], bounds[1:]))


def normalize_phone_for_mms(phone: str) -> str:
    # Kiritan uses uppercase N for moraic nasal; MMS vocab is lowercase letters.
    return phone.lower()


def align_mms_song(model, vocab, blank_id: int, sep_id: int | None, wav_path: Path,
                   gold: list[PhoneInterval], *, device: str, chunk_s: float) -> list[PhoneInterval]:
    import soundfile as sf
    import torch
    import torchaudio

    data, sr = sf.read(wav_path, dtype="float32")
    wave = torch.from_numpy(data.T if getattr(data, "ndim", 1) > 1 else data[None, :]).mean(dim=0)
    if sr != 16000:
        wave = torchaudio.functional.resample(wave, sr, 16000)
        sr = 16000

    targets: list[int] = []
    owner: list[int] = []
    for pi, it in enumerate(gold):
        word = normalize_phone_for_mms(it.phone)
        letters = [ch for ch in word if ch in vocab]
        if not letters:
            continue
        if targets and sep_id is not None:
            targets.append(sep_id)
            owner.append(-1)
        for ch in letters:
            targets.append(vocab[ch])
            owner.append(pi)
    if not targets:
        return []

    emissions = []
    with torch.inference_mode():
        for a, b in _chunk_boundaries(wave, sr, chunk_s):
            logits = model(wave[a:b].unsqueeze(0).to(device))
            logits = logits.logits if hasattr(logits, "logits") else logits[0]
            emissions.append(torch.log_softmax(logits, dim=-1).cpu())
    emission = torch.cat(emissions, dim=1)
    ratio = wave.shape[-1] / emission.shape[1] / sr

    tgt = torch.tensor([targets], dtype=torch.int32)
    path, scores = torchaudio.functional.forced_align(emission, tgt, blank=blank_id)
    spans = torchaudio.functional.merge_tokens(path[0], scores[0], blank=blank_id)

    by_phone: dict[int, list[tuple[float, float]]] = {}
    for ti, span in enumerate(spans):
        pi = owner[ti] if ti < len(owner) else -1
        if pi >= 0:
            by_phone.setdefault(pi, []).append((span.start * ratio, span.end * ratio))

    out: list[PhoneInterval | None] = []
    for pi, it in enumerate(gold):
        ts = by_phone.get(pi)
        if ts:
            out.append(PhoneInterval(min(t[0] for t in ts), max(t[1] for t in ts), it.phone))
        else:
            out.append(None)
    for i, it in enumerate(out):
        if it is None:
            prev_end = next((out[j].end for j in range(i - 1, -1, -1) if out[j]), 0.0)
            next_start = next((out[j].start for j in range(i + 1, len(out)) if out[j]), prev_end)
            out[i] = PhoneInterval(prev_end, max(prev_end, next_start), gold[i].phone)
    return [x for x in out if x is not None]


@click.group()
def cli() -> None:
    pass


@cli.command("prepare")
@click.option("--dataset", type=click.Path(exists=True, file_okay=False), default=str(DEFAULT_DATASET))
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="benchmarks/kiritan/phone_boundary")
@click.option("--limit", type=int, default=0, help="0 = all songs.")
@click.option(
    "--sofa-tokenization",
    type=click.Choice(["utterance", "phrase"]),
    default="phrase",
    show_default=True,
    help="How to package phone sequences for SOFA DictionaryG2P.",
)
def prepare_cmd(dataset: str, out_dir: str, limit: int, sofa_tokenization: str) -> None:
    dataset_path = Path(dataset)
    out = Path(out_dir)
    target = out / "target_htk"
    sofa_segments = out / "sofa_segments"
    sofa_dict = out / "sofa_phone_identity.tsv"
    target.mkdir(parents=True, exist_ok=True)
    sofa_segments.mkdir(parents=True, exist_ok=True)

    rows = []
    labs = sorted((dataset_path / "mono_label").glob("*.lab"))
    if limit:
        labs = labs[:limit]
    for lab in labs:
        song = lab.stem
        raw = read_kiritan_lab(lab)
        phones = [it for it in raw if it.phone not in DEFAULT_SKIP_INPUT]
        write_htk_seconds(target / f"{song}.lab", phones)
        wav_src = dataset_path / "wav" / f"{song}.wav"
        wav_dst = sofa_segments / f"{song}.wav"
        if not wav_dst.exists():
            os.symlink(wav_src, wav_dst)
        if sofa_tokenization == "utterance":
            (sofa_segments / f"{song}.lab").write_text(f"utt{song}\n", encoding="utf-8")
            rows.append(f"utt{song}\t{' '.join(it.phone for it in phones)}")
        else:
            tokens: list[str] = []
            for pi, phrase in enumerate(split_sung_phrases(raw, skip=DEFAULT_SKIP_INPUT), start=1):
                token = f"utt{song}_{pi:03d}"
                tokens.append(token)
                rows.append(f"{token}\t{' '.join(it.phone for it in phrase)}")
            (sofa_segments / f"{song}.lab").write_text(" ".join(tokens) + "\n", encoding="utf-8")
    sofa_dict.write_text("\n".join(rows) + "\n", encoding="utf-8")
    click.echo(f"[kiritan-phone] prepared {len(labs)} songs -> {out}")


@cli.command("run-mms")
@click.option("--dataset", type=click.Path(exists=True, file_okay=False), default=str(DEFAULT_DATASET))
@click.option("--method", type=click.Choice(["mms_ja", "mms_fa"]), required=True)
@click.option("--out", "out_dir", type=click.Path(file_okay=False), required=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--chunk-seconds", default=110.0, show_default=True)
@click.option("--limit", type=int, default=0, help="0 = all songs.")
def run_mms_cmd(dataset: str, method: str, out_dir: str, device: str,
                chunk_seconds: float, limit: int) -> None:
    dataset_path = Path(dataset)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labs = sorted((dataset_path / "mono_label").glob("*.lab"))
    if limit:
        labs = labs[:limit]
    model, vocab, blank_id, sep_id = load_model(method, device)
    for i, lab in enumerate(labs, start=1):
        song = lab.stem
        dest = out / f"{song}.lab"
        if dest.exists():
            continue
        gold = read_kiritan_lab(lab, skip=DEFAULT_SKIP_INPUT)
        pred = align_mms_song(
            model,
            vocab,
            blank_id,
            sep_id,
            dataset_path / "wav" / f"{song}.wav",
            gold,
            device=device,
            chunk_s=chunk_seconds,
        )
        write_htk_seconds(dest, pred)
        click.echo(f"[{method}] {i}/{len(labs)} {song}: {len(pred)} phones -> {dest}")


@cli.command("eval")
@click.option("--pred", "pred_dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--target", "target_dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--ignore", default=",".join(sorted(DEFAULT_IGNORE_EVAL)), show_default=True)
@click.option("--json-out", type=click.Path(dir_okay=False), default=None)
def eval_cmd(pred_dir: str, target_dir: str, ignore: str, json_out: str | None) -> None:
    result = evaluate_dirs(
        Path(pred_dir),
        Path(target_dir),
        ignore={x for x in ignore.split(",") if x},
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if json_out:
        dest = Path(json_out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
    print(
        f"songs={result['songs']} phones={result['phones']} mismatches={len(result['mismatches'])}\n"
        f"start_MAE={result['start']['mae']:.3f}s med={result['start']['median']:.3f}s "
        f"P90={result['start']['p90']:.3f}s BER50={result['start']['ber_50ms']:.1%}\n"
        f"end_MAE={result['end']['mae']:.3f}s med={result['end']['median']:.3f}s "
        f"P90={result['end']['p90']:.3f}s BER50={result['end']['ber_50ms']:.1%}\n"
        f"boundary_MAE={result['boundary']['mae']:.3f}s med={result['boundary']['median']:.3f}s "
        f"BER10/20/50={result['boundary']['ber_10ms']:.1%}/"
        f"{result['boundary']['ber_20ms']:.1%}/{result['boundary']['ber_50ms']:.1%}"
    )


if __name__ == "__main__":
    cli()
