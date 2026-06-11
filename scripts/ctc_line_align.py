#!/usr/bin/env python3
"""CTC forced alignment of lyrics to separated vocals — STOCK MMS_FA bundle.

Cross-check twin of scripts/forced_align_mms.py (which uses the
NextFire/mms-300m karaoke-ja fine-tune in the dedicated align venv): this
one uses torchaudio's stock multilingual MMS_FA pipeline in the melody venv,
zero extra dependencies. Two independent checkpoints reaching the same
line-start numbers (chidori MAE 0.053 stock vs 0.056 fine-tune) is the
replication evidence that CTC-as-timing-source is robust, not a checkpoint
artifact.

Aligns the KNOWN lyrics (romanized morae from tokens.json) directly against
frame-level acoustic posteriors — no ASR transcript in the loop. A star
token between lines absorbs ad-libs / breaths / interludes (the よ×3 class).
Known limitation (survey §3.2): raw CTC ends are peaky → all line ends land
early (bias ≈ −0.5s); pair with line_end_repair for usable offsets.

Output: line-level + mora-level timings JSON. With --gold, also reports
line start/end MAE / median / p90 / within-250ms against a gold TSV.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ── kana → romaji (Hepburn-ish, MMS dictionary is lowercase a-z) ────────────
_BASE = {
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "i", "ゑ": "e", "を": "o", "ん": "n",
    "ゔ": "vu",
}
_SMALL_Y = {"ゃ": "ya", "ゅ": "yu", "ょ": "yo"}
_SMALL_V = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}


def kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c for c in s)


def kana_to_mora_words(kana: str) -> list[str]:
    """Split a hiragana string into morae and romanize each; geminate っ folds
    into the next mora, long-vowel ー extends the previous one."""
    morae: list[str] = []
    i = 0
    pending_geminate = False
    while i < len(kana):
        c = kana[i]
        if c == "っ":
            pending_geminate = True
            i += 1
            continue
        if c == "ー":
            if morae:
                morae[-1] = morae[-1] + morae[-1][-1]  # repeat last vowel
            i += 1
            continue
        if c in _SMALL_V:
            if morae:
                morae[-1] = morae[-1][:-1] + _SMALL_V[c]
            else:
                morae.append(_SMALL_V[c])
            i += 1
            continue
        if c not in _BASE:
            i += 1  # not kana (punctuation slipped through) — skip
            continue
        rom = _BASE[c]
        if i + 1 < len(kana) and kana[i + 1] in _SMALL_Y:
            glide = _SMALL_Y[kana[i + 1]]
            # き+ゃ -> kya, し+ゃ -> sha, ち+ゃ -> cha, じ+ゃ -> ja
            head = rom[:-1]
            if head in ("sh", "ch", "j"):
                rom = head + glide[1:]
            else:
                rom = head + glide
            i += 1
        if pending_geminate and rom:
            rom = rom[0] + rom
            pending_geminate = False
        morae.append(rom)
        i += 1
    return [m for m in morae if m]


def line_kana(line: dict) -> str:
    """Sung kana for one tokens.json line: kanji tokens use their reading,
    kana tokens their surface."""
    parts = []
    for tok in line.get("tokens", []):
        if tok.get("is_punct"):
            continue
        reading = tok.get("reading")
        if reading and not tok.get("kana_only"):
            parts.append(kata_to_hira(reading))
        else:
            parts.append(kata_to_hira(tok.get("surface", "")))
    return "".join(parts)


@click.command()
@click.option("--vocals", "vocals_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--tokens", "tokens_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--gold", "gold_path", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Line-gold TSV (line_idx/gold_start/gold_end) for MAE report.")
@click.option("--device", default="cuda", show_default=True)
def main(vocals_path, tokens_path, out_path, gold_path, device):
    import torch
    import torchaudio

    lines = json.loads(Path(tokens_path).read_text(encoding="utf-8"))
    line_words: list[list[str]] = []
    for ln in lines:
        words = kana_to_mora_words(line_kana(ln))
        line_words.append(words)

    # transcript: star between lines (and at both ends) absorbs ad-libs,
    # breaths and instrumental gaps so they never steal a lyric mora
    transcript: list[str] = ["*"]
    spans_of_line: list[tuple[int, int]] = []  # word-index range per line
    for words in line_words:
        s = len(transcript)
        transcript.extend(words)
        spans_of_line.append((s, len(transcript)))
        transcript.append("*")

    import soundfile as sf

    bundle = torchaudio.pipelines.MMS_FA
    data, sr = sf.read(vocals_path, dtype="float32")
    wav = torch.from_numpy(data.T if data.ndim > 1 else data[None, :])
    if wav.dim() == 1:
        wav = wav[None, :]
    wav = wav.mean(0, keepdim=True)
    if sr != bundle.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, bundle.sample_rate)
        sr = bundle.sample_rate

    model = bundle.get_model(with_star=True).to(device).eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    with torch.inference_mode():
        emission, _ = model(wav.to(device))
        word_spans = aligner(emission[0], tokenizer(transcript))
    ratio = wav.size(1) / emission.size(1)

    def t(frame: int) -> float:
        return round(frame * ratio / sr, 4)

    out_lines = []
    for li, (ws, we) in enumerate(spans_of_line):
        words = word_spans[ws:we]
        if not words:
            out_lines.append({"line_idx": li, "text": lines[li]["text"],
                              "start": None, "end": None, "morae": []})
            continue
        morae = [{"rom": transcript[ws + k],
                  "start": t(sp[0].start), "end": t(sp[-1].end)}
                 for k, sp in enumerate(words)]
        out_lines.append({
            "line_idx": li,
            "text": lines[li]["text"],
            "start": morae[0]["start"],
            "end": morae[-1]["end"],
            "morae": morae,
        })

    Path(out_path).write_text(json.dumps(out_lines, ensure_ascii=False, indent=1))
    click.echo(f"[ctc-align] {len(out_lines)} lines -> {out_path}")

    if gold_path:
        import numpy as np
        gold = list(csv.DictReader(open(gold_path), delimiter="\t"))
        ds, de = [], []
        for g in gold:
            li = int(g["line_idx"])
            if li >= len(out_lines) or out_lines[li]["start"] is None:
                continue
            ds.append(out_lines[li]["start"] - float(g["gold_start"]))
            de.append(out_lines[li]["end"] - float(g["gold_end"]))
        ds, de = np.array(ds), np.array(de)
        for name, d in (("start", ds), ("end", de)):
            a = np.abs(d)
            click.echo(
                f"[ctc-align] line {name}: MAE {a.mean():.3f} median {np.median(a):.3f} "
                f"p90 {np.percentile(a, 90):.3f} within250 {(a <= 0.25).mean():.0%} "
                f"bias {d.mean():+.3f} (n={len(d)})")


if __name__ == "__main__":
    main()
