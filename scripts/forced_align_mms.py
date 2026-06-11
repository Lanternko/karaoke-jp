"""CTC forced alignment as the lyrics timing source (survey §3.5 phase 1).

Replaces NW-on-ASR + mora->note as the *timing* source: the mora sequence
(from user lyrics, reading-corrected via overrides) is romanized into the
aligner's letter vocabulary and force-aligned against the separated vocals.
Timing then comes from frame-level acoustic posteriors under the lyric
constraint — line-final particles own their actual phones (no note stealing),
ad-libs are absorbed by CTC blank, and onsets sit at consonant starts (the
same event the Audacity line gold marks).

Model: NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn — facebook/mms-300m
fine-tuned on Karaoke Mugen per-syllable timings (romaji, lowercase a-z +
"'" + "|" word separator). Per-mora words are in-domain.

Output keeps the aligned sidecar schema, so every downstream stage
(line_end_repair, export_lrc, markers, render, eval_alignment) is unchanged.
"""
from __future__ import annotations

import copy
import json
import sys
import unicodedata
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from midi_timing import (  # noqa: E402
    _retime_lines_after_char_update,
    _retime_unsung_chars,
    _writeback_char_timings,
    expand_line_to_morae,
)
from karaoke_jp.ruby import kata_to_hira  # noqa: E402

MORA_ROMAJI = {
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
    "わ": "wa", "ゐ": "wi", "ゑ": "we", "を": "o", "ん": "n",
    "ゔ": "vu",
}
_SMALL_Y = {"ゃ": "ya", "ゅ": "yu", "ょ": "yo"}
_SMALL_V = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}
_COMBINING_HOSTS = {"て", "で", "ふ", "う", "ゔ", "し", "ち", "じ"}
_VOWEL_LETTERS = set("aiueo")


def _particle_romaji(kana: str, tok: dict) -> str | None:
    """Sung particles differ from spelled kana: は=wa, へ=e (を=o is in the table)."""
    if not tok.get("kana_only") or len(tok.get("surface", "")) != 1:
        return None
    if tok.get("pos", "").startswith("助詞"):
        return {"は": "wa", "へ": "e"}.get(kana)
    return None


def records_to_words(records: list[dict]) -> tuple[list[str], list[list[int]]]:
    """Group mora records into aligner words with per-record letter attribution.

    Returns (words, letters_per_record) where letters_per_record[i] lists the
    flat letter positions (0-based over ''.join(words)) owned by record i.
    Small kana combine with their host; っ contributes the doubled consonant
    of the next mora; ー repeats the previous vowel as its own word (karaoke
    k-timing convention gives long vowels their own unit).
    """
    units: list[tuple[list[int], str, bool]] = []
    for i, rec in enumerate(records):
        kana = kata_to_hira(rec["kana"])
        ch = kana
        if ch.isascii() and ch.isalpha():
            units.append(([i], ch.lower(), True))
            continue
        if ch in _SMALL_Y and units:
            owners, rom, _ = units[-1]
            base = rom
            prefix = base[:-1] if base and base[-1] in _VOWEL_LETTERS else base
            small = _SMALL_Y[ch]
            if prefix.endswith("h") or prefix.endswith("j"):
                small = small[-1]
            units[-1] = (owners + [i], prefix + small, False)
            continue
        if ch in _SMALL_V and units:
            owners, rom, _ = units[-1]
            prev_kana = kata_to_hira(records[owners[0]]["kana"]) if owners else ""
            if prev_kana in _COMBINING_HOSTS:
                prefix = rom[:-1] if rom and rom[-1] in _VOWEL_LETTERS else rom
                if prev_kana == "う":
                    prefix = "w"
                units[-1] = (owners + [i], prefix + _SMALL_V[ch], False)
            else:
                units.append(([i], _SMALL_V[ch], False))
            continue
        if ch == "ー":
            prev_rom = units[-1][1] if units else "a"
            vowel = next((c for c in reversed(prev_rom) if c in _VOWEL_LETTERS), "a")
            units.append(([i], vowel, False))
            continue
        if ch == "っ":
            units.append(([i], "\x00", False))
            continue
        rom = MORA_ROMAJI.get(ch)
        if rom is None:
            continue
        units.append(([i], rom, False))

    words: list[str] = []
    letters_per_record: list[list[int]] = [[] for _ in records]
    pos = 0
    k = 0
    while k < len(units):
        owners, rom, is_latin = units[k]
        if rom == "\x00":
            nxt = units[k + 1] if k + 1 < len(units) else None
            if nxt and nxt[1] and nxt[1][0] not in _VOWEL_LETTERS and nxt[1] != "\x00":
                merged_owners = [(owners, nxt[1][0]), (nxt[0], nxt[1])]
                word = nxt[1][0] + nxt[1]
                for own_list, letters in merged_owners:
                    for _ in letters:
                        for o in own_list:
                            letters_per_record[o].append(pos)
                        pos += 1
                words.append(word)
                k += 2
                continue
            k += 1
            continue
        if is_latin:
            word_units = [units[k]]
            while k + 1 < len(units) and units[k + 1][2]:
                k += 1
                word_units.append(units[k])
            word = "".join(u[1] for u in word_units)
            for own_list, letters, _ in word_units:
                for _ in letters:
                    for o in own_list:
                        letters_per_record[o].append(pos)
                    pos += 1
            words.append(word)
            k += 1
            continue
        for ci, _ in enumerate(rom):
            owner_set = owners if len(owners) == 1 else (
                owners[:-1] if ci < len(rom) - 1 else owners[-1:])
            for o in owner_set:
                letters_per_record[o].append(pos)
        pos += len(rom)
        words.append(rom)
        k += 1
    return words, letters_per_record


def _chunk_boundaries(wave, sr: int, chunk_s: float) -> list[tuple[int, int]]:
    """Split at the quietest 20ms inside +-5s of each nominal boundary."""
    import torch

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
        cut = lo + int(rms.argmin().item()) * win
        bounds.append(cut)
    bounds.append(n)
    return list(zip(bounds[:-1], bounds[1:]))


@click.command()
@click.option("--vocals", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--aligned", "aligned_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Existing sidecar (schema + reading-corrected tokens); all timings are replaced.")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--model", "model_id", default="NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn",
              show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--chunk-seconds", default=110.0, show_default=True,
              help="Emission chunking bound (memory); cuts snap to the quietest nearby 20ms.")
def main(vocals: str, aligned_path: str, out_path: str, model_id: str,
         device: str, chunk_seconds: float) -> None:
    import torch
    import torchaudio
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    lines = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    lines = copy.deepcopy(lines)

    per_line_records: list[list[dict]] = [expand_line_to_morae(ln) for ln in lines]
    all_records: list[dict] = [r for recs in per_line_records for r in recs]
    for recs, line in zip(per_line_records, lines):
        for r in recs:
            tok = next((t for t in line["tokens"] if any(c is r["char"] for c in t.get("chars", []))), {})
            override = _particle_romaji(kata_to_hira(r["kana"]), tok)
            if override:
                r["kana_romaji_override"] = override

    words: list[str] = []
    letters_per_record: list[list[int]] = []
    offset_letters = 0
    for recs in per_line_records:
        for r in recs:
            if "kana_romaji_override" in r:
                r["kana"] = {"wa": "わ", "e": "え"}[r["kana_romaji_override"]]
        w, lpr = records_to_words(recs)
        words.extend(w)
        for lst in lpr:
            letters_per_record.append([offset_letters + p for p in lst])
        offset_letters += sum(len(x) for x in w)

    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
    vocab = processor.tokenizer.get_vocab()
    blank_id = processor.tokenizer.pad_token_id
    sep_id = vocab["|"]

    targets: list[int] = []
    target_letter_pos: list[int] = []
    lp = 0
    for wi, word in enumerate(words):
        if wi > 0:
            targets.append(sep_id)
            target_letter_pos.append(-1)
        for ch in word:
            targets.append(vocab[ch])
            target_letter_pos.append(lp)
            lp += 1

    import soundfile as sf

    data, sr = sf.read(vocals, dtype="float32")
    wave = torch.from_numpy(data.T if data.ndim > 1 else data[None, :]).mean(dim=0)
    if sr != 16000:
        wave = torchaudio.functional.resample(wave, sr, 16000)
        sr = 16000

    emissions = []
    with torch.inference_mode():
        for a, b in _chunk_boundaries(wave, sr, chunk_seconds):
            inp = wave[a:b].unsqueeze(0).to(device)
            logits = model(inp).logits
            emissions.append(torch.log_softmax(logits, dim=-1).cpu())
    emission = torch.cat(emissions, dim=1)

    ratio = wave.shape[-1] / emission.shape[1] / sr
    tgt = torch.tensor([targets], dtype=torch.int32)
    path, scores = torchaudio.functional.forced_align(emission, tgt, blank=blank_id)
    spans = torchaudio.functional.merge_tokens(path[0], scores[0], blank=blank_id)

    letter_times: dict[int, tuple[float, float]] = {}
    si = 0
    for ti, lpos in enumerate(target_letter_pos):
        while si < len(spans) and spans[si].token != targets[ti]:
            si += 1
        if si >= len(spans):
            break
        if lpos >= 0:
            letter_times[lpos] = (spans[si].start * ratio, spans[si].end * ratio)
        si += 1

    rec_spans: list[tuple[float, float] | None] = []
    for lst in letters_per_record:
        ts = [letter_times[p] for p in lst if p in letter_times]
        rec_spans.append((min(t[0] for t in ts), max(t[1] for t in ts)) if ts else None)
    for i, sp in enumerate(rec_spans):
        if sp is None:
            prev_end = next((rec_spans[j][1] for j in range(i - 1, -1, -1) if rec_spans[j]), 0.0)
            nxt_start = next((rec_spans[j][0] for j in range(i + 1, len(rec_spans)) if rec_spans[j]), prev_end)
            rec_spans[i] = (prev_end, max(prev_end, nxt_start))

    idx = 0
    updated = 0
    for recs, line in zip(per_line_records, lines):
        n = len(recs)
        if n == 0:
            idx += n
            continue
        spans_line = rec_spans[idx:idx + n]
        idx += n
        _writeback_char_timings(recs, spans_line)
        for tok in line.get("tokens", []):
            _retime_unsung_chars(tok.get("chars", []))
        updated += 1
    _retime_lines_after_char_update(lines)

    Path(out_path).write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding="utf-8")
    click.echo(f"[mms-align] {updated}/{len(lines)} lines retimed, "
               f"{len(words)} words / {len(targets)} targets, "
               f"emission {emission.shape[1]} frames @ {ratio*1000:.1f}ms -> {out_path}")


if __name__ == "__main__":
    main()
