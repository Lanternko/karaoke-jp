"""PoC: fill Whisper-skipped lyric lines via MMS_FA forced alignment.

Scope: a single time window of one song; emits a sidecar ``ctc_patch.json``
that ``apply_ctc_patch.py`` overlays onto ``aligned.json``. Does NOT modify
canonical pipeline outputs. Intentionally not wired into Snakefile.

Why this exists
---------------
faster-whisper's silero VAD silently drops sustained-vowel sections in
some songs (haru-hikage 113-147 s, 33 s of bridge missed entirely). With
the lyric text known a priori, CTC forced alignment doesn't need VAD or
transcription accuracy; it directly maps the given text to audio frames.
MMS_FA is used because torchaudio bundles it; it expects romanized
transcript, so we deterministically Hepburn-romanize the kana ``reading``
preserved in aligned.json's tokens (those readings already incorporate
any user override decisions).

Usage
-----
    python scripts/ctc_gap_fill.py \\
        --audio outputs/haru-hikage/vocals.wav \\
        --aligned outputs/haru-hikage/aligned.json \\
        --start 113.0 --end 147.0 \\
        --line-start 18 --line-end 21 \\
        --out outputs/haru-hikage/ctc_patch.json
"""
from __future__ import annotations

import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import click
import soundfile as sf
import torch
from torchaudio.functional import forced_align, merge_tokens
from torchaudio.pipelines import MMS_FA as bundle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from karaoke_jp.ruby import kata_to_hira  # noqa: E402


# ---------------------------------------------------------------------------
# Hepburn romanization (deterministic, no external G2P)
# ---------------------------------------------------------------------------

_HEPBURN: dict[str, str] = {
    # gojuon
    "あ": "a", "い": "i", "う": "u", "え": "e", "お": "o",
    "か": "ka", "き": "ki", "く": "ku", "け": "ke", "こ": "ko",
    "さ": "sa", "し": "shi", "す": "su", "せ": "se", "そ": "so",
    "た": "ta", "ち": "chi", "つ": "tsu", "て": "te", "と": "to",
    "な": "na", "に": "ni", "ぬ": "nu", "ね": "ne", "の": "no",
    "は": "ha", "ひ": "hi", "ふ": "fu", "へ": "he", "ほ": "ho",
    "ま": "ma", "み": "mi", "む": "mu", "め": "me", "も": "mo",
    "や": "ya", "ゆ": "yu", "よ": "yo",
    "ら": "ra", "り": "ri", "る": "ru", "れ": "re", "ろ": "ro",
    "わ": "wa", "ゐ": "i", "ゑ": "e", "を": "o",
    "ん": "n",
    # dakuon / handakuon
    "が": "ga", "ぎ": "gi", "ぐ": "gu", "げ": "ge", "ご": "go",
    "ざ": "za", "じ": "ji", "ず": "zu", "ぜ": "ze", "ぞ": "zo",
    "だ": "da", "ぢ": "ji", "づ": "zu", "で": "de", "ど": "do",
    "ば": "ba", "び": "bi", "ぶ": "bu", "べ": "be", "ぼ": "bo",
    "ぱ": "pa", "ぴ": "pi", "ぷ": "pu", "ぺ": "pe", "ぽ": "po",
    # vowel-swallowing small kana (handled separately for yoon below)
}

_YOON: dict[str, str] = {
    "きゃ": "kya", "きゅ": "kyu", "きょ": "kyo",
    "しゃ": "sha", "しゅ": "shu", "しょ": "sho",
    "ちゃ": "cha", "ちゅ": "chu", "ちょ": "cho",
    "にゃ": "nya", "にゅ": "nyu", "にょ": "nyo",
    "ひゃ": "hya", "ひゅ": "hyu", "ひょ": "hyo",
    "みゃ": "mya", "みゅ": "myu", "みょ": "myo",
    "りゃ": "rya", "りゅ": "ryu", "りょ": "ryo",
    "ぎゃ": "gya", "ぎゅ": "gyu", "ぎょ": "gyo",
    "じゃ": "ja", "じゅ": "ju", "じょ": "jo",
    "びゃ": "bya", "びゅ": "byu", "びょ": "byo",
    "ぴゃ": "pya", "ぴゅ": "pyu", "ぴょ": "pyo",
    # extended for foreign loans (rare in lyrics but cheap to include)
    "ふぁ": "fa", "ふぃ": "fi", "ふぇ": "fe", "ふぉ": "fo",
    "うぃ": "wi", "うぇ": "we", "うぉ": "wo",
    "てぃ": "ti", "でぃ": "di", "とぅ": "tu", "どぅ": "du",
    "しぇ": "she", "ちぇ": "che", "じぇ": "je",
}


def romanize_kana(kana: str) -> tuple[str, list[str]]:
    """Return (romaji, untranslatable_chars).

    Hepburn-ish, deterministic. Long-vowel mark ``ー`` repeats the
    preceding vowel. Sokuon ``っ`` doubles the next consonant. Spaces and
    common punctuation collapse to a single space. Anything else is
    flagged so the caller can stop-loss if the kana stream contains
    surprises (kanji slipped through, etc.).
    """
    s = kata_to_hira(kana)
    out: list[str] = []
    bad: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        # Whitespace / common Japanese punctuation -> space
        if ch.isspace() or ch in "、。「」『』,.!?！？・":
            if out and out[-1] != " ":
                out.append(" ")
            i += 1
            continue
        # Long-vowel mark
        if ch == "ー":
            if out:
                last = out[-1][-1]
                if last in "aeiou":
                    out.append(last)
            i += 1
            continue
        # Sokuon: double next consonant
        if ch == "っ":
            if i + 1 < n:
                # peek what next syllable's first roman char is
                nxt2 = s[i + 1 : i + 3]
                nxt1 = s[i + 1]
                roman_next = _YOON.get(nxt2) or _HEPBURN.get(nxt1)
                if roman_next and roman_next[0] not in "aeiou":
                    # 'tch' is the convention for っち
                    if roman_next.startswith("ch"):
                        out.append("t")
                    else:
                        out.append(roman_next[0])
            i += 1
            continue
        # Yoon (kana + small ya/yu/yo) take precedence
        if i + 1 < n and s[i + 1] in "ゃゅょぁぃぅぇぉ":
            di = s[i : i + 2]
            if di in _YOON:
                out.append(_YOON[di])
                i += 2
                continue
        # Single kana
        if ch in _HEPBURN:
            out.append(_HEPBURN[ch])
            i += 1
            continue
        # Hiragana small kana left over (orphaned ぁぃぅぇぉ)
        if ch in "ぁぃぅぇぉ":
            out.append({"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}[ch])
            i += 1
            continue
        # Unmapped char (kanji, latin, symbol)
        bad.append(ch)
        i += 1
    romaji = "".join(out).strip()
    while "  " in romaji:
        romaji = romaji.replace("  ", " ")
    return romaji, bad


# ---------------------------------------------------------------------------
# Per-line transcript building
# ---------------------------------------------------------------------------

@dataclass
class TokenRecord:
    """One lyric token's romanization + the indices of its words in the
    global transcript word list. Filled in by build_transcript; spans
    are filled in after MMS alignment."""
    line_idx: int     # index into the input aligned.json
    token_idx: int    # index into line.tokens (preserves punct positions)
    surface: str
    reading: str | None
    kana_only: bool
    word_indices: list[int]
    start: float | None = None
    end: float | None = None
    score: float | None = None


@dataclass
class LineSpec:
    line_idx: int
    text: str
    tokens: list[TokenRecord]


def build_transcript(
    lines: list[dict],
    line_offset: int,
) -> tuple[list[str], list[LineSpec], list[str]]:
    """Return (word_list, per_line_specs, untranslatable_chars).

    One token's reading becomes one or more whitespace-separated words in
    the MMS transcript; the mapping is tracked in TokenRecord.word_indices
    so that aligned word spans can be rolled back up to per-token spans
    later. Punctuation tokens are skipped (CTC has no token for them and
    midi_timing's _retime_unsung_chars repins them post-allocation).
    """
    words: list[str] = []
    specs: list[LineSpec] = []
    bad_total: list[str] = []
    for offset, ln in enumerate(lines):
        records: list[TokenRecord] = []
        line_idx = line_offset + offset
        for tok_idx, tok in enumerate(ln.get("tokens", [])):
            if tok.get("is_punct"):
                continue
            surf = tok.get("surface") or ""
            kana_only = bool(tok.get("kana_only"))
            kana = surf if kana_only else (tok.get("reading") or surf)
            if not kana:
                continue
            roman, bad = romanize_kana(kana)
            bad_total.extend(bad)
            tok_words = [w for w in roman.split() if w]
            if not tok_words:
                continue
            start_w = len(words)
            words.extend(tok_words)
            end_w = len(words)
            records.append(
                TokenRecord(
                    line_idx=line_idx,
                    token_idx=tok_idx,
                    surface=surf,
                    reading=tok.get("reading"),
                    kana_only=kana_only,
                    word_indices=list(range(start_w, end_w)),
                )
            )
        if records:
            specs.append(LineSpec(line_idx=line_idx, text=ln.get("text", ""), tokens=records))
    return words, specs, bad_total


# ---------------------------------------------------------------------------
# MMS_FA alignment
# ---------------------------------------------------------------------------

def _load_audio(audio_path: Path, start_s: float, end_s: float, target_sr: int) -> tuple[torch.Tensor, float]:
    """Read [start_s, end_s] of audio_path mono at target_sr.

    Returns (tensor [1, T], audio_offset_s) where audio_offset_s is the
    absolute time of frame 0 in the returned tensor (== start_s).
    """
    info = sf.info(str(audio_path))
    sr = info.samplerate
    s_frame = int(start_s * sr)
    e_frame = int(end_s * sr)
    data, _ = sf.read(str(audio_path), start=s_frame, stop=e_frame, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data).unsqueeze(0)  # [1, T]
    if sr != target_sr:
        from torchaudio.functional import resample
        wav = resample(wav, sr, target_sr)
    return wav, start_s


def align_chunk(
    audio_path: Path,
    start_s: float,
    end_s: float,
    words: list[str],
    pad_s: float = 0.5,
    device: str | None = None,
) -> tuple[list[tuple[float, float, float]], list[str]]:
    """Run MMS_FA over [start_s - pad, end_s + pad] and return per-word
    (abs_start_s, abs_end_s, score). Also returns the dict-validated
    cleaned word list (any words containing chars outside the bundle's
    accepted set are flagged but kept; if too many, caller should stop).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    target_sr = bundle.sample_rate
    chunk_start = max(0.0, start_s - pad_s)
    chunk_end = end_s + pad_s
    wav, audio_offset = _load_audio(audio_path, chunk_start, chunk_end, target_sr)
    wav = wav.to(device)

    model = bundle.get_model(with_star=False).to(device)
    model.eval()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    with torch.inference_mode():
        emission, _ = model(wav)

    # Validate / encode words
    accepted_words: list[str] = []
    rejected_words: list[str] = []
    for w in words:
        try:
            tokenizer([w])  # raises if any char not in dict
            accepted_words.append(w)
        except Exception:
            rejected_words.append(w)
    if not accepted_words:
        raise RuntimeError("MMS_FA tokenizer rejected every word; charset mismatch.")

    token_spans = aligner(emission[0], tokenizer(accepted_words))
    # token_spans: list of list[TokenSpan]. Outer = words, inner = tokens.
    # Each TokenSpan has .start (frame), .end (frame, exclusive), .score

    # Frame -> seconds. MMS_FA emission stride is 320 samples / 16 kHz = 20 ms
    # but be safe and recompute from tensor shapes.
    n_frames = emission.shape[1]
    audio_dur = wav.shape[1] / target_sr
    sec_per_frame = audio_dur / n_frames

    word_spans: list[tuple[float, float, float]] = []
    word_iter = iter(accepted_words)
    for spans in token_spans:
        if not spans:
            word_spans.append((float("nan"), float("nan"), 0.0))
            next(word_iter, None)
            continue
        s_frame = spans[0].start
        e_frame = spans[-1].end
        score = float(sum(t.score for t in spans) / len(spans))
        abs_s = audio_offset + s_frame * sec_per_frame
        abs_e = audio_offset + e_frame * sec_per_frame
        word_spans.append((abs_s, abs_e, score))
        next(word_iter, None)

    # Re-interleave NaNs for any rejected words
    if rejected_words:
        # PoC: just append placeholders at original positions. Caller
        # treats NaN spans as 'unaligned'.
        full: list[tuple[float, float, float]] = []
        ai = 0
        for w in words:
            if w in rejected_words:
                full.append((float("nan"), float("nan"), 0.0))
                rejected_words.remove(w)  # only first occurrence
            else:
                full.append(word_spans[ai])
                ai += 1
        word_spans = full

    return word_spans, rejected_words


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--audio", "audio_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--aligned", "aligned_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--start", "start_s", required=True, type=float, help="Window start (s).")
@click.option("--end", "end_s", required=True, type=float, help="Window end (s).")
@click.option("--line-start", "line_start", required=True, type=int, help="First aligned.json line index (inclusive).")
@click.option("--line-end", "line_end", required=True, type=int, help="Last aligned.json line index (inclusive).")
@click.option("--pad", default=0.5, show_default=True, help="Audio padding (s) outside window.")
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False))
def main(
    audio_path: str,
    aligned_path: str,
    start_s: float,
    end_s: float,
    line_start: int,
    line_end: int,
    pad: float,
    out_path: str,
) -> None:
    aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    if line_end >= len(aligned):
        raise click.ClickException(f"line_end={line_end} but aligned has {len(aligned)} lines")

    target_lines = aligned[line_start : line_end + 1]
    print(f"[ctc_gap_fill] window {start_s:.2f}-{end_s:.2f}s, lines [{line_start}..{line_end}]")
    for i, ln in enumerate(target_lines, start=line_start):
        print(f"  [{i}] {ln.get('text','')}")

    # Build transcript
    words, specs, bad = build_transcript(target_lines, line_offset=line_start)
    print(f"[ctc_gap_fill] transcript words: {len(words)}")
    print(f"[ctc_gap_fill] sample: {' '.join(words[:8])}{'...' if len(words) > 8 else ''}")
    if bad:
        unique_bad = sorted(set(bad))
        ratio = len(bad) / max(1, sum(len(w) for w in words) + len(bad))
        print(f"[ctc_gap_fill] WARN: {len(bad)} untranslatable chars ({ratio:.1%}): {unique_bad[:20]}")
        if ratio > 0.05:
            raise click.ClickException("Untranslatable ratio > 5%; aborting (charset mismatch).")

    # Run forced alignment
    word_spans, rejected = align_chunk(Path(audio_path), start_s, end_s, words, pad_s=pad)
    if rejected:
        print(f"[ctc_gap_fill] WARN: {len(rejected)} words rejected by tokenizer: {rejected[:10]}")

    # Roll word spans up to per-token, then per-line
    line_records: list[dict] = []
    for spec in specs:
        token_records: list[dict] = []
        line_starts: list[float] = []
        line_ends: list[float] = []
        line_scores: list[float] = []
        for tok in spec.tokens:
            tok_spans = [word_spans[i] for i in tok.word_indices]
            valid = [s for s in tok_spans if s[0] == s[0]]  # filter NaN
            if not valid:
                token_records.append({
                    "token_idx": tok.token_idx,
                    "surface": tok.surface,
                    "reading": tok.reading,
                    "kana_only": tok.kana_only,
                    "start": None,
                    "end": None,
                    "score": None,
                })
                continue
            t_start = min(v[0] for v in valid)
            t_end = max(v[1] for v in valid)
            t_score = sum(v[2] for v in valid) / len(valid)
            token_records.append({
                "token_idx": tok.token_idx,
                "surface": tok.surface,
                "reading": tok.reading,
                "kana_only": tok.kana_only,
                "start": round(t_start, 3),
                "end": round(t_end, 3),
                "score": round(t_score, 3),
            })
            line_starts.append(t_start)
            line_ends.append(t_end)
            line_scores.append(t_score)
        if not line_starts:
            line_records.append({
                "line_idx": spec.line_idx,
                "text": spec.text,
                "start": None, "end": None, "score": 0.0,
                "tokens": token_records,
            })
            continue
        line_records.append({
            "line_idx": spec.line_idx,
            "text": spec.text,
            "start": round(min(line_starts), 3),
            "end": round(max(line_ends), 3),
            "score": round(sum(line_scores) / len(line_scores), 3),
            "tokens": token_records,
        })

    payload = {
        "backend": "mms-fa",
        "audio_window": [start_s, end_s],
        "padding": pad,
        "line_index_range": [line_start, line_end],
        "lines": line_records,
        "word_spans": [
            {"word": w, "start": round(s, 3) if s == s else None,
             "end": round(e, 3) if e == e else None, "score": round(sc, 3)}
            for w, (s, e, sc) in zip(words, word_spans)
        ],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ctc_gap_fill] wrote {out_path}")
    for rec in line_records:
        print(f"  [{rec['line_idx']}] {rec['start']}-{rec['end']} score={rec['score']}  {rec['text'][:30]}")
        for tr in rec["tokens"]:
            print(f"      tok[{tr['token_idx']}] {tr['surface']:<6s} {tr['start']}-{tr['end']} score={tr['score']}")


if __name__ == "__main__":
    main()
