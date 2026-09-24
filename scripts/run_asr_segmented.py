"""Run faster-whisper on RMS-VAD segments instead of Whisper's own VAD.

Companion to ``scripts/rms_vad_segments.py``.  For each ``(start, end)`` segment
we slice the decoded vocals and transcribe *that fixed interval* with
``vad_filter=False`` — Whisper never gets a chance to re-drop a quiet sung entry
or fuse a whole phrase, because the boundaries are already fixed by the vocals'
energy.  Word/segment timestamps are offset back into absolute song time and
concatenated, so the output is schema-identical to ``run_asr.py``'s ``asr.json``
and drops straight into ``scripts/align_lyrics.py``.

This is a *sidecar*: it writes ``asr.vad.json`` (or whatever ``-o`` says) and
never overwrites the canonical ``asr.json``.  ``run_asr.py`` is untouched; we
reuse its ``is_hallucinated_segment`` so the hallucination filter stays in one
place.

Usage:
    python scripts/run_asr_segmented.py outputs/<song>/vocals.wav \
        --segments outputs/<song>/rms_segments.json \
        -o outputs/<song>/asr.vad.json
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import click
import numpy as np

# Import is_hallucinated_segment from run_asr.py without making scripts/ a
# package (it isn't one).  Load the sibling module by path.
_RUN_ASR = Path(__file__).resolve().parent / "run_asr.py"
_spec = importlib.util.spec_from_file_location("_run_asr", _RUN_ASR)
_run_asr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_asr)  # type: ignore[union-attr]
is_hallucinated_segment = _run_asr.is_hallucinated_segment


@click.command()
@click.argument("vocals_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--segments", "segments_path",
              type=click.Path(exists=True, dir_okay=False), required=True,
              help="rms_segments.json from scripts/rms_vad_segments.py")
@click.option("--out", "-o", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--model", default="large-v3")
@click.option("--language", default="ja")
@click.option("--device", default="cuda")
@click.option("--compute-type", default="float16")
@click.option(
    "--lyrics", "lyrics_path",
    type=click.Path(exists=True, dir_okay=False), default=None,
    help="Optional lyrics.txt head used as initial_prompt. Off by default: with "
    "fixed RMS-VAD boundaries the quiet-entry bias is no longer needed, and a "
    "verse-1 prompt can leak verse-1 text into chorus/bridge clips.",
)
def main(
    vocals_path: str,
    segments_path: str,
    out_path: str,
    model: str,
    language: str,
    device: str,
    compute_type: str,
    lyrics_path: str | None,
) -> None:
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    seg_data = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    windows = [(s["start"], s["end"]) for s in seg_data["segments"]]
    sr = int(seg_data.get("params", {}).get("sr", 16000))

    initial_prompt = None
    if lyrics_path:
        raw = Path(lyrics_path).read_text(encoding="utf-8")
        initial_prompt = "".join(raw.split())[:150]
        print(f"using initial_prompt ({len(initial_prompt)} chars)", flush=True)

    print(f"decoding {vocals_path} @ {sr} Hz ...", flush=True)
    y = np.asarray(decode_audio(vocals_path, sampling_rate=sr), dtype=np.float32)
    total_dur = len(y) / sr

    print(f"loading {model} ({compute_type}) on {device}...", flush=True)
    whisper = WhisperModel(model, device=device, compute_type=compute_type)

    print(f"transcribing {len(windows)} RMS-VAD segment(s) ...", flush=True)
    out: list[dict] = []
    dropped: list[dict] = []
    next_id = 0
    for w_start, w_end in windows:
        i0 = max(0, int(round(w_start * sr)))
        i1 = min(len(y), int(round(w_end * sr)))
        if i1 - i0 < int(0.1 * sr):  # skip <100 ms slivers
            continue
        clip = y[i0:i1]
        segments, _info = whisper.transcribe(
            clip,
            language=language,
            word_timestamps=True,
            initial_prompt=initial_prompt,
            vad_filter=False,  # boundaries already come from RMS-VAD
            beam_size=5,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        for seg in segments:
            text = seg.text
            s_abs = round(seg.start + w_start, 3)
            e_abs = round(seg.end + w_start, 3)
            if is_hallucinated_segment(text):
                dropped.append({"start": s_abs, "end": e_abs, "text": text[:80]})
                continue
            words = []
            for word in seg.words or []:
                words.append({
                    "start": round(word.start + w_start, 3),
                    "end": round(word.end + w_start, 3),
                    "word": word.word,
                    "probability": round(word.probability, 4),
                })
            out.append({
                "id": next_id,
                "start": s_abs,
                "end": e_abs,
                "text": text,
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
                "words": words,
                "rms_window": [round(w_start, 3), round(w_end, 3)],
            })
            next_id += 1

    out.sort(key=lambda s: s["start"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(
            {
                "language": language,
                "duration": round(total_dur, 3),
                "source_segments": segments_path,
                "segments": out,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    n_words = sum(len(s["words"]) for s in out)
    print(f"{len(out)} segments, {n_words} words -> {out_path}", flush=True)
    if dropped:
        print(f"dropped {len(dropped)} hallucinated segment(s):", flush=True)
        for d in dropped:
            print(f"  {d['start']:7.2f}-{d['end']:7.2f}  {d['text']!r}", flush=True)


if __name__ == "__main__":
    main()
