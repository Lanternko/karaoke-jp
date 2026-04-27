"""Run faster-whisper on a vocals file and emit segment + word timestamps as JSON.

Usage:
    python scripts/run_asr.py outputs/<song>/vocals.wav -o outputs/<song>/asr.json
"""
from __future__ import annotations

import json
from pathlib import Path

import click


@click.command()
@click.argument("vocals_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "-o", "out_path", type=click.Path(dir_okay=False), required=True)
@click.option("--model", default="large-v3", help="faster-whisper model name.")
@click.option(
    "--language",
    default="ja",
    help="Language hint. 'ja' is what we want; do not pass an empty string.",
)
@click.option("--device", default="cuda")
@click.option(
    "--compute-type",
    default="float16",
    help="float16 / int8_float16 / float32. float16 is fast on RTX 5090.",
)
@click.option(
    "--lyrics",
    "lyrics_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to lyrics.txt; used as initial_prompt to bias ASR against "
    "missing quiet verses (Whisper's 224-token cap is respected by truncation).",
)
def main(
    vocals_path: str,
    out_path: str,
    model: str,
    language: str,
    device: str,
    compute_type: str,
    lyrics_path: str | None,
) -> None:
    from faster_whisper import WhisperModel

    initial_prompt = None
    if lyrics_path:
        # ~200 chars stays inside Whisper's 224-token prompt window for JP.
        raw = Path(lyrics_path).read_text(encoding="utf-8")
        compact = "".join(raw.split())  # drop newlines and 全角空白
        initial_prompt = compact[:200]
        print(f"using initial_prompt ({len(initial_prompt)} chars)", flush=True)

    print(f"loading {model} ({compute_type}) on {device}...", flush=True)
    whisper = WhisperModel(model, device=device, compute_type=compute_type)

    print(f"transcribing {vocals_path} ...", flush=True)
    segments, info = whisper.transcribe(
        vocals_path,
        language=language,
        word_timestamps=True,
        initial_prompt=initial_prompt,
        # Singing has long held vowels; sensitive VAD picks up quiet entries
        # like the verse-1 opening that the default missed.
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500, "threshold": 0.3},
        beam_size=5,
        condition_on_previous_text=False,  # avoid hallucinated repeats in choruses
        temperature=0.0,
    )

    out: list[dict] = []
    for seg in segments:
        words = []
        if seg.words:
            for w in seg.words:
                words.append(
                    {
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "word": w.word,
                        "probability": round(w.probability, 4),
                    }
                )
        out.append(
            {
                "id": seg.id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text,
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
                "words": words,
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(
            {
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "duration": round(info.duration, 3),
                "segments": out,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    n_words = sum(len(s["words"]) for s in out)
    print(
        f"{len(out)} segments, {n_words} words, "
        f"lang={info.language} (p={info.language_probability:.3f}) -> {out_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
