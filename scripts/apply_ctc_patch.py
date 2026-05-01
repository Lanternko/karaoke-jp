"""Overlay a ctc_patch.json onto aligned.json (sidecar output only).

Modes
-----
* default: only line.start/end are patched. Char/token timings remain at
  whatever upstream Whisper / align_lyrics produced. Useful for verifying
  that line-level CTC alignment is sane before trusting per-char output.
* ``--stretch-chars``: linear-rescale every char/token start/end into the
  new line window. Helps if the existing chars were roughly correct but
  the line window was wrong; useless when upstream had collapsed all the
  chars into a single endpoint (the haru-hikage bridge case).
* ``--apply-ctc-chars``: replace per-char/token timings with values
  derived from the patch's ``tokens`` array, distributing each token's
  CTC span across its chars using the same mora-allocation rule
  midi_timing.py uses (so the per-char hints we write are coherent with
  how midi_timing later allocates notes).

PoC support tool for ctc_gap_fill.py. Always writes a NEW aligned file
so the canonical aligned.json stays untouched.

Usage
-----
    python scripts/apply_ctc_patch.py \\
        --aligned outputs/haru-hikage/aligned.json \\
        --patch outputs/haru-hikage/ctc_patch.json \\
        --out outputs/haru-hikage/aligned.ctc.json \\
        [--apply-ctc-chars | --stretch-chars]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from karaoke_jp.lrc_export import split_furigana  # noqa: E402
from karaoke_jp.ruby import kata_to_hira  # noqa: E402

# Reuse midi_timing's helpers to keep semantics identical.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from midi_timing import _is_sung_char, _split_morae_across_chars  # noqa: E402


# ---------------------------------------------------------------------------
# Stretch mode (Phase 1.5 fallback)
# ---------------------------------------------------------------------------

def _stretch_line(line: dict, new_start: float, new_end: float) -> None:
    """Linear-rescale every char/token start/end on this line to fit
    [new_start, new_end]. Assumes existing chars share a common origin
    (line.start) and span (line.end - line.start)."""
    old_start = float(line.get("start", new_start))
    old_end = float(line.get("end", new_end))
    old_span = max(1e-6, old_end - old_start)
    new_span = new_end - new_start

    def remap(t: float) -> float:
        rel = (float(t) - old_start) / old_span
        rel = max(0.0, min(1.0, rel))
        return round(new_start + rel * new_span, 3)

    for tok in line.get("tokens", []):
        if "start" in tok:
            tok["start"] = remap(tok["start"])
        if "end" in tok:
            tok["end"] = remap(tok["end"])
        for ch in tok.get("chars", []) or []:
            if "start" in ch:
                ch["start"] = remap(ch["start"])
            if "end" in ch:
                ch["end"] = remap(ch["end"])

    line["start"] = round(new_start, 3)
    line["end"] = round(new_end, 3)


# ---------------------------------------------------------------------------
# CTC-char mode (Phase 1.5+: distribute token span per mora rule)
# ---------------------------------------------------------------------------

def _distribute_token_span(
    tok: dict,
    span_start: float,
    span_end: float,
) -> None:
    """Mutate ``tok``'s chars start/end so each sung char's window is
    proportional to its mora count, mirroring midi_timing's
    expand_line_to_morae logic.

    Punctuation / unsung chars are not retimed here; midi_timing's
    _retime_unsung_chars later pins them to neighbouring sung-char
    boundaries, so we leave them alone.
    """
    chars = tok.get("chars") or []
    sung_chars = [c for c in chars if _is_sung_char(c["char"])]
    if not sung_chars:
        return

    reading = tok.get("reading")
    surface = tok.get("surface", "")
    if reading and not tok.get("kana_only"):
        kata = kata_to_hira(reading)
        segments = split_furigana(surface, kata)
    else:
        segments = [(surface, None, 0, len(surface))]

    char_morae: list[tuple[dict, int]] = []
    for _seg_text, seg_reading, c_lo, c_hi in segments:
        seg_chars = chars[c_lo:c_hi]
        sung_seg_chars = [c for c in seg_chars if _is_sung_char(c["char"])]
        if not sung_seg_chars:
            continue
        if seg_reading is None:
            for ch in sung_seg_chars:
                char_morae.append((ch, 1))
        else:
            n_morae = len(seg_reading)
            splits = _split_morae_across_chars(n_morae, len(sung_seg_chars))
            for ch, n in zip(sung_seg_chars, splits, strict=True):
                if n > 0:
                    char_morae.append((ch, n))

    if not char_morae:
        return

    total_morae = sum(n for _, n in char_morae)
    span = max(1e-6, span_end - span_start)

    cum = 0
    for ch, n in char_morae:
        s_frac = cum / total_morae
        cum += n
        e_frac = cum / total_morae
        ch["start"] = round(span_start + s_frac * span, 3)
        ch["end"] = round(span_start + e_frac * span, 3)

    # Token's own start/end track its char range
    tok["start"] = sung_chars[0]["start"]
    tok["end"] = sung_chars[-1]["end"]


def _apply_ctc_chars(line: dict, line_rec: dict) -> int:
    """Apply per-token CTC spans onto a line. Returns count of tokens
    written. Tokens missing from the patch (unaligned, NaN) are skipped
    silently; midi_timing will fall back to whatever value they currently
    hold."""
    tok_records = {t["token_idx"]: t for t in line_rec.get("tokens", [])}
    written = 0
    for idx, tok in enumerate(line.get("tokens", [])):
        rec = tok_records.get(idx)
        if rec is None:
            continue
        if rec.get("start") is None or rec.get("end") is None:
            continue
        _distribute_token_span(tok, float(rec["start"]), float(rec["end"]))
        written += 1
    line["start"] = round(float(line_rec["start"]), 3)
    line["end"] = round(float(line_rec["end"]), 3)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--aligned", "aligned_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--patch", "patch_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False))
@click.option("--stretch-chars", "mode_stretch", is_flag=True, default=False,
              help="Linear-rescale per-char/token timings into the new line window.")
@click.option("--apply-ctc-chars", "mode_ctc", is_flag=True, default=False,
              help="Write per-char/token timings from the patch's CTC spans (mora-distributed).")
def main(
    aligned_path: str,
    patch_path: str,
    out_path: str,
    mode_stretch: bool,
    mode_ctc: bool,
) -> None:
    if mode_stretch and mode_ctc:
        raise click.ClickException("Pass at most one of --stretch-chars / --apply-ctc-chars.")
    mode = "ctc" if mode_ctc else ("stretch" if mode_stretch else "line-only")

    aligned = json.loads(Path(aligned_path).read_text(encoding="utf-8"))
    patch = json.loads(Path(patch_path).read_text(encoding="utf-8"))

    applied = 0
    skipped = 0
    chars_written = 0
    for rec in patch.get("lines", []):
        idx = rec["line_idx"]
        new_start = rec.get("start")
        new_end = rec.get("end")
        if idx >= len(aligned):
            click.echo(f"  [skip] line_idx={idx} out of range")
            skipped += 1
            continue
        if new_start is None or new_end is None:
            click.echo(f"  [skip] line_idx={idx} has null start/end (alignment failed)")
            skipped += 1
            continue
        line = aligned[idx]
        old_s, old_e = line.get("start"), line.get("end")
        click.echo(
            f"  [{idx}] {old_s}->{new_start}  {old_e}->{new_end}  "
            f"(mode={mode})  {line.get('text','')[:30]}"
        )
        if mode == "ctc":
            chars_written += _apply_ctc_chars(line, rec)
        elif mode == "stretch":
            _stretch_line(line, new_start, new_end)
        else:
            line["start"] = round(new_start, 3)
            line["end"] = round(new_end, 3)
        applied += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(aligned, ensure_ascii=False, indent=2), encoding="utf-8")
    click.echo(
        f"[apply_ctc_patch] mode={mode} applied={applied} skipped={skipped} "
        f"tokens_written={chars_written} -> {out_path}"
    )


if __name__ == "__main__":
    main()
