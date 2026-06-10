"""Run alignment ablations on local or manifest-provided gold sets.

This is an experiment harness, not the canonical pipeline. It sweeps opt-in
``midi_timing`` priors plus ``line_end_repair`` parameters, evaluates every
variant with ``eval_alignment.py`` metrics, and writes ranked TSV/JSON reports.

Default local suite:

* ``haru-hikage`` with ``data/alignment_gold/haru-hikage.gold.tsv``
* ``tuki-zero`` with ``data/alignment_gold/tuki-zero.gold.tsv``
* ``chidori`` with ``data/alignment_gold/chidori.gold.tsv``

External/literature datasets can be evaluated once converted to the same
sidecar schema through ``--manifest``. See
``experiments/literature_alignment_manifest.example.json``.
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import click


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


midi_timing = _load_script("ablation_midi_timing", "scripts/midi_timing.py")
line_end_repair = _load_script("ablation_line_end_repair", "scripts/line_end_repair.py")
line_start_repair = _load_script("ablation_line_start_repair", "scripts/line_start_repair.py")
eval_alignment = _load_script("ablation_eval_alignment", "scripts/eval_alignment.py")


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    song_id: str
    aligned: Path
    midi: Path
    vocals: Path
    gold: Path
    source: str = "local"
    notes: str = ""


@dataclass(frozen=True)
class MidiVariant:
    name: str
    mode: str = "mora"
    margin: float = 0.4
    allocator: str = "greedy"
    first_mora_min_delay: float | None = None
    first_mora_gate_prev_gap: float | None = None
    first_mora_gate_lead_tolerance: float | None = None
    absorb_trailing_notes: bool = False
    next_line_hint_guard: float | None = None
    next_line_hint_min_start_delay: float | None = None
    dp_skip_penalty: float = 0.20
    dp_extra_note_penalty: float = 0.06
    dp_max_notes_per_mora: int = 4


@dataclass(frozen=True)
class RepairVariant:
    name: str
    tail_top_db: float = 26.0
    max_extend: float = 2.0
    next_guard: float = 0.25
    tail_gap: float = 0.18


@dataclass(frozen=True)
class StartRepairVariant:
    name: str
    enabled: bool = False
    max_shift: float = 1.2
    min_late: float = 0.45
    onset_top_db: float = 30.0
    onset_sustain: float = 0.096
    prev_guard: float = 0.04
    blend: float = 0.0
    min_move: float = 0.0
    skip_first_line: bool = False


@dataclass
class AblationResult:
    dataset_id: str
    song_id: str
    source: str
    midi_variant: str
    start_repair_variant: str
    repair_variant: str
    n: int
    start_mae: float
    start_median: float
    start_p90: float
    start_within_250ms: float
    start_within_500ms: float
    start_bias: float
    end_mae: float
    end_median: float
    end_p90: float
    end_within_250ms: float
    end_within_500ms: float
    end_bias: float
    stress_n: int
    zero_duration_sung_chars: int
    backward_sung_chars: int
    overlap_sung_chars: int
    short_sung_lines: int
    start_repair_changes: int
    score: float
    status: str = "ok"
    detail: str = ""


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def default_local_specs(include_candidates: bool) -> list[DatasetSpec]:
    specs = [
        DatasetSpec(
            dataset_id="haru-hikage/vad",
            song_id="haru-hikage",
            aligned=ROOT / "outputs/haru-hikage/aligned.vad.json",
            midi=ROOT / "outputs/haru-hikage/melody_quantized.mid",
            vocals=ROOT / "outputs/haru-hikage/vocals.wav",
            gold=ROOT / "data/alignment_gold/haru-hikage.gold.tsv",
            notes="user Audacity gold, line 20-31",
        ),
        DatasetSpec(
            dataset_id="tuki-zero/vad",
            song_id="tuki-zero",
            aligned=ROOT / "outputs/tuki-zero/aligned.vad.json",
            midi=ROOT / "outputs/tuki-zero/melody_quantized.mid",
            vocals=ROOT / "outputs/tuki-zero/vocals.wav",
            gold=ROOT / "data/alignment_gold/tuki-zero.gold.tsv",
            notes="user Audacity gold, n=17",
        ),
        DatasetSpec(
            dataset_id="chidori/vad",
            song_id="chidori",
            aligned=ROOT / "outputs/chidori/aligned.vad.json",
            midi=ROOT / "outputs/chidori/melody_quantized.mid",
            vocals=ROOT / "outputs/chidori/vocals.wav",
            gold=ROOT / "data/alignment_gold/chidori.gold.tsv",
            notes="user Audacity gold",
        ),
    ]
    if include_candidates:
        candidates = {
            "haru-hikage": ["vad_cand12"],
            "tuki-zero": ["vad_cand12"],
            "chidori": ["vad_short"],
        }
        for song, labels in candidates.items():
            for label in labels:
                aligned = ROOT / f"outputs/{song}/aligned.{label}.json"
                if not aligned.exists():
                    continue
                specs.append(
                    DatasetSpec(
                        dataset_id=f"{song}/{label}",
                        song_id=song,
                        aligned=aligned,
                        midi=ROOT / f"outputs/{song}/melody_quantized.mid",
                        vocals=ROOT / f"outputs/{song}/vocals.wav",
                        gold=ROOT / f"data/alignment_gold/{song}.gold.tsv",
                        notes=f"candidate aligned.{label}.json artifact",
                    )
                )
        # Back-compat for historical haru/tuki naming in older outputs.
        for song in ["haru-hikage", "tuki-zero"]:
            aligned = ROOT / f"outputs/{song}/aligned.vad_cand12.json"
            if aligned.exists():
                dataset_id = f"{song}/vad_cand12"
                if not any(spec.dataset_id == dataset_id for spec in specs):
                    specs.append(
                        DatasetSpec(
                            dataset_id=dataset_id,
                            song_id=song,
                            aligned=aligned,
                            midi=ROOT / f"outputs/{song}/melody_quantized.mid",
                            vocals=ROOT / f"outputs/{song}/vocals.wav",
                            gold=ROOT / f"data/alignment_gold/{song}.gold.tsv",
                            notes="candidate RMS-VAD max_len=12 artifact",
                        )
                    )
    return specs


def load_manifest(path: Path) -> list[DatasetSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    specs: list[DatasetSpec] = []
    for item in data.get("datasets", []):
        specs.append(
            DatasetSpec(
                dataset_id=item["dataset_id"],
                song_id=item["song_id"],
                aligned=resolve_path(item["aligned"]),
                midi=resolve_path(item["midi"]),
                vocals=resolve_path(item["vocals"]),
                gold=resolve_path(item["gold"]),
                source=item.get("source", str(path)),
                notes=item.get("notes", ""),
            )
        )
    return specs


def midi_variants(profile: str) -> list[MidiVariant]:
    variants: list[MidiVariant] = [
        MidiVariant("mora_default"),
        MidiVariant("char_legacy", mode="char"),
    ]

    margins = [0.2, 0.3, 0.5, 0.6] if profile == "full" else [0.3, 0.5]
    variants += [MidiVariant(f"mora_margin_{m:.2f}", margin=m) for m in margins]

    delays = [0.0, 0.05, 0.10, 0.15, 0.20] if profile == "full" else [0.05, 0.10, 0.15]
    variants += [
        MidiVariant(f"global_gate_d{d:.2f}", first_mora_min_delay=d)
        for d in delays
    ]

    guarded_grid: list[tuple[float, float, float]]
    if profile == "full":
        guarded_grid = [
            (delay, prev_gap, lead_tol)
            for delay in [0.05, 0.10, 0.15]
            for prev_gap in [0.25, 0.50, 0.75]
            for lead_tol in [0.05, 0.08, 0.10, 0.15]
        ]
    else:
        guarded_grid = [
            (0.10, 0.50, 0.08),
            (0.10, 0.25, 0.08),
            (0.10, 0.75, 0.08),
            (0.05, 0.50, 0.08),
            (0.15, 0.50, 0.08),
            (0.10, 0.50, 0.05),
            (0.10, 0.50, 0.10),
        ]
    variants += [
        MidiVariant(
            f"guarded_gate_d{delay:.2f}_pg{prev_gap:.2f}_lt{lead_tol:.2f}",
            first_mora_min_delay=delay,
            first_mora_gate_prev_gap=prev_gap,
            first_mora_gate_lead_tolerance=lead_tol,
        )
        for delay, prev_gap, lead_tol in guarded_grid
    ]

    next_hint_guards = [0.12, 0.20, 0.30] if profile == "full" else [0.20]
    variants += [
        MidiVariant(
            f"next_hint_g{guard:.2f}_late1.00",
            next_line_hint_guard=guard,
            next_line_hint_min_start_delay=1.00,
        )
        for guard in next_hint_guards
    ]
    variants += [
        MidiVariant(
            f"global_gate_d{delay:.2f}_next_hint_g{guard:.2f}_late1.00",
            first_mora_min_delay=delay,
            next_line_hint_guard=guard,
            next_line_hint_min_start_delay=1.00,
        )
        for delay in ([0.05, 0.10] if profile == "full" else [0.05])
        for guard in next_hint_guards
    ]

    dp_grid: list[tuple[float, float, int]]
    if profile == "full":
        dp_grid = [
            (skip, extra, max_group)
            for skip in [0.12, 0.20, 0.30, 0.45]
            for extra in [0.02, 0.06, 0.10, 0.16]
            for max_group in [2, 3, 4]
        ]
    else:
        dp_grid = [
            (0.20, 0.06, 4),
            (0.30, 0.03, 4),
            (0.20, 0.10, 3),
            (0.45, 0.06, 4),
        ]
    variants += [
        MidiVariant(
            f"dp_s{skip:.2f}_x{extra:.2f}_m{max_group}",
            allocator="dp",
            dp_skip_penalty=skip,
            dp_extra_note_penalty=extra,
            dp_max_notes_per_mora=max_group,
        )
        for skip, extra, max_group in dp_grid
    ]
    variants += [
        MidiVariant(
            f"dp_guarded_d0.10_pg{prev_gap:.2f}_lt{lead_tol:.2f}_s{skip:.2f}_x{extra:.2f}_m{max_group}",
            allocator="dp",
            first_mora_min_delay=0.10,
            first_mora_gate_prev_gap=prev_gap,
            first_mora_gate_lead_tolerance=lead_tol,
            dp_skip_penalty=skip,
            dp_extra_note_penalty=extra,
            dp_max_notes_per_mora=max_group,
        )
        for prev_gap in ([0.50, 0.75] if profile == "full" else [0.50])
        for lead_tol in ([0.05, 0.08, 0.10] if profile == "full" else [0.08])
        for skip, extra, max_group in dp_grid
    ]

    variants.append(MidiVariant("absorb_trailing", absorb_trailing_notes=True))
    variants += [
        MidiVariant(
            f"guarded_absorb_d{delay:.2f}_pg{prev_gap:.2f}_lt{lead_tol:.2f}",
            first_mora_min_delay=delay,
            first_mora_gate_prev_gap=prev_gap,
            first_mora_gate_lead_tolerance=lead_tol,
            absorb_trailing_notes=True,
        )
        for delay, prev_gap, lead_tol in guarded_grid
    ]

    seen = set()
    out = []
    for variant in variants:
        if variant.name in seen:
            continue
        seen.add(variant.name)
        out.append(variant)
    return out


def repair_variants(profile: str) -> list[RepairVariant]:
    variants = [RepairVariant("repair_default")]
    if profile == "quick":
        variants += [
            RepairVariant("tail_top_22", tail_top_db=22.0),
            RepairVariant("tail_top_30", tail_top_db=30.0),
            RepairVariant("next_guard_015", next_guard=0.15),
            RepairVariant("next_guard_035", next_guard=0.35),
            RepairVariant("tail_gap_012", tail_gap=0.12),
            RepairVariant("tail_gap_024", tail_gap=0.24),
        ]
    else:
        for tail_top_db in [20.0, 22.0, 26.0, 30.0, 34.0]:
            for next_guard in [0.15, 0.20, 0.25, 0.35]:
                for tail_gap in [0.12, 0.18, 0.24]:
                    name = f"repair_t{tail_top_db:.0f}_g{next_guard:.2f}_gap{tail_gap:.2f}"
                    variants.append(
                        RepairVariant(
                            name,
                            tail_top_db=tail_top_db,
                            next_guard=next_guard,
                            tail_gap=tail_gap,
                        )
                    )
    seen = set()
    out = []
    for variant in variants:
        if variant.name in seen:
            continue
        seen.add(variant.name)
        out.append(variant)
    return out


def start_repair_variants(profile: str, include: bool) -> list[StartRepairVariant]:
    variants = [StartRepairVariant("start_none")]
    if not include:
        return variants

    # Conservative candidate from the chidori gold sweep: require a clearly late
    # MIDI start, use a strict RMS threshold, and only move partway toward the
    # RMS onset so weak breath/noise does not become the visible lyric onset.
    variants.append(
        StartRepairVariant(
            "start_blend35_t18_s128_move250",
            enabled=True,
            onset_top_db=18.0,
            onset_sustain=0.128,
            blend=0.35,
            min_move=0.25,
        )
    )
    variants.append(
        StartRepairVariant(
            "start_blend50_t18_s128_move080",
            enabled=True,
            onset_top_db=18.0,
            onset_sustain=0.128,
            blend=0.50,
            min_move=0.08,
        )
    )
    variants.append(
        StartRepairVariant(
            "start_blend35_t16_s096_late030_move120",
            enabled=True,
            min_late=0.30,
            onset_top_db=16.0,
            onset_sustain=0.096,
            blend=0.35,
            min_move=0.12,
        )
    )
    if profile == "full":
        variants += [
            StartRepairVariant(
                "start_blend35_t18_s128_move250_skipfirst",
                enabled=True,
                onset_top_db=18.0,
                onset_sustain=0.128,
                blend=0.35,
                min_move=0.25,
                skip_first_line=True,
            ),
            StartRepairVariant(
                "start_raw_t30_s096",
                enabled=True,
                onset_top_db=30.0,
                onset_sustain=0.096,
            ),
        ]
    return variants


def apply_midi_variant(lines: list[dict], notes: list[tuple[float, float, int]], variant: MidiVariant) -> None:
    if variant.mode == "char":
        midi_timing.apply_char_timing(lines, notes)
        return
    midi_timing.apply_mora_timing(
        lines,
        notes,
        margin=variant.margin,
        allocator=variant.allocator,
        first_mora_min_delay=variant.first_mora_min_delay,
        first_mora_gate_prev_gap=variant.first_mora_gate_prev_gap,
        first_mora_gate_lead_tolerance=variant.first_mora_gate_lead_tolerance,
        absorb_trailing_notes=variant.absorb_trailing_notes,
        next_line_hint_guard=variant.next_line_hint_guard,
        next_line_hint_min_start_delay=variant.next_line_hint_min_start_delay,
        dp_skip_penalty=variant.dp_skip_penalty,
        dp_extra_note_penalty=variant.dp_extra_note_penalty,
        dp_max_notes_per_mora=variant.dp_max_notes_per_mora,
    )


def apply_start_repair_variant(
    lines: list[dict],
    hint_lines: list[dict],
    rms_db,
    hop_s: float,
    variant: StartRepairVariant,
) -> list[tuple[int, float, float]]:
    if not variant.enabled:
        return []
    return line_start_repair.repair(
        lines,
        hint_lines,
        rms_db,
        hop_s,
        max_shift=variant.max_shift,
        min_late=variant.min_late,
        onset_top_db=variant.onset_top_db,
        onset_sustain=variant.onset_sustain,
        prev_guard=variant.prev_guard,
        blend=variant.blend,
        min_move=variant.min_move,
        skip_first_line=variant.skip_first_line,
    )


def composite_score(summary: dict[str, Any]) -> float:
    inv = summary["invariants"]
    invalid_penalty = (
        inv["zero_duration_sung_chars"]
        + inv["backward_sung_chars"]
        + inv["overlap_sung_chars"]
        + inv["short_sung_lines"]
    ) * 100.0
    return (
        invalid_penalty
        + summary["start"]["mae"]
        + summary["end"]["mae"]
        + 0.35 * summary["start"]["p90"]
        + 0.35 * summary["end"]["p90"]
    )


def result_from_summary(
    spec: DatasetSpec,
    midi_variant: MidiVariant,
    start_repair_variant: StartRepairVariant,
    repair_variant: RepairVariant,
    summary: dict[str, Any],
    start_repair_changes: int,
) -> AblationResult:
    inv = summary["invariants"]
    return AblationResult(
        dataset_id=spec.dataset_id,
        song_id=spec.song_id,
        source=spec.source,
        midi_variant=midi_variant.name,
        start_repair_variant=start_repair_variant.name,
        repair_variant=repair_variant.name,
        n=summary["n"],
        start_mae=summary["start"]["mae"],
        start_median=summary["start"]["median"],
        start_p90=summary["start"]["p90"],
        start_within_250ms=summary["start"]["within_250ms"],
        start_within_500ms=summary["start"]["within_500ms"],
        start_bias=summary["start"]["bias"],
        end_mae=summary["end"]["mae"],
        end_median=summary["end"]["median"],
        end_p90=summary["end"]["p90"],
        end_within_250ms=summary["end"]["within_250ms"],
        end_within_500ms=summary["end"]["within_500ms"],
        end_bias=summary["end"]["bias"],
        stress_n=summary["stress_n"],
        zero_duration_sung_chars=inv["zero_duration_sung_chars"],
        backward_sung_chars=inv["backward_sung_chars"],
        overlap_sung_chars=inv["overlap_sung_chars"],
        short_sung_lines=inv["short_sung_lines"],
        start_repair_changes=start_repair_changes,
        score=composite_score(summary),
    )


def validate_spec(spec: DatasetSpec) -> str | None:
    missing = [
        str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
        for path in [spec.aligned, spec.midi, spec.vocals, spec.gold]
        if not path.exists()
    ]
    if missing:
        return "missing: " + ", ".join(missing)
    return None


def run_one_dataset(
    spec: DatasetSpec,
    midi_grid: list[MidiVariant],
    start_grid: list[StartRepairVariant],
    repair_grid: list[RepairVariant],
    *,
    write_best: bool,
    out_dir: Path,
) -> tuple[list[AblationResult], dict[str, Any] | None, list[Any] | None, list[dict] | None]:
    reason = validate_spec(spec)
    if reason:
        return [
            AblationResult(
                dataset_id=spec.dataset_id,
                song_id=spec.song_id,
                source=spec.source,
                midi_variant="-",
                start_repair_variant="-",
                repair_variant="-",
                n=0,
                start_mae=math.nan,
                start_median=math.nan,
                start_p90=math.nan,
                start_within_250ms=math.nan,
                start_within_500ms=math.nan,
                start_bias=math.nan,
                end_mae=math.nan,
                end_median=math.nan,
                end_p90=math.nan,
                end_within_250ms=math.nan,
                end_within_500ms=math.nan,
                end_bias=math.nan,
                stress_n=0,
                zero_duration_sung_chars=0,
                backward_sung_chars=0,
                overlap_sung_chars=0,
                short_sung_lines=0,
                start_repair_changes=0,
                score=math.inf,
                status="skipped",
                detail=reason,
            )
        ], None, None, None

    base_lines = json.loads(spec.aligned.read_text(encoding="utf-8"))
    notes = midi_timing.extract_notes(spec.midi)
    gold = eval_alignment.read_gold(spec.gold)

    sr = line_end_repair.DEFAULTS["sr"]
    y = line_end_repair.load_audio_mono(spec.vocals, sr)
    rms_db = line_end_repair.frame_rms_db(
        y,
        line_end_repair.DEFAULTS["frame_length"],
        line_end_repair.DEFAULTS["hop_length"],
    )
    hop_s = line_end_repair.DEFAULTS["hop_length"] / sr
    duration = len(y) / sr

    results: list[AblationResult] = []
    best_score = math.inf
    best_summary: dict[str, Any] | None = None
    best_rows: list[Any] | None = None
    best_lines: list[dict] | None = None

    for midi_variant in midi_grid:
        midi_lines = copy.deepcopy(base_lines)
        apply_midi_variant(midi_lines, notes, midi_variant)
        for start_variant in start_grid:
            start_lines = copy.deepcopy(midi_lines)
            start_changes = apply_start_repair_variant(
                start_lines,
                base_lines,
                rms_db,
                hop_s,
                start_variant,
            )
            for repair_variant in repair_grid:
                lines = copy.deepcopy(start_lines)
                line_end_repair.repair(
                    lines,
                    rms_db,
                    hop_s,
                    duration,
                    tail_top_db=repair_variant.tail_top_db,
                    max_extend=repair_variant.max_extend,
                    next_guard=repair_variant.next_guard,
                    tail_gap=repair_variant.tail_gap,
                )
                summary, rows = eval_alignment.evaluate(
                    lines,
                    gold,
                    stress_error_threshold=0.5,
                    stress_gap_threshold=0.25,
                )
                result = result_from_summary(
                    spec,
                    midi_variant,
                    start_variant,
                    repair_variant,
                    summary,
                    len(start_changes),
                )
                results.append(result)
                if result.score < best_score:
                    best_score = result.score
                    best_summary = summary
                    best_rows = rows
                    best_lines = lines

    if write_best and best_summary is not None and best_rows is not None and best_lines is not None:
        ds_dir = out_dir / spec.dataset_id.replace("/", "__")
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "best.aligned.json").write_text(
            json.dumps(best_lines, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (ds_dir / "best.summary.json").write_text(
            json.dumps(best_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        eval_alignment.write_rows(ds_dir / "best.diff.tsv", best_rows)
        eval_alignment.write_rows(ds_dir / "best.stress.tsv", [r for r in best_rows if r.is_stress])

    return results, best_summary, best_rows, best_lines


def write_results(path: Path, results: list[AblationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(results[0]).keys()) if results else list(AblationResult.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for result in results:
            row = asdict(result)
            for key, value in row.items():
                if isinstance(value, float):
                    row[key] = "inf" if math.isinf(value) else f"{value:.6f}"
            writer.writerow(row)


def write_group_best(path: Path, results: list[AblationResult]) -> None:
    ok = [r for r in results if r.status == "ok"]
    best_by_dataset: dict[str, AblationResult] = {}
    for result in ok:
        cur = best_by_dataset.get(result.dataset_id)
        if cur is None or result.score < cur.score:
            best_by_dataset[result.dataset_id] = result
    ordered = sorted(best_by_dataset.values(), key=lambda r: (r.source, r.dataset_id))
    write_results(path, ordered)


def discover_literature_manifest_entries() -> list[DatasetSpec]:
    """Return manifest-like entries for known literature datasets if present.

    This does not download datasets. It just lets the report show clearly
    whether a literature benchmark has been staged locally in the expected
    converted sidecar schema.
    """
    base = ROOT / "data/literature_alignment"
    return [
        DatasetSpec(
            dataset_id="jamendo_mirex/converted",
            song_id="jamendo_mirex",
            aligned=base / "jamendo/aligned.json",
            midi=base / "jamendo/melody.mid",
            vocals=base / "jamendo/vocals.wav",
            gold=base / "jamendo/gold.tsv",
            source="literature: Jamendo/MIREX",
            notes="Expected converted subset from Jamendo lyrics alignment benchmark.",
        ),
        DatasetSpec(
            dataset_id="dali_v2/converted",
            song_id="dali_v2",
            aligned=base / "dali/aligned.json",
            midi=base / "dali/melody.mid",
            vocals=base / "dali/vocals.wav",
            gold=base / "dali/gold.tsv",
            source="literature: DALI v2",
            notes="Expected converted subset from DALI line/word/note annotations.",
        ),
        DatasetSpec(
            dataset_id="winterreise_rt/converted",
            song_id="winterreise_rt",
            aligned=base / "winterreise_rt/aligned.json",
            midi=base / "winterreise_rt/melody.mid",
            vocals=base / "winterreise_rt/vocals.wav",
            gold=base / "winterreise_rt/gold.tsv",
            source="literature: winterreise_rt",
            notes="Expected converted subset from Park et al. real-time lyrics alignment benchmark.",
        ),
        DatasetSpec(
            dataset_id="gtsinger/converted",
            song_id="gtsinger",
            aligned=base / "gtsinger/aligned.json",
            midi=base / "gtsinger/melody.mid",
            vocals=base / "gtsinger/vocals.wav",
            gold=base / "gtsinger/gold.tsv",
            source="literature: GTSinger",
            notes="Expected converted Japanese subset from TextGrid phoneme/word boundaries.",
        ),
    ]


@click.command()
@click.option("--out-dir", type=click.Path(file_okay=False), default="tmp/alignment-ablation")
@click.option("--profile", type=click.Choice(["quick", "full"]), default="quick", show_default=True)
@click.option("--manifest", "manifest_paths", type=click.Path(exists=True, dir_okay=False), multiple=True)
@click.option("--include-local/--no-include-local", default=True, show_default=True)
@click.option("--include-candidates/--no-include-candidates", default=True, show_default=True)
@click.option("--include-literature-placeholders/--no-include-literature-placeholders", default=True, show_default=True)
@click.option("--include-start-repair/--no-include-start-repair", default=True, show_default=True)
@click.option("--write-best-sidecars/--no-write-best-sidecars", default=True, show_default=True)
def main(
    out_dir: str,
    profile: str,
    manifest_paths: tuple[str, ...],
    include_local: bool,
    include_candidates: bool,
    include_literature_placeholders: bool,
    include_start_repair: bool,
    write_best_sidecars: bool,
) -> None:
    out_path = resolve_path(out_dir)
    specs: list[DatasetSpec] = []
    if include_local:
        specs.extend(default_local_specs(include_candidates=include_candidates))
    for manifest in manifest_paths:
        specs.extend(load_manifest(resolve_path(manifest)))
    if include_literature_placeholders:
        specs.extend(discover_literature_manifest_entries())

    midi_grid = midi_variants(profile)
    start_grid = start_repair_variants(profile, include=include_start_repair)
    repair_grid = repair_variants(profile)
    print(
        f"[ablation] datasets={len(specs)} midi_variants={len(midi_grid)} "
        f"start_variants={len(start_grid)} repair_variants={len(repair_grid)} "
        f"profile={profile}",
        flush=True,
    )

    all_results: list[AblationResult] = []
    for spec in specs:
        print(f"[ablation] {spec.dataset_id} ({spec.source})", flush=True)
        results, _summary, _rows, _lines = run_one_dataset(
            spec,
            midi_grid,
            start_grid,
            repair_grid,
            write_best=write_best_sidecars,
            out_dir=out_path,
        )
        all_results.extend(results)
        ok = [r for r in results if r.status == "ok"]
        if ok:
            best = min(ok, key=lambda r: r.score)
            print(
                f"  best {best.midi_variant} + {best.start_repair_variant} "
                f"+ {best.repair_variant}: "
                f"start_MAE={best.start_mae:.3f}s end_MAE={best.end_mae:.3f}s "
                f"score={best.score:.3f}",
                flush=True,
            )
        else:
            print(f"  skipped: {results[0].detail}", flush=True)

    write_results(out_path / "all_results.tsv", all_results)
    write_group_best(out_path / "best_by_dataset.tsv", all_results)
    (out_path / "run_config.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "datasets": [asdict(s) | {
                    "aligned": str(s.aligned),
                    "midi": str(s.midi),
                    "vocals": str(s.vocals),
                    "gold": str(s.gold),
                } for s in specs],
                "midi_variants": [asdict(v) for v in midi_grid],
                "start_repair_variants": [asdict(v) for v in start_grid],
                "repair_variants": [asdict(v) for v in repair_grid],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[ablation] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
