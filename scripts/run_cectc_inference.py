"""CTC+CE singing transcription inference wrapper.

Bypasses the upstream Spleeter dependency by feeding our pre-separated
``vocals.wav`` and ``instrumental.wav`` (Demucs Kim FT2 Bleedless) directly
into ``get_all_feature``. Writes a MIDI file with onset/offset/pitch notes.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import mido
import torch
import yaml

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CECTC_ROOT = PROJECT_ROOT / "third_party" / "CTC_CE_for_AST"


def _import_from_path(mod_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _notes_to_midi(notes: list[list[float]], midi_path: Path, *, tempo: float = 120.0) -> None:
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    new_tempo = mido.bpm2tempo(tempo)
    track.append(mido.MetaMessage("set_tempo", tempo=new_tempo))
    track.append(mido.Message("program_change", program=0, time=0))

    cur_total_tick = 0
    for note in notes:
        if note[2] == 0:
            continue
        pitch = int(round(note[2]))
        ticks_since_previous_onset = int(
            mido.second2tick(note[0], ticks_per_beat=480, tempo=new_tempo)
        )
        ticks_current_note = int(
            mido.second2tick(note[1] - 0.0001, ticks_per_beat=480, tempo=new_tempo)
        )
        note_on_length = ticks_since_previous_onset - cur_total_tick
        note_off_length = ticks_current_note - note_on_length - cur_total_tick
        track.append(mido.Message("note_on", note=pitch, velocity=100, time=max(0, note_on_length)))
        track.append(mido.Message("note_off", note=pitch, velocity=100, time=max(1, note_off_length)))
        cur_total_tick += max(0, note_on_length) + max(1, note_off_length)

    midi_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(midi_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocals", required=True, type=Path)
    parser.add_argument("--instrumental", required=True, type=Path)
    parser.add_argument("--midi", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--yaml", dest="yaml_path", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames-chunk", type=int, default=20000)
    args = parser.parse_args()

    sys.path.insert(0, str(CECTC_ROOT))

    with open(args.yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    feature_module = _import_from_path(
        "cectc_get_feature", CECTC_ROOT / "data_utils" / "get_feature.py"
    )
    Extractor = getattr(feature_module, cfg["feature_extractor_class_name"])
    extractor = Extractor()

    cqt_data = extractor.get_all_feature(str(args.vocals), str(args.instrumental))
    print(f"[cectc] feature shape (channel, time, freq) = {tuple(cqt_data.shape)}")

    net_module = _import_from_path(
        "cectc_net", CECTC_ROOT / "net" / "onset_and_pitch_0901.py"
    )
    Model = getattr(net_module, cfg["network_class_name"])

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    print(f"[cectc] device = {device}")

    model = Model().to(device)
    state = torch.load(str(args.model), map_location=device)
    model.load_state_dict(state)
    model.eval()

    predictor_module = _import_from_path(
        "cectc_predictor", CECTC_ROOT / "predictor.py"
    )
    NoteLevelAST = predictor_module.NoteLevelAST

    onset_thres = float(cfg["onset_thres"])
    offset_thres = float(cfg["offset_thres"])

    chunk = args.num_frames_chunk
    frame_num = cqt_data.shape[1]
    print(f"[cectc] frames = {frame_num}, chunks = {(frame_num + chunk - 1) // chunk}")

    frame_info_all = []
    with torch.no_grad():
        for start in range(0, frame_num, chunk):
            end = min(frame_num, start + chunk)
            chunk_feat = cqt_data[:, start:end, :].unsqueeze(0).to(device)
            _, on_off_logits_sm, pitch_octave_logits, pitch_class_logits, _ = model(chunk_feat)
            onset_probs = on_off_logits_sm[:, :, 1].cpu().numpy()[0]
            offset_probs = on_off_logits_sm[:, :, 2].cpu().numpy()[0]
            poct = pitch_octave_logits.cpu()[0]
            pcls = pitch_class_logits.cpu()[0]
            for i in range(onset_probs.shape[0]):
                frame_info_all.append(
                    (
                        float(onset_probs[i]),
                        float(offset_probs[i]),
                        int(torch.argmax(poct[i]).item()),
                        int(torch.argmax(pcls[i]).item()),
                    )
                )

    # _parse_frame_info reads no instance state; bind via the class to skip __init__'s model load.
    notes = NoteLevelAST._parse_frame_info(
        None, frame_info_all, onset_thres=onset_thres, offset_thres=offset_thres
    )
    print(f"[cectc] notes = {len(notes)}")

    _notes_to_midi(notes, args.midi)
    print(f"[cectc] wrote {args.midi}")


if __name__ == "__main__":
    main()
