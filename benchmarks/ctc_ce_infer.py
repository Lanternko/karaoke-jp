#!/usr/bin/env python3
"""Batch inference driver for Wang & Jang's CE+CTC singing transcription
(TASLP 2022, github.com/york135/CTC_CE_for_AST) on OUR separated stems.

Same-front-end comparison with GAME: instead of their internal Spleeter
(do_svs=True), we rebuild the (vocal, mixture) channel pair from our
Mel-Band-RoFormer vocals + instrumental, matching the training-time feature
layout (6ch CQT: {voc, mix} x 3 power scales). For a cappella sources
(Kiritan) the mixture IS the vocal.

Output: MIR-ST500-format prediction JSON {song_id: [[on, off, pitch], ...]}.
Run inside the GAME venv (torch cu129 + librosa).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="CTC_CE_for_AST checkout")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocals", required=True,
                    help="dir of <id>.wav vocals OR <id>/vocals.wav layout")
    ap.add_argument("--acc-layout", default=None,
                    help="optional accompaniment path template with {id}, "
                    "e.g. sep/{id}/instrumental.wav; omit for a cappella")
    ap.add_argument("--out", required=True)
    ap.add_argument("--onset-thres", type=float, default=0.26)
    ap.add_argument("--offset-thres", type=float, default=0.7)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    import librosa
    from predictor import NoteLevelAST

    sys.path.insert(0, str(repo / "data_utils"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "get_feature", repo / "data_utils" / "get_feature.py")
    feat_mod = importlib.util.module_from_spec(spec)
    sys.modules["get_feature"] = feat_mod
    spec.loader.exec_module(feat_mod)
    extractor = feat_mod.CQT_feature_extractor()

    predictor = NoteLevelAST(
        network_file=str(repo / "net" / "onset_and_pitch_0901.py"),
        network_class_name="Split_onset_pitch",
        device=args.device,
        model_path=args.ckpt,
    )

    vocals_dir = Path(args.vocals)
    wavs = sorted(vocals_dir.glob("*.wav"))
    if not wavs:
        wavs = sorted(vocals_dir.glob("*/vocals.wav"))

    class OneSong:
        """Minimal stand-in for SeqDataset over a prebuilt feature tensor."""
        def __init__(self, feats, song_id):
            self.chunks = []
            self.song_id = song_id
            for i in range(0, feats.shape[1], 20000):
                self.chunks.append(feats[:, i:min(feats.shape[1], i + 20000), :])
        def __getitem__(self, idx):
            return (self.chunks[idx], self.song_id)
        def __len__(self):
            return len(self.chunks)

    def load44k(p):
        y, sr = librosa.core.load(p, sr=None, mono=True)
        if sr != 44100:
            y = librosa.core.resample(y=y, orig_sr=sr, target_sr=44100)
        return y

    results = {}
    for wav in wavs:
        sid = wav.stem if wav.name != "vocals.wav" else wav.parent.name
        y_voc = load44k(wav)
        if args.acc_layout:
            y_acc = load44k(args.acc_layout.format(id=sid))
            n = min(len(y_voc), len(y_acc))
            y_voc, y_acc = y_voc[:n], y_acc[:n]
        else:
            y_acc = np.zeros_like(y_voc)
        # mirror the do_svs=True normalization: joint peak of the mixture
        max_mag = np.max(np.abs(y_voc + y_acc))
        y_voc = y_voc / (max_mag + 1e-4)
        y_mix = y_voc + y_acc / (max_mag + 1e-4)
        feats = extractor.get_all_feature_from_array(y_voc, y_mix).permute(1, 0, 2)

        ds = OneSong(feats, sid)
        # show_tqdm=True is load-bearing: the upstream predict() only runs
        # its forward loop inside the `if show_tqdm == True:` branch
        results, _ = predictor.predict(
            ds, results_dict=results, show_tqdm=True,
            onset_thres=args.onset_thres, offset_thres=args.offset_thres)
        print(f"[ctc-ce] {sid}: {len(results[sid])} notes", flush=True)

    Path(args.out).write_text(json.dumps(results))
    total = sum(len(v) for v in results.values())
    print(f"[ctc-ce] wrote {args.out}: {len(results)} songs, {total} notes")


if __name__ == "__main__":
    main()
