#!/usr/bin/env python3
"""Seeded launcher for GAME's ``infer.py`` (runs in the GAME venv).

Why this exists: GAME's segmenter is a **D3PM discrete diffusion** model, and
its sampling loop is stochastic at INFERENCE time, not just in training.
``inference/me_infer.py:forward_segmenter_main`` iterates over the 8 sampling
timesteps, and every step calls ``modules/d3pm.remove_mutable_boundaries()``,
which draws ``torch.rand_like(boundaries)`` to randomly drop boundaries before
re-predicting them. Nothing in ``infer.py`` / ``inference/api.py`` seeds torch,
so back-to-back runs on identical audio land on different note segmentations.

Dropout is not the culprit (``model.eval()`` is set and attention dropout is
gated on ``self.training``), and the DataLoader is ``shuffle=False`` with
``num_workers=0``, so seeding the global torch RNG in the inference process is
enough to pin the sampler.

Usage (argv after ``--`` is passed to infer.py verbatim)::

    python scripts/game_infer_seeded.py --game-dir <GAME> --seed 0 -- extract ...
"""
from __future__ import annotations

import argparse
import os
import random
import runpy
import sys


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--game-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--deterministic-algorithms", action="store_true",
                    help="Also demand bitwise-deterministic CUDA kernels "
                         "(torch.use_deterministic_algorithms). Off by "
                         "default: not needed on sm_120 for this model, and "
                         "some ops have no deterministic implementation.")
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    # cuBLAS needs this set BEFORE the first CUDA context to be able to offer a
    # deterministic GEMM workspace at all.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    sys.path.insert(0, args.game_dir)
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)

    # runpy.run_path overwrites argv[0] with the script path; click only uses
    # argv[0] for help text, so infer.py sees the same options either way.
    sys.argv = ["infer.py", *rest]
    runpy.run_path(os.path.join(args.game_dir, "infer.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
