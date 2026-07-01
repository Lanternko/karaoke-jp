import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import genre_router as g  # noqa: E402
import numpy as np  # noqa: E402


def test_classify_modes():
    assert g.classify(0.29, 2.8) == "jpop"   # repo J-pop baseline (0.22-0.31)
    assert g.classify(0.45, 2.5) == "enka"   # kobushi-heavy
    assert g.classify(0.12, 6.0) == "rap"    # fast + low pitch
    assert g.classify(0.50, 6.0) == "enka"   # high portamento wins over rate


def test_jpop_mode_is_noop():
    assert g.MODE_OVERRIDES["jpop"] == {}    # default = current canonical behaviour


def test_port_density_detects_glide():
    n = 500
    assert g.port_density(np.full(n, 440.0)) < 0.05   # steady tone -> ~0
    # fast glide (~48 cents/frame, well past the 20 c/frame threshold the
    # real-song baseline 0.22-0.31 is calibrated against)
    sweep = 440.0 * 2 ** (0.25 * np.arange(n) / 12)  # 25 cents/frame
    assert g.port_density(sweep) > 0.5


def test_syllable_rate():
    aligned = [{"start": 0.0, "end": 2.0, "tokens": [{"chars": [
        {"char": "あ", "start": 0.0, "end": 0.5},
        {"char": "い", "start": 0.5, "end": 1.0}]}]}]
    assert abs(g.syllable_rate(aligned) - 1.0) < 1e-6  # 2 chars / 2 s
