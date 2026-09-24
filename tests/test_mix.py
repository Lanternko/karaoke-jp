from __future__ import annotations

import inspect

from karaoke_jp.mix import mix_vocals


def test_mix_vocals_default_uses_thirty_percent_guide_vocal() -> None:
    params = inspect.signature(mix_vocals).parameters

    assert params["vocal_ratio"].default == 0.30
