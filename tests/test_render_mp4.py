from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_mp4.py"
SPEC = importlib.util.spec_from_file_location("render_mp4", SCRIPT)
render_mp4 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(render_mp4)


class FakeApp:
    def __init__(self) -> None:
        self.current_time = 0.0
        self.calls = 0
        self.lyrics_types = [
            "background_main_lyric",
            "front_main_lyric",
            "background_ruby",
            "front_ruby",
        ]
        self.lyrics = [
            {
                typ: {"start": 10.0, "end": 12.0}
                for typ in self.lyrics_types
            }
        ]

    def draw_notes(self) -> str:
        self.calls += 1
        return "drawn"


def test_hide_notes_without_visible_lyrics_uses_lyric_visibility() -> None:
    app = FakeApp()
    render_mp4._hide_notes_without_visible_lyrics(app)

    app.current_time = 9.9
    assert app.draw_notes() is None
    assert app.calls == 0

    app.current_time = 10.0
    assert app.draw_notes() == "drawn"
    assert app.calls == 1

    app.current_time = 12.0
    assert app.draw_notes() is None
    assert app.calls == 1
