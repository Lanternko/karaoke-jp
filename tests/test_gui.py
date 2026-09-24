from __future__ import annotations

import pytest

from karaoke_jp.gui import _build_song_id, _normalize_lyrics, _youtube_video_id


def test_youtube_video_id_supports_watch_urls() -> None:
    assert (
        _youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        == "dQw4w9WgXcQ"
    )


def test_youtube_video_id_supports_short_urls() -> None:
    assert _youtube_video_id("https://youtu.be/abc123xyz99?t=12") == "abc123xyz99"


def test_build_song_id_uses_uploaded_filename_when_no_url() -> None:
    song_id = _build_song_id("", "/tmp/My Test Clip.mp4")
    assert song_id.startswith("gui-")
    assert song_id.endswith("-my-test-clip")


def test_normalize_lyrics_trims_edges_and_keeps_body() -> None:
    assert _normalize_lyrics("\n  first line  \r\nsecond line\r\n") == "first line\nsecond line\n"


def test_normalize_lyrics_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        _normalize_lyrics(" \n\r\n ")
