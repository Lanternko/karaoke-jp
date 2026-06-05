"""Local Gradio GUI for the karaoke-jp Snakemake pipeline."""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
SONGS_DIR = REPO_ROOT / "songs"
OUTPUTS_DIR = REPO_ROOT / "outputs"
MAIN_VENV_BIN = Path.home() / "venvs" / "karaoke-jp" / "bin"
BLACK_BG_LABEL = "純黑背景"
ORIGINAL_BG_LABEL = "保留原始影片"


def _import_gradio():
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - exercised only at runtime
        raise RuntimeError(
            "找不到 gradio。請先執行 "
            "~/venvs/karaoke-jp/bin/pip install -e '.[batch,gui]'"
        ) from exc
    return gr


def _tool_path(tool_name: str) -> str:
    candidate = MAIN_VENV_BIN / tool_name
    if candidate.exists():
        return str(candidate)

    resolved = shutil.which(tool_name)
    if resolved:
        return resolved

    raise FileNotFoundError(f"找不到執行檔：{tool_name}")


def _youtube_video_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    if host.endswith("youtu.be"):
        video_id = parsed.path.strip("/").split("/", 1)[0]
        return video_id or None
    if "youtube.com" in host:
        values = parse_qs(parsed.query)
        if values.get("v"):
            return values["v"][0]
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed"}:
            return path_parts[1]
    return None


def _slug_fragment(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "song"


def _build_song_id(youtube_url: str, uploaded_video: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if youtube_url.strip():
        base = _youtube_video_id(youtube_url) or "youtube"
    elif uploaded_video:
        base = Path(uploaded_video).stem
    else:
        base = "song"
    return f"gui-{timestamp}-{_slug_fragment(base)}"


def _normalize_lyrics(lyrics_text: str) -> str:
    normalized = lyrics_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    joined = "\n".join(lines).strip()
    if not joined:
        raise ValueError("請貼上 lyrics 純文字。")
    return joined + "\n"


def _run_checked(cmd: list[str], *, cwd: Path = REPO_ROOT) -> str:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"指令失敗：{shlex.join(cmd)}\n{message}")
    return "\n".join(
        chunk for chunk in (completed.stdout.strip(), completed.stderr.strip()) if chunk
    )


def _extract_audio_to_wav(video_path: Path, wav_path: Path) -> None:
    cmd = [
        _tool_path("ffmpeg"),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        str(wav_path),
    ]
    _run_checked(cmd)


def _create_black_background(video_path: Path) -> None:
    cmd = [
        _tool_path("ffmpeg"),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=1920x1080:r=30",
        "-t",
        "5",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    _run_checked(cmd)


def _tail_log(lines: list[str], *, keep: int = 160) -> str:
    return "\n".join(lines[-keep:])


def _prepare_inputs(
    *,
    youtube_url: str,
    uploaded_video: str | None,
    lyrics_text: str,
    background_mode: str,
    logs: list[str],
) -> str:
    song_id = _build_song_id(youtube_url, uploaded_video)
    song_dir = SONGS_DIR / song_id
    song_dir.mkdir(parents=True, exist_ok=True)

    lyrics_path = song_dir / "lyrics.txt"
    lyrics_path.write_text(_normalize_lyrics(lyrics_text), encoding="utf-8")
    logs.append(f"[prep] lyrics -> {lyrics_path}")

    if youtube_url:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "download_song.py"),
            youtube_url,
            "-o",
            str(song_dir),
        ]
        if background_mode == BLACK_BG_LABEL:
            cmd.append("--no-video")
        logs.append(f"[prep] $ {shlex.join(cmd)}")
        output = _run_checked(cmd)
        if output:
            logs.extend(output.splitlines())
        if background_mode == BLACK_BG_LABEL:
            bg_path = song_dir / "background.mp4"
            _create_black_background(bg_path)
            logs.append(f"[prep] blank background -> {bg_path}")
        return song_id

    if not uploaded_video:
        raise ValueError("請擇一提供 YouTube 連結或 mp4 檔案。")

    upload_path = Path(uploaded_video).resolve()
    if upload_path.suffix.lower() != ".mp4":
        raise ValueError("目前 GUI 只接受 mp4 檔案。")

    staged_video = (
        song_dir / "background.mp4"
        if background_mode == ORIGINAL_BG_LABEL
        else song_dir / "source_video.mp4"
    )
    shutil.copy2(upload_path, staged_video)
    logs.append(f"[prep] video -> {staged_video}")

    source_wav = song_dir / "source.wav"
    _extract_audio_to_wav(staged_video, source_wav)
    logs.append(f"[prep] audio -> {source_wav}")

    if background_mode == BLACK_BG_LABEL:
        bg_path = song_dir / "background.mp4"
        _create_black_background(bg_path)
        logs.append(f"[prep] blank background -> {bg_path}")

    return song_id


def _run_pipeline(
    youtube_url: str,
    uploaded_video: str | None,
    lyrics_text: str,
    vocal_ratio_percent: float,
    background_mode: str,
):
    logs: list[str] = []

    def _snapshot(status: str, video: str | None = None, download: str | None = None):
        return status, _tail_log(logs), video, download

    youtube_url = youtube_url.strip()
    if bool(youtube_url) == bool(uploaded_video):
        message = "請擇一提供 YouTube 連結或 mp4 檔案。"
        yield _snapshot(message)
        return

    try:
        vocal_ratio = float(vocal_ratio_percent) / 100.0
        if not (0.0 <= vocal_ratio <= 1.0):
            raise ValueError("人聲比例必須介於 0 到 100。")

        logs.append("[start] 準備輸入檔案")
        yield _snapshot("正在準備輸入檔案…")

        song_id = _prepare_inputs(
            youtube_url=youtube_url,
            uploaded_video=uploaded_video,
            lyrics_text=lyrics_text,
            background_mode=background_mode,
            logs=logs,
        )
        output_mp4 = OUTPUTS_DIR / song_id / "karaoke.mp4"
        cmd = [
            _tool_path("snakemake"),
            "--rerun-triggers",
            "mtime",
            "-j",
            "1",
            str(output_mp4),
        ]
        env = os.environ.copy()
        env["VOCAL_RATIO"] = f"{vocal_ratio:.4f}"
        logs.append(f"[run] $ {shlex.join(cmd)}")
        logs.append(f"[run] VOCAL_RATIO={env['VOCAL_RATIO']}")
        yield _snapshot(f"正在產生 MP4：{song_id}")

        process = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None

        for raw_line in process.stdout:
            logs.append(raw_line.rstrip())
            yield _snapshot(f"正在產生 MP4：{song_id}")

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError("Snakemake 執行失敗，請查看處理紀錄。")
        if not output_mp4.exists():
            raise FileNotFoundError(f"找不到輸出檔：{output_mp4}")

        logs.append(f"[done] {output_mp4}")
        yield _snapshot(f"完成：{song_id}", str(output_mp4), str(output_mp4))
    except Exception as exc:
        logs.append(f"[error] {exc}")
        yield _snapshot(f"失敗：{exc}")


def build_app():
    gr = _import_gradio()

    with gr.Blocks(title="karaoke-jp GUI") as demo:
        gr.Markdown(
            "## karaoke-jp GUI\n"
            "貼上 YouTube 連結或上傳 mp4，補上 lyrics 純文字後，直接產生 `karaoke.mp4`。"
        )

        with gr.Row():
            youtube_url = gr.Textbox(
                label="YouTube 連結",
                placeholder="https://www.youtube.com/watch?v=...",
            )
            uploaded_video = gr.File(
                label="或上傳 mp4",
                file_types=[".mp4"],
                file_count="single",
                type="filepath",
            )

        lyrics_text = gr.Textbox(
            label="Lyrics 純文字",
            placeholder="每行一句，直接貼上即可。",
            lines=12,
        )

        with gr.Row():
            vocal_ratio = gr.Slider(
                label="人聲比例 (%)",
                minimum=0,
                maximum=100,
                value=30,
                step=5,
            )
            background_mode = gr.Radio(
                label="背景模式",
                choices=[ORIGINAL_BG_LABEL, BLACK_BG_LABEL],
                value=ORIGINAL_BG_LABEL,
            )

        submit = gr.Button("開始產生 MP4", variant="primary")
        status = gr.Textbox(label="狀態", interactive=False)
        logs = gr.Textbox(label="處理紀錄", lines=16, max_lines=24, interactive=False)
        preview = gr.Video(label="輸出預覽")
        download = gr.File(label="下載 MP4")

        submit.click(
            fn=_run_pipeline,
            inputs=[
                youtube_url,
                uploaded_video,
                lyrics_text,
                vocal_ratio,
                background_mode,
            ],
            outputs=[status, logs, preview, download],
            show_progress="minimal",
        )

    demo.queue(default_concurrency_limit=1)
    return demo


def launch_app(*, host: str = "127.0.0.1", port: int = 7860) -> None:
    demo = build_app()
    demo.launch(
        server_name=host,
        server_port=port,
        inbrowser=False,
        show_api=False,
        allowed_paths=[str(REPO_ROOT)],
    )
