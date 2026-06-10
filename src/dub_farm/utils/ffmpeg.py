from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def find_ffmpeg() -> str:
    path = _find_tool("ffmpeg", "DUB_FARM_FFMPEG")
    if not path:
        raise FFmpegError(
            "FFmpeg not found. Install FFmpeg, add it to PATH, set DUB_FARM_FFMPEG, "
            "or place a portable build under tools/ffmpeg*/bin."
        )
    return path


def run_ffmpeg(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, *args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        raise FFmpegError(
            f"FFmpeg failed (exit {result.returncode}):\n"
            f"{' '.join(cmd)}\n"
            f"{result.stderr.strip()}"
        )
    return result


def get_duration(path: Path) -> float:
    ffprobe = _find_tool("ffprobe", "DUB_FARM_FFPROBE")
    if not ffprobe:
        raise FFmpegError(
            "ffprobe not found. Install FFmpeg, add it to PATH, set DUB_FARM_FFPROBE, "
            "or place a portable build under tools/ffmpeg*/bin."
        )
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _find_tool(name: str, env_var: str) -> str | None:
    env_path = os.environ.get(env_var)
    if env_path and Path(env_path).exists():
        return env_path

    path = shutil.which(name)
    if path:
        return path

    exe_name = f"{name}.exe" if os.name == "nt" else name
    project_root = Path(__file__).resolve().parents[3]
    for candidate in (project_root / "tools").glob("ffmpeg*/bin/" + exe_name):
        if candidate.exists():
            return str(candidate)
    return None
