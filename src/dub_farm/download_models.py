from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from dub_farm.config import AppConfig
from dub_farm.utils.proxy import apply_proxy_env

logger = logging.getLogger(__name__)
console = Console()

WEIGHTS_LAYOUT: dict[str, list[dict[str, str]]] = {
    "tts": [
        {"repo": "Qwen/Qwen3-TTS-Tokenizer-12Hz", "desc": "Qwen3-TTS tokenizer"},
        {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "desc": "Qwen3-TTS custom voices"},
        {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "desc": "Qwen3-TTS voice clone"},
    ],
    "whisper": [
        {"repo": "Systran/faster-whisper-large-v3", "desc": "faster-whisper large-v3"},
    ],
    "vlm": [
        {"repo": "Qwen/Qwen2-VL-2B-Instruct", "desc": "Qwen2-VL scene analysis"},
    ],
}


def download_all(
    weights_dir: Path,
    *,
    include_vlm: bool = False,
    include_whisper: bool = True,
    tts_only: bool = False,
    config: AppConfig | None = None,
) -> None:
    """Download models into weights/{tts,whisper,vlm}/."""
    if config is not None:
        apply_proxy_env(config)

    weights_dir = weights_dir.resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)

    groups = ["tts"]
    if include_whisper and not tts_only:
        groups.append("whisper")
    if include_vlm and not tts_only:
        groups.append("vlm")

    entries: list[tuple[str, str, Path]] = []
    for group in groups:
        target = weights_dir / group
        for item in WEIGHTS_LAYOUT[group]:
            local_dir = target / item["repo"].replace("/", "--")
            entries.append((item["repo"], item["desc"], local_dir))

    console.print(f"[bold]Weights directory:[/] {weights_dir}")
    console.print(f"[bold]Downloading {len(entries)} model(s)...[/]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for repo, desc, local_dir in entries:
            task = progress.add_task(f"{desc}...", total=None)
            _hf_download(repo, local_dir)
            progress.update(task, description=f"[green]OK[/] {repo}")

    _write_layout_readme(weights_dir)
    console.print("[bold green]Done.[/]")
    _print_next_steps()


def _hf_download(repo: str, local_dir: Path) -> None:
    if _has_weights(local_dir):
        logger.info("Already downloaded: %s", local_dir)
        return

    local_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"  [cyan]>>[/] {repo} -> {local_dir}")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo, local_dir=str(local_dir))
        return
    except ImportError:
        pass

    subprocess.run(
        [
            sys.executable,
            "-m",
            "huggingface_hub.cli.huggingface_cli",
            "download",
            repo,
            "--local-dir",
            str(local_dir),
        ],
        check=True,
    )


def _has_weights(path: Path) -> bool:
    if not path.exists():
        return False
    for pattern in ("*.safetensors", "*.bin", "*.pt"):
        if any(path.rglob(pattern)):
            return True
    return False


def _write_layout_readme(weights_dir: Path) -> None:
    readme = weights_dir / "README.md"
    readme.write_text(
        """# dub-farm model weights

```
weights/
├── tts/          → Qwen3-TTS (озвучка, dub-farm)
├── whisper/      → faster-whisper (транскрибация)
└── vlm/          → Qwen2-VL (анализ сцен, опционально)
```

Translation via Ollama separately (ollama pull qwen2.5:7b).
Do not point vLLM at weights/tts - these are TTS models only.
""",
        encoding="utf-8",
    )


def _print_next_steps() -> None:
    console.print(
        "\n[bold]Next steps:[/]\n"
        "  ollama pull qwen2.5:7b\n"
        "  ollama serve\n"
        "  py -m dub_farm.cli run video.mp4 --target-lang ru"
    )
