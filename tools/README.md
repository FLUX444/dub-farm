# Local FFmpeg (not in Git)

GitHub не принимает бинарники FFmpeg (>50–100 MB). Скачай локально:

1. https://www.gyan.dev/ffmpeg/builds/ — **ffmpeg-release-essentials.zip**
2. Распакуй в `tools/`, чтобы получилось:

```text
tools/ffmpeg-8.1.1-essentials_build/bin/ffmpeg.exe
tools/ffmpeg-8.1.1-essentials_build/bin/ffprobe.exe
```

Или установи FFmpeg в системный `PATH`, либо задай переменные:

```powershell
$env:DUB_FARM_FFMPEG = "C:\path\to\ffmpeg.exe"
$env:DUB_FARM_FFPROBE = "C:\path\to\ffprobe.exe"
```
