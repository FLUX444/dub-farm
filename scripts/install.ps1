# dub-farm full setup for Windows
# Usage: .\scripts\install.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== dub-farm setup ===" -ForegroundColor Cyan

# Prefer Python 3.12 for best ML compatibility
$py = $null
foreach ($ver in @("3.12", "3.13", "3.11", "3.14")) {
    try {
        & py "-$ver" --version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            $py = "py -$ver"
            break
        }
    } catch {}
}
if (-not $py) { $py = "py" }

Write-Host "Using: $py" -ForegroundColor Green

Invoke-Expression "$py -m pip install --upgrade pip"
Invoke-Expression "$py -m pip install -e `".[ml]`""

Write-Host "`n=== Downloading models (~6-8 GB) ===" -ForegroundColor Cyan
Invoke-Expression "$py -m dub_farm.cli download-models --weights-dir weights"

Write-Host "`n=== Ollama (translation) ===" -ForegroundColor Cyan
Write-Host "Install Ollama from https://ollama.com then run:"
Write-Host "  ollama pull qwen2.5:14b"
Write-Host "  ollama serve"

Write-Host "`n=== Docker vLLM (Windows) ===" -ForegroundColor Cyan
Write-Host "Enable BIOS virtualization + WSL2, put Qwen translation weights in llm-models, then run:"
Write-Host "  .\scripts\start-docker-vllm.ps1"

Write-Host "`n=== FFmpeg ===" -ForegroundColor Cyan
Write-Host "Install FFmpeg and add to PATH: https://ffmpeg.org/download.html"

Write-Host "`n=== Ready ===" -ForegroundColor Green
Write-Host "$py -m dub_farm.cli run video.mp4 --target-lang ru"
