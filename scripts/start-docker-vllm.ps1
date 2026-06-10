$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) {
    $dockerPath = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
    if (Test-Path $dockerPath) {
        $docker = Get-Item $dockerPath
    }
}

if (-not $docker) {
    throw "docker.exe not found. Install Docker Desktop or add docker.exe to PATH."
}

Write-Host "Starting both vLLM containers (profile: all) from $projectRoot" -ForegroundColor Yellow
Write-Host "WARNING: Whisper + Qwen together may exceed 8 GB VRAM." -ForegroundColor Yellow
Write-Host "Prefer scripts/start-whisper-vllm.ps1 and scripts/start-qwen-vllm.ps1 separately." -ForegroundColor Yellow
& $docker.FullName compose --env-file .env -f docker-compose.vllm.yml --profile all up
