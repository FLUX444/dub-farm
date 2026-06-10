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

$whisperModel = $env:WHISPER_MODEL
if (-not $whisperModel) { $whisperModel = "openai/whisper-large-v3" }

Write-Host "Starting Whisper vLLM (profile: whisper) from $projectRoot" -ForegroundColor Cyan
Write-Host "Model: $whisperModel" -ForegroundColor DarkCyan
Write-Host "API:   http://127.0.0.1:8000/v1" -ForegroundColor DarkCyan

& $docker.FullName compose --env-file .env -f docker-compose.vllm.yml up whisper-vllm
