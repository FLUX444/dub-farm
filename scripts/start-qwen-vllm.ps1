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

if (Test-Path (Join-Path $projectRoot ".env")) {
    Write-Host "Loading settings from .env" -ForegroundColor DarkGray
}

$qwenModel = $env:QWEN_MODEL
if (-not $qwenModel) { $qwenModel = "Qwen/Qwen2.5-1.5B-Instruct" }
$gpuMem = $env:QWEN_GPU_MEMORY
if (-not $gpuMem) { $gpuMem = "0.70" }
$maxLen = $env:QWEN_MAX_MODEL_LEN
if (-not $maxLen) { $maxLen = "512" }

Write-Host "Starting Qwen vLLM (profile: qwen) from $projectRoot" -ForegroundColor Cyan
Write-Host "Model:            $qwenModel" -ForegroundColor DarkCyan
Write-Host "GPU memory util:  $gpuMem" -ForegroundColor DarkCyan
Write-Host "Max model len:    $maxLen" -ForegroundColor DarkCyan
Write-Host "API:              http://127.0.0.1:5001/v1" -ForegroundColor DarkCyan

& $docker.FullName compose --env-file .env -f docker-compose.vllm.yml up qwen-translate
