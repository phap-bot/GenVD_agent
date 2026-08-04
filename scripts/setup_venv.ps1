param(
    [switch]$SkipInstall,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvDir = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$HfExe = Join-Path $VenvDir "Scripts\hf.exe"
$TokenizerDir = Join-Path $ProjectRoot "models\MOSS-Audio-Tokenizer-Nano"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

Set-Location $ProjectRoot

if (-not (Test-Path $PythonExe)) {
    Write-Step "Creating Python virtual environment"
    py -3.12 -m venv $VenvDir
}

if (-not $SkipInstall) {
    Write-Step "Installing Python dependencies"
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
}

if ($SkipModels) {
    Write-Step "Skipping Hugging Face model downloads"
    exit 0
}

Write-Step "Checking Hugging Face authentication"
& $HfExe auth whoami

Write-Step "Downloading MOSS audio tokenizer without Windows symlink cache"
New-Item -ItemType Directory -Force -Path $TokenizerDir | Out-Null
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
& $HfExe download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano `
    --local-dir $TokenizerDir `
    --max-workers 1

Write-Host ""
Write-Host "Public tokenizer is ready at: $TokenizerDir" -ForegroundColor Green
Write-Host ""
Write-Host "VieNeu-TTS model note:" -ForegroundColor Yellow
Write-Host "  pnnbao-ump/VieNeu-TTS-v3-Turbo-Fixed-vi-emotion returned 401 while this machine is not logged in."
Write-Host "  If that repo is private or gated, run: .\.venv\Scripts\hf.exe auth login"
Write-Host "  Then retry: .\.venv\Scripts\hf.exe download pnnbao-ump/VieNeu-TTS-v3-Turbo-Fixed-vi-emotion --local-dir .\models\VieNeu-TTS-v3-Turbo-Fixed-vi-emotion --max-workers 1"
