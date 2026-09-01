param(
    [switch]$SkipInstall,
    [switch]$SkipModels,
    [switch]$CpuOnly,
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [string]$CpuTorchIndexUrl = "https://download.pytorch.org/whl/cpu"
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
    Write-Step "Preparing Python package installer"
    & $PythonExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

    $hasNvidiaGpu = (-not $CpuOnly) -and $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
    if ($hasNvidiaGpu) {
        $cudaCheck = & $PythonExe -c "import torch, torchaudio, torchvision; ok = torch.__version__.startswith('2.8.0') and torchaudio.__version__.startswith('2.8.0') and torchvision.__version__.startswith('0.23.0') and torch.cuda.is_available(); print('1' if ok else '0')" 2>$null | Select-Object -Last 1
        if ($LASTEXITCODE -ne 0 -or $cudaCheck -ne "1") {
            Write-Step "Installing CUDA-enabled PyTorch runtime"
            & $PythonExe -m pip install --upgrade --force-reinstall `
                torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 `
                --index-url $TorchIndexUrl
            if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed" }
        }
    } elseif ($CpuOnly) {
        $cpuCheck = & $PythonExe -c "import torch, torchaudio, torchvision; ok = torch.__version__.startswith('2.8.0+cpu') and torchaudio.__version__.startswith('2.8.0+cpu') and torchvision.__version__.startswith('0.23.0+cpu'); print('1' if ok else '0')" 2>$null | Select-Object -Last 1
        if ($LASTEXITCODE -ne 0 -or $cpuCheck -ne "1") {
            Write-Step "Installing CPU-only PyTorch runtime"
            & $PythonExe -m pip install --upgrade --force-reinstall `
                torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 `
                --index-url $CpuTorchIndexUrl
            if ($LASTEXITCODE -ne 0) { throw "CPU PyTorch installation failed" }
        }
    }

    Write-Step "Installing Python dependencies"
    & $PythonExe -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed" }

    if ($hasNvidiaGpu) {
        & $PythonExe -c "import torch; assert torch.cuda.is_available(), 'CUDA PyTorch is not available after installation'; print(torch.cuda.get_device_name(0))"
        if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch verification failed" }
    }
}

if ($SkipModels) {
    Write-Step "Skipping Hugging Face model downloads"
    exit 0
}

Write-Step "Checking Hugging Face authentication"
& $HfExe auth whoami
if ($LASTEXITCODE -ne 0) {
    Write-Host "No Hugging Face login detected; public downloads will still be attempted." -ForegroundColor Yellow
}

Write-Step "Downloading MOSS audio tokenizer without Windows symlink cache"
New-Item -ItemType Directory -Force -Path $TokenizerDir | Out-Null
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
& $HfExe download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano `
    --local-dir $TokenizerDir `
    --max-workers 1
if ($LASTEXITCODE -ne 0) { throw "MOSS audio tokenizer download failed" }

Write-Host ""
Write-Host "Public tokenizer is ready at: $TokenizerDir" -ForegroundColor Green
Write-Host ""
Write-Host "VieNeu-TTS model note:" -ForegroundColor Yellow
Write-Host "  The current public v3 Turbo repository is pnnbao-ump/VieNeu-TTS-v3-Turbo."
Write-Host "  If Hugging Face requires authentication, run: .\.venv\Scripts\hf.exe auth login"
Write-Host "  Then retry: .\.venv\Scripts\hf.exe download pnnbao-ump/VieNeu-TTS-v3-Turbo --local-dir .\models\VieNeu-TTS-v3-Turbo --max-workers 1"
