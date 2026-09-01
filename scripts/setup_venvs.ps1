param(
    [switch]$SkipInstall,
    [switch]$InstallAsr,
    [switch]$InstallAll,
    [switch]$DownloadModels,
    [string]$ParaformerModel = "funasr/paraformer-zh"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

function Ensure-Venv([string]$Name) {
    $path = Join-Path $ProjectRoot $Name
    $python = Join-Path $path "Scripts\python.exe"
    if (-not (Test-Path $python)) {
        Write-Host "Creating $Name" -ForegroundColor Cyan
        py -3.12 -m venv $path
    }
    return $python
}

$whisperPython = Ensure-Venv ".venv-whisper"
$asrPython = Ensure-Venv ".venv-asr"
$vieneuPython = Ensure-Venv ".venv-vieneu"

if ($SkipInstall) { exit 0 }

Write-Host "Installing isolated Whisper runtime" -ForegroundColor Cyan
& $whisperPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Whisper pip upgrade failed" }
& $whisperPython -m pip install -r (Join-Path $ProjectRoot "requirements-whisper.txt")
if ($LASTEXITCODE -ne 0) { throw "Whisper dependency installation failed" }

if ($InstallAsr -or $InstallAll) {
    Write-Host "Installing isolated Paraformer runtime" -ForegroundColor Cyan
    & $asrPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "ASR pip upgrade failed" }
    & $asrPython -m pip install -r (Join-Path $ProjectRoot "requirements-asr.txt")
    if ($LASTEXITCODE -ne 0) { throw "ASR dependency installation failed" }
}

if ($InstallAll) {
    Write-Host "Installing isolated VieNeu runtime" -ForegroundColor Cyan
    & $vieneuPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "VieNeu pip upgrade failed" }
    & $vieneuPython -m pip install -r (Join-Path $ProjectRoot "requirements-vieneu.txt")
    if ($LASTEXITCODE -ne 0) { throw "VieNeu dependency installation failed" }
}

if ($DownloadModels) {
    Write-Host "Downloading Paraformer checkpoint into models/paraformer-zh" -ForegroundColor Cyan
    $paraformerDir = Join-Path $ProjectRoot "models\paraformer-zh"
    New-Item -ItemType Directory -Force -Path $paraformerDir | Out-Null
    # HF hosts the same FunASR checkpoint with a faster resumable transfer;
    # the runtime still uses FunASR/ModelScope for aliases when no local path
    # is configured.
    & $asrPython -c "from huggingface_hub import snapshot_download; snapshot_download('$ParaformerModel', local_dir=r'$paraformerDir', allow_patterns=['model.pt','config.yaml','configuration.json','tokens.json','am.mvn','README.md'], max_workers=4)"
    if ($LASTEXITCODE -ne 0) { throw "Paraformer download failed" }

    Write-Host "Downloading VieNeu-TTS v3 Turbo into models/VieNeu-TTS-v3-Turbo" -ForegroundColor Cyan
    $vieneuModelDir = Join-Path $ProjectRoot "models\VieNeu-TTS-v3-Turbo"
    New-Item -ItemType Directory -Force -Path $vieneuModelDir | Out-Null
    $hf = Join-Path $vieneuPython "Scripts\hf.exe"
    & $hf download pnnbao-ump/VieNeu-TTS-v3-Turbo --local-dir $vieneuModelDir --max-workers 1
    if ($LASTEXITCODE -ne 0) { throw "VieNeu model download failed" }

    Write-Host "Downloading VieNeu ONNX tokenizer" -ForegroundColor Cyan
    $codecDir = Join-Path $ProjectRoot "models\MOSS-Audio-Tokenizer-Nano-ONNX"
    New-Item -ItemType Directory -Force -Path $codecDir | Out-Null
    & $vieneuPython -c "from huggingface_hub import snapshot_download; snapshot_download('OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX', local_dir=r'$codecDir', max_workers=1)"
    if ($LASTEXITCODE -ne 0) { throw "VieNeu ONNX tokenizer download failed" }
}

Write-Host "Isolated runtimes ready: .venv-whisper, .venv-asr, .venv-vieneu" -ForegroundColor Green
