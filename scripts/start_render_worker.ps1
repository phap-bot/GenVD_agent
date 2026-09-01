$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found: $python"
}

$env:AUTODUB_QUEUE_BACKEND = "celery"
if (-not $env:AUTODUB_REDIS_URL) {
    $env:AUTODUB_REDIS_URL = "redis://127.0.0.1:6379/0"
}

Set-Location $projectRoot
& $python -m celery -A app.celery_app:celery_app worker `
    --loglevel=INFO `
    --pool=solo `
    --concurrency=1 `
    --prefetch-multiplier=1 `
    --queues=render

