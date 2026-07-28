# Auto-Dubbing Video API

Synchronous FastAPI pipeline for local, low-VRAM auto-dubbing.

## Run

```powershell
.\scripts\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

## Project Structure

The backend lives under `app/`:

- `app/main.py` is the canonical FastAPI entrypoint.
- `app/api/routes.py` owns all `/api` and `/api/v1` routes.
- `app/services/` contains pipeline orchestration and stage services.
- root `utils/` contains shared model, STT, translation, OCR, TTS, workspace, and VRAM helpers.

The root `main.py` is only a compatibility shim so older `uvicorn main:app`
commands resolve to the same app without mounting duplicate legacy routes.

The setup script downloads `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` into
`models\MOSS-Audio-Tokenizer-Nano` with `hf download --local-dir`, which avoids
Windows symlink privilege errors in the default Hugging Face cache.

If a private or gated Hugging Face model returns `401 Unauthorized`, log in
first:

```powershell
.\.venv\Scripts\hf.exe auth login
```

For long video processing jobs, prefer running without `--reload` so file
watcher restarts do not interrupt an active pipeline:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Windows FFmpeg Requirement

The backend needs both `ffmpeg.exe` and `ffprobe.exe` available in PATH.

Check:

```powershell
ffmpeg -version
ffprobe -version
```

Fast dependency check:

```text
GET /api/v1/health/dependencies
```

If `ffmpeg_binary` or `ffprobe_binary` is `false`, install FFmpeg and add its
`bin` folder to PATH, then restart the backend.

## Force VieNeu-TTS On RTX

The backend defaults VieNeu-TTS to GPU:

```text
tts_device=cuda
voice_model=Trúc Ly
mock_tts=false
```

Verify CUDA is visible to this Python environment:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

If it prints `False`, reinstall PyTorch with a CUDA wheel from the official
PyTorch index. Example for CUDA 12.8 wheels:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then restart the backend and check:

```text
GET /api/v1/health/dependencies
```

POST a video to:

```text
POST /api/v1/dub
```

Frontend-compatible streaming endpoint:

```text
POST /api/process-video
```

Additional pipeline endpoints:

```text
POST /api/v1/dub-with-srt
POST /api/v1/batch-dub
POST /api/v1/douyin
```

The default request uses mock translation and mock TTS so the API can be tested
before installing/configuring external translation and TTS providers.

See [PIPELINE.md](PIPELINE.md) for the full pipeline map.

## Low-VRAM Rule

AI stages run strictly in this order:

1. WhisperX ASR
2. Translation
3. TTS
4. CPU ffmpeg video composition

Each model-backed service unloads its model and calls `VRAMManager.cleanup()`
before the next stage begins.
