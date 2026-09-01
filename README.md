# Auto-Dubbing Video API

Synchronous FastAPI pipeline for local, low-VRAM auto-dubbing.

## Run

```powershell
.\scripts\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

## Automatic media cleanup

The API periodically removes abandoned temporary workspaces after 24 hours
and files in `output/` after 12 hours. Active workspaces and queued/running
render jobs are protected. Override the defaults in `.env` when needed:

```text
AUTODUB_TEMP_CLEANUP_MAX_AGE_HOURS=24
AUTODUB_TEMP_CLEANUP_INTERVAL_HOURS=24
AUTODUB_OUTPUT_CLEANUP_MAX_AGE_HOURS=12
AUTODUB_OUTPUT_CLEANUP_INTERVAL_HOURS=1
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

## Durable Render Queue

Edited-script renders use a durable job fingerprint and persist their source,
manifest, progress events, and intermediate chunks under `render_jobs/`.
Repeated requests with the same video, script, timeline, and clone reference
attach to the same job. A changed segment creates a new job but still reuses
unchanged content-addressed TTS chunks.

Local development defaults to one in-process render worker. For API reloads or
production use, run Redis and the dedicated Celery worker:

```powershell
docker compose up -d redis
.\scripts\start_render_worker.ps1
```

Start the API in another terminal:

```powershell
.\scripts\start_queued_backend.ps1
```

The render worker intentionally uses Celery's `solo` pool, concurrency `1`,
prefetch `1`, late acknowledgements, and the dedicated `render` queue. This
keeps CUDA/model work sequential. Do not use Uvicorn `--reload` for the queued
backend; API restarts no longer stop the external render worker.

Redis in `compose.yaml` enables AOF with `appendfsync everysec` and periodic RDB
snapshots. Redis stores queue/result state; large media and checkpoints remain
on the project disk.

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

## Voice source modes

The studio exposes two isolated VieNeu-TTS paths:

- `system`: every segment must resolve to the selected built-in voice ID.
- `clone`: choose an exact 3-second window from a longer clean reference. The
  browser uploads only that cut, and the backend validates/canonicalizes it to
  24 kHz mono PCM WAV (72,000 frames) before reusing its speaker data.

Voice selection is strict. An unavailable system voice or an invalid clone
reference stops the render and returns an error; it never switches to another
voice or to the model default.

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

Additive Short Video workflow (the existing Clone Video workflow remains
unchanged):

```text
POST /api/v1/short-video/inspect
POST /api/v1/short-video/analyze
POST /api/v1/short-video/dub
POST /api/v1/short-video/render-script
GET  /api/v1/short-video/profiles
```

`/short-video/inspect` stores the upload once, probes its duration/audio/video
metadata, and returns the backend-owned profile (`micro`, `short`,
`short_extended`, or `long`). Analyze and dub accept the returned `media_id`;
videos above `AUTODUB_SHORT_VIDEO_MAX_SECONDS` are rejected by the Short Video
workflow and should be sent to Clone Video instead.

The default Short limit is 120 seconds (2 minutes). Short requests use the
dedicated `ShortVideoPipeline`: semantic stitching is disabled, word/OCR cues
stay independently editable, and each timeline cue gets its own TTS chunk so
timing is not silently merged with the long-video policy. The Short Video UI
also exposes ASR/Whisper/Paraformer, translation and OCR model controls; those
values are sent as request fields rather than hardcoded in the browser.

Clone Video uses the same backend settings contract at `GET /api/pipeline/settings`
(also available at `/api/v1/pipeline/settings`). It supplies source languages,
ASR engines/models, OCR defaults and timing limits to the UI; long-video
analyze/render endpoints accept those values explicitly. Whisper repetition and
number-flooding hallucinations are retried with guarded decoding and rejected
before translation if they remain suspect, so bad ASR is not silently turned
into subtitles.

For continuous dialogue, ASR runs without an aggressive VAD filter and the
pipeline checks the original PCM audio between cues. Audio-bearing gaps are
closed at the midpoint; genuine silence is preserved. This behavior is
controlled by `AUTODUB_FILL_SPEECH_GAPS` and `AUTODUB_SPEECH_GAP_MAX_S`.

The default request uses mock translation and mock TTS so the API can be tested
before installing/configuring external translation and TTS providers.

### Optimized recognition and durable stages

ASR is selected with `AUTODUB_ASR_ENGINE=auto|whisper|paraformer`. Whisper
uses `WHISPER_MODEL=auto`/`AUTODUB_WHISPER_MODEL` and a configurable beam size;
explicit Paraformer jobs use the isolated `.venv-asr` runtime. Every decoded
segment receives a language hint and confidence, so mixed-language shorts are
translated in contiguous language groups instead of forcing one language over
the whole file.

ASR, adaptive OCR and translation checkpoints are content-addressed under
`AUTODUB_CHECKPOINT_ROOT` and are written atomically. They can be reused after
a backend restart or machine shutdown. `scripts/benchmark_ocr.py` measures
real OCR elapsed time, frame budget and extracted cues for a supplied video.

Model weights are also cached in-process. With `AUTODUB_CPU_OFFLOAD=0`, the
registry keeps the last active model warm and moves other stages to CPU before
switching, so repeated renders do not deserialize the same Whisper/VieNeu
weights again. `AUTODUB_PRELOAD_MODELS=asr` optionally warms ASR during backend
startup; use `AUTODUB_CPU_OFFLOAD=1` instead on GPUs with very little VRAM.

The shared pipeline honors `TRANSLATE_ANALYSIS`, `TRANSLATE_REVIEW`,
`TRANSLATE_BATCH_SIZE`, `TRANSLATE_CPS_BUDGET`, soft timing limits, voice/video
speed, LUFS normalization and background ducking. The Short Video clone-voice
control uploads its reference through the same `/api/voice-reference` endpoint
used by Clone Video; it does not create a second TTS implementation.

Voice-reference uploads are capped by `AUTODUB_VOICE_REFERENCE_MAX_BYTES`
(16 MiB by default). Files shorter than 3 seconds, full untrimmed recordings,
renamed non-audio files, and clips outside the exact-duration tolerance are
rejected instead of being passed through to VieNeu.

Heavy dependencies can be isolated with:

```powershell
scripts/setup_venvs.ps1 -InstallAll
```

This creates `.venv-whisper`, `.venv-asr` and `.venv-vieneu`; the main runtime
can remain on `requirements-core.txt` when model workers are deployed
separately.

After package installation, download the local checkpoints (Paraformer-zh,
VieNeu-TTS v3 Turbo and its ONNX tokenizer) with:

```powershell
scripts/setup_venvs.ps1 -InstallAll -DownloadModels
```

The downloaded paths are `models/paraformer-zh`,
`models/VieNeu-TTS-v3-Turbo` and `models/MOSS-Audio-Tokenizer-Nano-ONNX`.
The setup script keeps these model directories out of git and configures the
backend to use them offline.

See [PIPELINE.md](PIPELINE.md) for the full pipeline map.

## Low-VRAM Rule

AI stages run strictly in this order:

1. WhisperX ASR
2. Translation
3. TTS
4. CPU ffmpeg video composition

Each model-backed service unloads its model and calls `VRAMManager.cleanup()`
before the next stage begins.
