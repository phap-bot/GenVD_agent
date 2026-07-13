# Auto-Dubbing Pipeline

## 1. Video to Dubbed Video

Endpoint: `POST /api/v1/dub`

Flow:

1. Upload video.
2. Extract audio with ffmpeg.
3. Run WhisperX ASR to create timestamped text.
4. Unload ASR model and clear VRAM.
5. Translate text while preserving timestamps.
6. Run TTS for each translated segment.
7. Unload TTS model and clear VRAM.
8. Generate translated SRT.
9. Mix original background audio with generated TTS audio.
10. Burn subtitles and export final video.

## 2. Video + SRT to Dubbed Video

Endpoint: `POST /api/v1/dub-with-srt`

Flow:

1. Upload video and `.srt`.
2. Parse SRT timestamps/text.
3. Translate subtitle text.
4. Generate TTS by timestamp segment.
5. Mix new audio with original audio.
6. Burn translated subtitles and export final video.

## 3. Batch Video Processing

Endpoint: `POST /api/v1/batch-dub`

Flow:

1. Upload multiple videos.
2. Process each file strictly one by one.
3. Clear VRAM after every file and every model stage.
4. Return completed outputs and per-file failures.

## 4. Douyin Download + Processing

Endpoint: `POST /api/v1/douyin`

Flow:

1. Send Douyin video URL or user URL.
2. Download with `yt-dlp`.
3. For `single`, process one downloaded video.
4. For `user`, process up to `max_items` videos.
5. Run the same sequential dubbing pipeline per file.

## Low VRAM Rule

The GPU stages are never parallelized. The backend must always follow:

```text
load model -> inference -> del model -> torch.cuda.empty_cache() -> next stage
```

Batch and Douyin processing are also strictly sequential.
