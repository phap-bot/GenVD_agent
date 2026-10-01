# Auto-Dubbing Pipeline

## 1. Video to Dubbed Video

Endpoint: `POST /api/v1/dub`

Flow:

1. Upload video.
2. Extract audio with ffmpeg.
3. Run WhisperX ASR to create timestamped text.
4. Offload ASR model back to CPU RAM and clear VRAM cache.
5. Translate text while preserving timestamps.
6. Normalize the translated script into one canonical timeline.
7. Run TTS from that same canonical timeline.
8. Offload TTS model back to CPU RAM and clear VRAM cache.
9. Generate translated SRT from that same canonical timeline.
10. Mix original background audio with generated TTS audio.
11. Burn subtitles and export final video.

After translation, the pipeline also sends the paired source/translated script
to the configured translation LLM and returns up to three grounded, one-line
caption suggestions. This is a separate metadata stage: captions do not alter
the canonical subtitle/TTS timeline. Results are checkpointed by media, script,
provider and model, with an extractive translated-script fallback when an LLM
gateway is unavailable.

## 2. Video + SRT to Dubbed Video

Endpoint: `POST /api/v1/dub-with-srt`

Flow:

1. Upload video and `.srt`.
2. Parse SRT timestamps/text.
3. Translate subtitle text.
4. Normalize the translated subtitle script into one canonical timeline.
5. Generate TTS and SRT from that same canonical timeline.
6. Mix new audio with original audio.
7. Burn translated subtitles and export final video.

## Canonical Timeline Rule

Subtitle text, TTS text, segment start time, and segment end time must come from
the same normalized segment list. After ASR/OCR/SRT parsing, translation, or
human script edits, the pipeline trims whitespace, reindexes segments, fixes
minimum duration, and then passes that single timeline to both subtitle rendering
and TTS generation.

Each TTS chunk is time-fit to its segment duration before it is delayed onto the
mixed track, so the spoken line occupies the same slot as the subtitle line.

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

The GPU stages are never parallelized. Models are held by the process-wide
registry and moved back to CPU RAM after use:

```text
singleton load -> move to GPU -> inference -> offload to CPU RAM -> torch.cuda.empty_cache() -> next stage
```

Batch and Douyin processing are also strictly sequential.
