"""Offline tests for the Python PCM-based _combine_audio_chunks.

Run:  python -m pytest tests/test_combine_audio_chunks.py -v
  or: python tests/test_combine_audio_chunks.py
"""
from __future__ import annotations

import math
import struct
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal stubs so the test can run without importing the full pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AudioChunk:
    segment_id: int
    path: Path
    start: float
    end: float


def _write_test_tone_wav(
    path: Path,
    duration: float,
    sample_rate: int = 24000,
    frequency: float = 440.0,
    amplitude: int = 8000,
    channels: int = 1,
) -> None:
    """Write a simple sine-tone WAV for testing."""
    frames = max(1, math.ceil(duration * sample_rate))
    samples: list[int] = []
    for i in range(frames):
        value = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
        for _ in range(channels):
            samples.append(value)
    raw = struct.pack(f"<{len(samples)}h", *samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)


def _write_silent_wav(path: Path, duration: float, sample_rate: int = 24000) -> None:
    frames = max(1, math.ceil(duration * sample_rate))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * frames)


def _read_wav(path: Path) -> tuple[int, int, int, int, bytes]:
    """Return (channels, sampwidth, framerate, nframes, raw_bytes)."""
    with wave.open(str(path), "rb") as wf:
        return (
            wf.getnchannels(),
            wf.getsampwidth(),
            wf.getframerate(),
            wf.getnframes(),
            wf.readframes(wf.getnframes()),
        )


# ---------------------------------------------------------------------------
# Import the actual _combine_audio_chunks via the pipeline class
# ---------------------------------------------------------------------------

# Add project root to sys.path so imports work from the tests/ directory.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.services.pipeline import AutoDubbingPipeline, AudioChunk as RealAudioChunk  # noqa: E402
from app.models.schemas import PipelineConfig  # noqa: E402


def _make_pipeline() -> AutoDubbingPipeline:
    config = PipelineConfig(
        copyright_confirmed=True,
        copyright_source="owned",
    )
    return AutoDubbingPipeline(config)


# ========================= TESTS =========================


def test_empty_chunks(tmp_path: Path) -> None:
    """No chunks → output is a silent WAV at the requested duration."""
    dest = tmp_path / "mix.wav"
    pipeline = _make_pipeline()
    pipeline._combine_audio_chunks([], dest, total_duration=3.0)

    channels, sampwidth, framerate, nframes, _ = _read_wav(dest)
    # Empty chunks calls _write_silent_wav which produces mono—correct
    # behaviour since _render_video handles channel conversion via FFmpeg.
    assert channels >= 1
    assert framerate in (24000, 44100)
    assert nframes > 0
    print("  PASS: test_empty_chunks")


def test_single_chunk_at_zero(tmp_path: Path) -> None:
    """One chunk placed at t=0 should produce a valid stereo WAV."""
    tone_path = tmp_path / "0001.wav"
    _write_test_tone_wav(tone_path, duration=1.0, sample_rate=24000)

    dest = tmp_path / "mix.wav"
    chunk = RealAudioChunk(segment_id=1, path=tone_path, start=0.0, end=1.0)
    pipeline = _make_pipeline()
    pipeline._combine_audio_chunks([chunk], dest, total_duration=2.0)

    channels, sampwidth, framerate, nframes, raw = _read_wav(dest)
    assert channels == 2
    assert sampwidth == 2
    assert framerate == 44100
    expected_frames = math.ceil(2.0 * 44100)
    assert nframes == expected_frames, f"Expected {expected_frames} frames, got {nframes}"

    # The first ~44100 frames should have non-zero audio (resampled tone).
    import numpy as np
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    first_second = samples[:44100, 0]
    assert np.any(first_second != 0), "First second should contain the tone"

    # The second second should be silence (zero-padded).
    second_second = samples[44100:, 0]
    assert np.all(second_second == 0), "Second second should be silence"
    print("  PASS: test_single_chunk_at_zero")


def test_chunk_placement_timing(tmp_path: Path) -> None:
    """Chunk at t=1.0 → audio should start at sample offset 44100."""
    tone_path = tmp_path / "0001.wav"
    _write_test_tone_wav(tone_path, duration=0.5, sample_rate=24000)

    dest = tmp_path / "mix.wav"
    chunk = RealAudioChunk(segment_id=1, path=tone_path, start=1.0, end=1.5)
    pipeline = _make_pipeline()
    pipeline._combine_audio_chunks([chunk], dest, total_duration=3.0)

    channels, sampwidth, framerate, nframes, raw = _read_wav(dest)
    import numpy as np
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)

    # Frames 0..44099 should be silence
    before = samples[:44100, 0]
    assert np.all(before == 0), "Audio before t=1.0 should be silent"

    # Frames 44100..66150 should have audio (~0.5s at 44100Hz = 22050 frames)
    tone_region = samples[44100:44100 + 22050, 0]
    assert np.any(tone_region != 0), "Tone region should have non-zero audio"

    # Frames after the tone should be silence
    after = samples[44100 + 22050:, 0]
    assert np.all(after == 0), "Audio after the tone should be silent"
    print("  PASS: test_chunk_placement_timing")


def test_multiple_chunks_no_overlap(tmp_path: Path) -> None:
    """Two non-overlapping chunks should both be present at their offsets."""
    tone1 = tmp_path / "0001.wav"
    tone2 = tmp_path / "0002.wav"
    _write_test_tone_wav(tone1, duration=0.5, sample_rate=24000, frequency=440)
    _write_test_tone_wav(tone2, duration=0.5, sample_rate=24000, frequency=880)

    dest = tmp_path / "mix.wav"
    chunks = [
        RealAudioChunk(segment_id=1, path=tone1, start=0.0, end=0.5),
        RealAudioChunk(segment_id=2, path=tone2, start=1.0, end=1.5),
    ]
    pipeline = _make_pipeline()
    pipeline._combine_audio_chunks(chunks, dest, total_duration=2.0)

    channels, sampwidth, framerate, nframes, raw = _read_wav(dest)
    import numpy as np
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)

    # Chunk 1 region: 0..22050
    region1 = samples[:22050, 0]
    assert np.any(region1 != 0), "Chunk 1 region should have audio"

    # Gap: 22050..44100
    gap = samples[22050:44100, 0]
    assert np.all(gap == 0), "Gap between chunks should be silent"

    # Chunk 2 region: 44100..66150
    region2 = samples[44100:44100 + 22050, 0]
    assert np.any(region2 != 0), "Chunk 2 region should have audio"
    print("  PASS: test_multiple_chunks_no_overlap")


def test_resampling_24k_to_44k(tmp_path: Path) -> None:
    """A 24000 Hz chunk should be resampled to 44100 Hz in the output."""
    tone_path = tmp_path / "0001.wav"
    input_sr = 24000
    duration = 1.0
    _write_test_tone_wav(tone_path, duration=duration, sample_rate=input_sr)

    dest = tmp_path / "mix.wav"
    chunk = RealAudioChunk(segment_id=1, path=tone_path, start=0.0, end=duration)
    pipeline = _make_pipeline()
    pipeline._combine_audio_chunks([chunk], dest, total_duration=duration)

    channels, sampwidth, framerate, nframes, raw = _read_wav(dest)
    assert framerate == 44100, f"Output should be 44100 Hz, got {framerate}"
    expected_frames = math.ceil(duration * 44100)
    assert nframes == expected_frames
    print("  PASS: test_resampling_24k_to_44k")


def test_stereo_output(tmp_path: Path) -> None:
    """Output should always be stereo with identical L/R channels."""
    tone_path = tmp_path / "0001.wav"
    _write_test_tone_wav(tone_path, duration=0.5, sample_rate=24000)

    dest = tmp_path / "mix.wav"
    chunk = RealAudioChunk(segment_id=1, path=tone_path, start=0.0, end=0.5)
    pipeline = _make_pipeline()
    pipeline._combine_audio_chunks([chunk], dest, total_duration=1.0)

    channels, _, _, _, raw = _read_wav(dest)
    assert channels == 2

    import numpy as np
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    assert np.array_equal(samples[:, 0], samples[:, 1]), "L and R channels should be identical"
    print("  PASS: test_stereo_output")


def test_large_chunk_count(tmp_path: Path) -> None:
    """Simulate 200 chunks — the scenario that caused WinError 206.

    This would have produced a 32k+ character FFmpeg command line. The new
    implementation must handle it without error.
    """
    chunk_count = 200
    chunks: list[RealAudioChunk] = []
    for i in range(chunk_count):
        tone_path = tmp_path / f"{i:04d}.wav"
        _write_test_tone_wav(tone_path, duration=0.3, sample_rate=24000, amplitude=400)
        start = i * 0.5
        chunks.append(RealAudioChunk(segment_id=i, path=tone_path, start=start, end=start + 0.3))

    total = chunk_count * 0.5 + 0.3
    dest = tmp_path / "mix.wav"
    pipeline = _make_pipeline()
    pipeline._combine_audio_chunks(chunks, dest, total_duration=total)

    channels, sampwidth, framerate, nframes, _ = _read_wav(dest)
    assert channels == 2
    assert framerate == 44100
    expected_frames = math.ceil(total * 44100)
    assert nframes == expected_frames, f"Expected {expected_frames} frames, got {nframes}"
    print(f"  PASS: test_large_chunk_count ({chunk_count} chunks)")


def test_chunk_clipped_to_total_duration(tmp_path: Path) -> None:
    """A chunk extending beyond total_duration should be trimmed."""
    tone_path = tmp_path / "0001.wav"
    _write_test_tone_wav(tone_path, duration=2.0, sample_rate=24000)

    dest = tmp_path / "mix.wav"
    chunk = RealAudioChunk(segment_id=1, path=tone_path, start=0.0, end=2.0)
    pipeline = _make_pipeline()
    # total_duration < chunk duration
    pipeline._combine_audio_chunks([chunk], dest, total_duration=1.0)

    channels, sampwidth, framerate, nframes, _ = _read_wav(dest)
    expected = math.ceil(1.0 * 44100)
    assert nframes == expected, f"Output should be trimmed to 1.0s ({expected} frames), got {nframes}"
    print("  PASS: test_chunk_clipped_to_total_duration")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_empty_chunks,
    test_single_chunk_at_zero,
    test_chunk_placement_timing,
    test_multiple_chunks_no_overlap,
    test_resampling_24k_to_44k,
    test_stereo_output,
    test_large_chunk_count,
    test_chunk_clipped_to_total_duration,
]


def main() -> None:
    import tempfile

    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        name = test_fn.__name__
        print(f"Running {name} ...")
        with tempfile.TemporaryDirectory(prefix="audiomix_test_") as tmp:
            try:
                test_fn(Path(tmp))
                passed += 1
            except Exception as exc:
                print(f"  FAIL: {name} -> {exc}")
                import traceback
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(ALL_TESTS)}")
    if failed:
        sys.exit(1)
    print("All tests passed OK")


if __name__ == "__main__":
    main()
