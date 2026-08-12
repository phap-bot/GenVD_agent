from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow ``python scripts/benchmark_ocr.py`` from the repository root as well
# as ``python -m scripts.benchmark_ocr``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.ocr import _capture_subtitle_frames, extract_video_ocr_segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark adaptive subtitle OCR for a real video file.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-adaptive", action="store_true")
    parser.add_argument("--scene-threshold", type=float, default=0.28)
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Benchmark adaptive frame capture/encoding without calling the remote OCR endpoint.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    if args.capture_only:
        frames, duration = _capture_subtitle_frames(
            args.video,
            args.interval,
            0.35,
            args.max_frames,
            adaptive=not args.no_adaptive,
            scene_threshold=args.scene_threshold,
        )
        segments = []
    else:
        frames = []
        duration = None
        segments = extract_video_ocr_segments(
            args.video,
            interval_seconds=args.interval,
            max_frames=args.max_frames,
            model=args.model,
            adaptive=not args.no_adaptive,
            scene_threshold=args.scene_threshold,
        )
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "video": str(args.video),
        "elapsed_seconds": round(elapsed, 3),
        "segments": len(segments),
        "captured_frames": len(frames),
        "duration_seconds": duration,
        "adaptive": not args.no_adaptive,
        "max_frames": args.max_frames,
        "capture_only": args.capture_only,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
