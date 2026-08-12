"use client";

import { Loader2, Pause, Play, Scissors } from "lucide-react";
import { PointerEvent, useCallback, useEffect, useRef, useState } from "react";

export const VOICE_REFERENCE_SECONDS = 3;
const MIN_SOURCE_TOLERANCE_SECONDS = 0.001;

type Selection = {
  start: number;
  end: number;
};

export type PreparedAudioClip = {
  file: File;
  start: number;
  end: number;
  sourceDuration: number;
};

type AudioClipSelectorProps = {
  file: File;
  initialSelection?: Selection | null;
  disabled?: boolean;
  onClipReady: (clip: PreparedAudioClip | null) => void;
  onError: (message: string) => void;
};

type DragState = {
  mode: "move" | "new";
  anchorTime: number;
  initial: Selection;
};

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function formatTime(value: number) {
  const safeValue = Math.max(0, value);
  const minutes = Math.floor(safeValue / 60);
  const seconds = safeValue - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}

function normalizedSelection(selection: Selection, duration: number): Selection {
  if (duration < VOICE_REFERENCE_SECONDS) return { start: 0, end: duration };
  const start = clamp(Math.min(selection.start, selection.end), 0, duration - VOICE_REFERENCE_SECONDS);
  return { start, end: start + VOICE_REFERENCE_SECONDS };
}

function centeredSelection(center: number, duration: number): Selection {
  return normalizedSelection(
    {
      start: center - VOICE_REFERENCE_SECONDS / 2,
      end: center + VOICE_REFERENCE_SECONDS / 2,
    },
    duration,
  );
}

function encodeSelectionAsWav(buffer: AudioBuffer, selection: Selection) {
  const frameCount = Math.round(VOICE_REFERENCE_SECONDS * buffer.sampleRate);
  const maxStartFrame = Math.max(0, buffer.length - frameCount);
  const startFrame = Math.round(clamp(selection.start * buffer.sampleRate, 0, maxStartFrame));
  const endFrame = startFrame + frameCount;
  const wav = new ArrayBuffer(44 + frameCount * 2);
  const view = new DataView(wav);

  const writeText = (offset: number, value: string) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeText(0, "RIFF");
  view.setUint32(4, 36 + frameCount * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, frameCount * 2, true);

  const channels = Array.from({ length: buffer.numberOfChannels }, (_, index) => buffer.getChannelData(index));
  let byteOffset = 44;
  for (let frame = startFrame; frame < endFrame; frame += 1) {
    let sample = 0;
    for (const channel of channels) sample += channel[frame] ?? 0;
    sample = clamp(sample / channels.length, -1, 1);
    view.setInt16(byteOffset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    byteOffset += 2;
  }

  return new Blob([wav], { type: "audio/wav" });
}

function drawWaveform(canvas: HTMLCanvasElement, buffer: AudioBuffer) {
  const rect = canvas.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * pixelRatio));
  const height = Math.max(1, Math.round(rect.height * pixelRatio));
  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext("2d");
  if (!context) return;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#f8fafc";
  context.fillRect(0, 0, width, height);

  const data = buffer.getChannelData(0);
  const samplesPerPixel = Math.max(1, Math.floor(data.length / width));
  const center = height / 2;
  context.fillStyle = "#64748b";
  for (let x = 0; x < width; x += 1) {
    const from = x * samplesPerPixel;
    const to = Math.min(data.length, from + samplesPerPixel);
    let peak = 0;
    for (let index = from; index < to; index += 1) peak = Math.max(peak, Math.abs(data[index]));
    const barHeight = Math.max(1, peak * height * 0.88);
    context.fillRect(x, center - barHeight / 2, 1, barHeight);
  }
}

export default function AudioClipSelector({ file, initialSelection, disabled = false, onClipReady, onError }: AudioClipSelectorProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const stopTimerRef = useRef<number | null>(null);
  const playAttemptRef = useRef(0);
  const dragRef = useRef<DragState | null>(null);
  const [buffer, setBuffer] = useState<AudioBuffer | null>(null);
  const [selection, setSelection] = useState<Selection>({ start: 0, end: 0 });
  const [clipPreviewUrl, setClipPreviewUrl] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loadError, setLoadError] = useState("");

  const clearStopTimer = useCallback(() => {
    if (stopTimerRef.current !== null) window.clearTimeout(stopTimerRef.current);
    stopTimerRef.current = null;
  }, []);

  const stopPreview = useCallback(() => {
    playAttemptRef.current += 1;
    clearStopTimer();
    audioRef.current?.pause();
    setIsPlaying(false);
  }, [clearStopTimer]);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setBuffer(null);
    setClipPreviewUrl("");
    setLoadError("");
    onClipReady(null);

    const decode = async () => {
      const AudioContextClass = window.AudioContext;
      const context = new AudioContextClass();
      try {
        const decoded = await context.decodeAudioData(await file.arrayBuffer());
        if (cancelled) return;
        if (decoded.duration + MIN_SOURCE_TOLERANCE_SECONDS < VOICE_REFERENCE_SECONDS) {
          const message = `File giọng mẫu phải dài ít nhất ${VOICE_REFERENCE_SECONDS} giây để cắt đúng đoạn yêu cầu.`;
          setLoadError(message);
          onError(message);
          return;
        }
        const nextSelection = normalizedSelection(
          initialSelection ?? { start: 0, end: VOICE_REFERENCE_SECONDS },
          decoded.duration,
        );
        setBuffer(decoded);
        setSelection(nextSelection);
      } catch {
        if (!cancelled) {
          const message = "Không đọc được file âm thanh này. Hãy thử WAV, MP3, M4A hoặc FLAC khác.";
          setLoadError(message);
          onError(message);
        }
      } finally {
        await context.close();
        if (!cancelled) setIsLoading(false);
      }
    };

    void decode();
    return () => {
      cancelled = true;
    };
  }, [file, initialSelection, onClipReady, onError]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !buffer) return;
    const render = () => drawWaveform(canvas, buffer);
    render();
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [buffer, isLoading]);

  useEffect(() => {
    if (!buffer || selection.end <= selection.start) return;
    // Invalidate the previously encoded file immediately. Without this, a
    // user can move the window and click Analyze/Render during the debounce,
    // which would upload the prior three-second region.
    stopPreview();
    setClipPreviewUrl("");
    onClipReady(null);
    let generatedPreviewUrl = "";
    const timeout = window.setTimeout(() => {
      const blob = encodeSelectionAsWav(buffer, selection);
      const baseName = file.name.replace(/\.[^.]+$/, "") || "voice-reference";
      const preparedFile = new File([blob], `${baseName}-cut.wav`, { type: "audio/wav" });
      generatedPreviewUrl = URL.createObjectURL(preparedFile);
      setClipPreviewUrl(generatedPreviewUrl);
      onClipReady({
        file: preparedFile,
        start: selection.start,
        end: selection.end,
        sourceDuration: buffer.duration,
      });
    }, 120);
    return () => {
      window.clearTimeout(timeout);
      if (generatedPreviewUrl) URL.revokeObjectURL(generatedPreviewUrl);
    };
  }, [buffer, file.name, onClipReady, selection, stopPreview]);

  useEffect(() => {
    if (disabled) {
      dragRef.current = null;
      stopPreview();
    }
  }, [disabled, stopPreview]);

  useEffect(() => () => {
    playAttemptRef.current += 1;
    clearStopTimer();
    audioRef.current?.pause();
  }, [clearStopTimer]);

  const togglePreview = async () => {
    const audio = audioRef.current;
    if (!audio || !buffer || !clipPreviewUrl || disabled) return;
    if (isPlaying) {
      stopPreview();
      return;
    }
    stopPreview();
    const playAttempt = playAttemptRef.current;
    audio.currentTime = 0;
    try {
      await audio.play();
      if (playAttempt !== playAttemptRef.current) {
        audio.pause();
        return;
      }
      setIsPlaying(true);
      stopTimerRef.current = window.setTimeout(stopPreview, VOICE_REFERENCE_SECONDS * 1000);
    } catch {
      onError("Trình duyệt không thể phát đoạn âm thanh đã chọn.");
    }
  };

  const timeFromPointer = (event: PointerEvent<HTMLDivElement>) => {
    const rect = timelineRef.current?.getBoundingClientRect();
    if (!rect || !buffer) return 0;
    return clamp(((event.clientX - rect.left) / rect.width) * buffer.duration, 0, buffer.duration);
  };

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!buffer || disabled) return;
    event.preventDefault();
    stopPreview();
    event.currentTarget.setPointerCapture(event.pointerId);
    const target = event.target as HTMLElement;
    const mode = (target.closest<HTMLElement>("[data-drag-mode]")?.dataset.dragMode || "new") as DragState["mode"];
    const pointerTime = timeFromPointer(event);
    if (mode === "new") {
      const nextSelection = centeredSelection(pointerTime, buffer.duration);
      setSelection(nextSelection);
      dragRef.current = { mode, anchorTime: pointerTime, initial: nextSelection };
      return;
    }
    dragRef.current = { mode, anchorTime: pointerTime, initial: selection };
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || !buffer || disabled) return;
    const pointerTime = timeFromPointer(event);

    if (drag.mode === "move") {
      const length = drag.initial.end - drag.initial.start;
      const start = clamp(drag.initial.start + pointerTime - drag.anchorTime, 0, buffer.duration - length);
      setSelection({ start, end: start + length });
      return;
    }
    setSelection(centeredSelection(pointerTime, buffer.duration));
  };

  const handlePointerUp = (event: PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current || !buffer || disabled) return;
    dragRef.current = null;
    event.currentTarget.releasePointerCapture(event.pointerId);
    setSelection((current) => normalizedSelection(current, buffer.duration));
  };

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 rounded-md border border-blue-200 bg-white px-3 py-4 text-xs font-medium text-slate-600">
        <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
        Đang đọc và tạo waveform...
      </div>
    );
  }

  if (!buffer) {
    return loadError ? (
      <p className="rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-xs font-semibold text-rose-700">
        {loadError}
      </p>
    ) : null;
  }

  const startPercent = (selection.start / buffer.duration) * 100;
  const endPercent = (selection.end / buffer.duration) * 100;
  const clipDuration = selection.end - selection.start;

  return (
    <section className="grid gap-2 rounded-md border border-blue-200 bg-white p-3" aria-label="Cắt đoạn giọng mẫu">
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="inline-flex items-center gap-1 font-semibold text-blue-800">
          <Scissors className="h-3.5 w-3.5" /> Chọn đúng đoạn giọng 3 giây
        </span>
        <span className="tabular-nums text-slate-500">Tổng {formatTime(buffer.duration)}</span>
      </div>

      <div
        ref={timelineRef}
        className={`relative h-24 touch-none select-none overflow-hidden rounded border border-slate-200 bg-slate-50 ${disabled ? "cursor-not-allowed opacity-60" : "cursor-crosshair"}`}
        aria-disabled={disabled}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
      >
        <canvas ref={canvasRef} className="h-full w-full" />
        <div className="pointer-events-none absolute inset-y-0 left-0 bg-slate-900/40" style={{ width: `${startPercent}%` }} />
        <div className="pointer-events-none absolute inset-y-0 right-0 bg-slate-900/40" style={{ width: `${100 - endPercent}%` }} />
        <div
          data-drag-mode="move"
          className="absolute inset-y-0 cursor-grab border-2 border-blue-500 bg-blue-400/20 shadow-[0_0_0_1px_rgba(255,255,255,0.8)] active:cursor-grabbing"
          style={{ left: `${startPercent}%`, width: `${endPercent - startPercent}%` }}
          title="Kéo khung để chọn vị trí của đoạn 3 giây"
        >
          <span className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded bg-blue-600/90 px-2 py-1 text-[10px] font-bold text-white shadow">
            3,0 GIÂY
          </span>
        </div>
      </div>

      <label className="grid gap-1 text-[11px] font-semibold text-slate-600">
        Điều chỉnh vị trí bắt đầu
        <input
          type="range"
          min={0}
          max={Math.max(0, buffer.duration - VOICE_REFERENCE_SECONDS)}
          step={0.01}
          value={selection.start}
          disabled={disabled}
          onChange={(event) => {
            stopPreview();
            const start = Number(event.target.value);
            setSelection(normalizedSelection({ start, end: start + VOICE_REFERENCE_SECONDS }, buffer.duration));
          }}
          className="accent-blue-600 disabled:cursor-not-allowed disabled:opacity-60"
          aria-label="Vị trí bắt đầu đoạn giọng 3 giây"
        />
      </label>

      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-xs tabular-nums text-slate-700">
          <span className="rounded bg-slate-100 px-2 py-1">Từ {formatTime(selection.start)}</span>
          <span>→</span>
          <span className="rounded bg-slate-100 px-2 py-1">Đến {formatTime(selection.end)}</span>
          <strong className="text-blue-700">{clipDuration.toFixed(1)} giây</strong>
        </div>
        <button
          type="button"
          onClick={() => void togglePreview()}
          disabled={disabled || !clipPreviewUrl}
          className="inline-flex items-center gap-1.5 rounded border border-blue-200 bg-blue-50 px-2.5 py-1.5 text-xs font-semibold text-blue-700 hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          {isPlaying ? "Dừng nghe" : "Nghe đoạn chọn"}
        </button>
      </div>
      <p className="text-[11px] leading-4 text-slate-500">
        Kéo khung xanh hoặc thanh vị trí để chọn nội dung mong muốn. Chỉ file WAV chứa đúng 3 giây trong khung được gửi xuống backend; file gốc không được upload.
      </p>
      <audio ref={audioRef} src={clipPreviewUrl || undefined} onEnded={stopPreview} className="hidden" />
    </section>
  );
}
