"use client";

import { GripVertical, Loader2, Pause, Play, Scissors } from "lucide-react";
import { PointerEvent, useEffect, useRef, useState } from "react";

const MIN_CLIP_SECONDS = 3;
const MAX_CLIP_SECONDS = 10;
const MIN_DRAG_SECONDS = 0.1;

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
  onClipReady: (clip: PreparedAudioClip | null) => void;
  onError: (message: string) => void;
};

type DragState = {
  mode: "start" | "end" | "move" | "new";
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
  const maxLength = Math.min(MAX_CLIP_SECONDS, duration);
  const minLength = Math.min(MIN_CLIP_SECONDS, duration);
  let start = clamp(Math.min(selection.start, selection.end), 0, duration);
  let end = clamp(Math.max(selection.start, selection.end), 0, duration);
  let length = end - start;

  if (length > maxLength) {
    end = start + maxLength;
    length = maxLength;
  }
  if (length < minLength) {
    end = Math.min(duration, start + minLength);
    start = Math.max(0, end - minLength);
  }
  return { start, end };
}

function encodeSelectionAsWav(buffer: AudioBuffer, selection: Selection) {
  const startFrame = Math.floor(selection.start * buffer.sampleRate);
  const endFrame = Math.min(buffer.length, Math.ceil(selection.end * buffer.sampleRate));
  const frameCount = Math.max(1, endFrame - startFrame);
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

export default function AudioClipSelector({ file, initialSelection, onClipReady, onError }: AudioClipSelectorProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const timelineRef = useRef<HTMLDivElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const stopTimerRef = useRef<number | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const objectUrlRef = useRef("");
  const [buffer, setBuffer] = useState<AudioBuffer | null>(null);
  const [selection, setSelection] = useState<Selection>({ start: 0, end: 0 });
  const [isLoading, setIsLoading] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setBuffer(null);
    onClipReady(null);

    const decode = async () => {
      const AudioContextClass = window.AudioContext;
      const context = new AudioContextClass();
      try {
        const decoded = await context.decodeAudioData(await file.arrayBuffer());
        if (cancelled) return;
        const nextSelection = normalizedSelection(
          initialSelection ?? { start: 0, end: Math.min(MAX_CLIP_SECONDS, decoded.duration) },
          decoded.duration,
        );
        setBuffer(decoded);
        setSelection(nextSelection);
        objectUrlRef.current = URL.createObjectURL(file);
        if (audioRef.current) audioRef.current.src = objectUrlRef.current;
      } catch {
        if (!cancelled) onError("Không đọc được file âm thanh này. Hãy thử WAV, MP3, M4A hoặc FLAC khác.");
      } finally {
        await context.close();
        if (!cancelled) setIsLoading(false);
      }
    };

    void decode();
    return () => {
      cancelled = true;
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = "";
    };
  }, [file, initialSelection, onClipReady, onError]);

  useEffect(() => {
    if (audioRef.current && objectUrlRef.current) audioRef.current.src = objectUrlRef.current;

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
    const timeout = window.setTimeout(() => {
      const blob = encodeSelectionAsWav(buffer, selection);
      const baseName = file.name.replace(/\.[^.]+$/, "") || "voice-reference";
      const preparedFile = new File([blob], `${baseName}-cut.wav`, { type: "audio/wav" });
      onClipReady({
        file: preparedFile,
        start: selection.start,
        end: selection.end,
        sourceDuration: buffer.duration,
      });
    }, 120);
    return () => window.clearTimeout(timeout);
  }, [buffer, file.name, onClipReady, selection]);

  useEffect(() => () => {
    if (stopTimerRef.current !== null) window.clearTimeout(stopTimerRef.current);
  }, []);

  const stopPreview = () => {
    if (stopTimerRef.current !== null) window.clearTimeout(stopTimerRef.current);
    stopTimerRef.current = null;
    audioRef.current?.pause();
    setIsPlaying(false);
  };

  const togglePreview = async () => {
    const audio = audioRef.current;
    if (!audio || !buffer) return;
    if (isPlaying) {
      stopPreview();
      return;
    }
    audio.currentTime = selection.start;
    try {
      await audio.play();
      setIsPlaying(true);
      stopTimerRef.current = window.setTimeout(stopPreview, Math.max(0, selection.end - selection.start) * 1000);
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
    if (!buffer) return;
    event.preventDefault();
    stopPreview();
    event.currentTarget.setPointerCapture(event.pointerId);
    const target = event.target as HTMLElement;
    const mode = (target.closest<HTMLElement>("[data-drag-mode]")?.dataset.dragMode || "new") as DragState["mode"];
    dragRef.current = { mode, anchorTime: timeFromPointer(event), initial: selection };
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || !buffer) return;
    const pointerTime = timeFromPointer(event);

    if (drag.mode === "move") {
      const length = drag.initial.end - drag.initial.start;
      const start = clamp(drag.initial.start + pointerTime - drag.anchorTime, 0, buffer.duration - length);
      setSelection({ start, end: start + length });
      return;
    }
    if (drag.mode === "start") {
      setSelection((current) => ({
        start: clamp(pointerTime, Math.max(0, current.end - MAX_CLIP_SECONDS), current.end - MIN_DRAG_SECONDS),
        end: current.end,
      }));
      return;
    }
    if (drag.mode === "end") {
      setSelection((current) => ({
        start: current.start,
        end: clamp(pointerTime, current.start + MIN_DRAG_SECONDS, Math.min(buffer.duration, current.start + MAX_CLIP_SECONDS)),
      }));
      return;
    }

    const start = Math.min(drag.anchorTime, pointerTime);
    const end = Math.max(drag.anchorTime, pointerTime);
    setSelection({ start, end: Math.min(buffer.duration, Math.min(end, start + MAX_CLIP_SECONDS)) });
  };

  const handlePointerUp = (event: PointerEvent<HTMLDivElement>) => {
    if (!dragRef.current || !buffer) return;
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

  if (!buffer) return null;

  const startPercent = (selection.start / buffer.duration) * 100;
  const endPercent = (selection.end / buffer.duration) * 100;
  const clipDuration = selection.end - selection.start;

  return (
    <section className="grid gap-2 rounded-md border border-blue-200 bg-white p-3" aria-label="Cắt đoạn giọng mẫu">
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="inline-flex items-center gap-1 font-semibold text-blue-800">
          <Scissors className="h-3.5 w-3.5" /> Cắt đoạn giọng phù hợp
        </span>
        <span className="tabular-nums text-slate-500">Tổng {formatTime(buffer.duration)}</span>
      </div>

      <div
        ref={timelineRef}
        className="relative h-24 touch-none select-none overflow-hidden rounded border border-slate-200 bg-slate-50 cursor-crosshair"
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
          className="absolute inset-y-0 cursor-grab border-y-2 border-blue-500 bg-blue-400/20 active:cursor-grabbing"
          style={{ left: `${startPercent}%`, width: `${endPercent - startPercent}%` }}
          title="Kéo để di chuyển cả đoạn đã chọn"
        >
          <div
            data-drag-mode="start"
            className="absolute inset-y-0 left-0 flex w-5 -translate-x-1/2 cursor-ew-resize items-center justify-center rounded-sm bg-blue-600 text-white shadow"
            title="Kéo điểm bắt đầu"
          >
            <GripVertical className="h-4 w-4" />
          </div>
          <div
            data-drag-mode="end"
            className="absolute inset-y-0 right-0 flex w-5 translate-x-1/2 cursor-ew-resize items-center justify-center rounded-sm bg-blue-600 text-white shadow"
            title="Kéo điểm kết thúc"
          >
            <GripVertical className="h-4 w-4" />
          </div>
        </div>
      </div>

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
          className="inline-flex items-center gap-1.5 rounded border border-blue-200 bg-blue-50 px-2.5 py-1.5 text-xs font-semibold text-blue-700 hover:bg-blue-100"
        >
          {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          {isPlaying ? "Dừng nghe" : "Nghe đoạn chọn"}
        </button>
      </div>
      <p className="text-[11px] leading-4 text-slate-500">
        Kéo hai tay nắm để cắt, hoặc kéo vùng màu xanh để đổi vị trí. Độ dài được giữ trong khoảng 3–10 giây khi file cho phép.
      </p>
      <audio ref={audioRef} onEnded={() => setIsPlaying(false)} className="hidden" />
    </section>
  );
}
