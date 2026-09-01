"use client";

import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, Download, FileVideo, Headphones, Loader2, Play, Scissors, Trash2, UploadCloud } from "lucide-react";
import AudioClipSelector, { PreparedAudioClip, VOICE_REFERENCE_SECONDS } from "./audio-clip-selector";
import { downloadUrlToDestination } from "@/lib/download-destination";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
const INSPECT_URL = `${BACKEND_URL}/api/v1/short-video/inspect`;
const ANALYZE_URL = `${BACKEND_URL}/api/v1/short-video/analyze`;
const DUB_URL = `${BACKEND_URL}/api/v1/short-video/dub`;
const RENDER_URL = `${BACKEND_URL}/api/v1/short-video/render-script`;
const VOICE_REFERENCE_URL = `${BACKEND_URL}/api/voice-reference`;

type ShortProfile = {
  name: "micro" | "short" | "short_extended" | "long";
  route: "short_video" | "clone_video";
  duration_seconds: number;
  max_short_seconds: number;
  asr_model: string;
  source_mode: "auto" | "voice" | "subtitle" | "hybrid";
  vocal_separation_default: boolean;
};

type Inspection = {
  media_id: string;
  filename: string;
  input_url: string;
  duration_seconds: number;
  has_audio: boolean;
  width?: number | null;
  height?: number | null;
  profile: ShortProfile;
};

type Segment = {
  id: number;
  start: number;
  end: number;
  original_text: string;
  translated_text: string;
  voice_model?: string;
  subtitle_style?: Record<string, unknown>;
  blur_style?: Record<string, unknown>;
};

type Props = { onOpenClone: () => void };

const SHORT_ASR_MODELS = [
  { label: "Tự động (base ≤60s / small >60s)", value: "auto" },
  { label: "Whisper tiny · nhanh", value: "tiny" },
  { label: "Whisper base · cân bằng", value: "base" },
  { label: "Whisper small · chính xác hơn", value: "small" },
  { label: "Whisper medium · nặng", value: "medium" },
  { label: "Paraformer · tiếng Trung", value: "paraformer-zh" },
];

const SHORT_TRANSLATION_MODELS = [
  { label: "Gemini 3 Flash Agent", value: "ag/gemini-3-flash-agent" },
  { label: "Gemini 3 Flash", value: "ag/gemini-3-flash" },
  { label: "Gemini 2.5 Flash", value: "gemini/gemini-2.5-flash" },
];

const SHORT_OCR_MODELS = [
  { label: "Gemini 2.5 Flash OCR", value: "gemini/gemini-2.5-flash" },
  { label: "Gemini 2.5 Pro OCR", value: "gemini/gemini-2.5-pro" },
];

function formatTime(seconds: number) {
  const safe = Math.max(0, seconds);
  return `${Math.floor(safe / 60)}:${Math.floor(safe % 60).toString().padStart(2, "0")}`;
}

function mediaUrl(value: string) {
  return value.startsWith("http") ? value : new URL(value, BACKEND_URL).toString();
}

async function errorMessage(response: Response) {
  const body = await response.text();
  try {
    const parsed = JSON.parse(body) as { detail?: string };
    return parsed.detail || body || `HTTP ${response.status}`;
  } catch {
    return body || `HTTP ${response.status}`;
  }
}

export default function ShortVideoWorkspace({ onOpenClone }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [segments, setSegments] = useState<Segment[]>([]);
  const [sourceLanguage, setSourceLanguage] = useState("auto");
  const [targetLanguage, setTargetLanguage] = useState("vi");
  const [asrEngine, setAsrEngine] = useState<"auto" | "whisper" | "paraformer">("auto");
  const [asrModel, setAsrModel] = useState("auto");
  const [whisperModel, setWhisperModel] = useState("auto");
  const [whisperBeamSize, setWhisperBeamSize] = useState(5);
  const [computeType, setComputeType] = useState<"int8" | "float16">("int8");
  const [translationProvider, setTranslationProvider] = useState<"9router" | "google" | "mock">("9router");
  const [translationModel, setTranslationModel] = useState(SHORT_TRANSLATION_MODELS[0].value);
  const [ocrModel, setOcrModel] = useState(SHORT_OCR_MODELS[0].value);
  const [ocrEnabled, setOcrEnabled] = useState(false);
  const [segmentLanguageDetection, setSegmentLanguageDetection] = useState(true);
  const [softTimingFit, setSoftTimingFit] = useState(true);
  const [timingMaxDrift, setTimingMaxDrift] = useState(0.35);
  const [timingMaxAtempo, setTimingMaxAtempo] = useState(1.08);
  const [voiceModel, setVoiceModel] = useState("Trúc Ly");
  const [voiceMode, setVoiceMode] = useState<"system" | "clone">("system");
  const [cloneReferenceFile, setCloneReferenceFile] = useState<File | null>(null);
  const [preparedCloneClip, setPreparedCloneClip] = useState<PreparedAudioClip | null>(null);
  const [cloneReferencePath, setCloneReferencePath] = useState("");
  const [copyrightConfirmed, setCopyrightConfirmed] = useState(false);
  const [copyrightSource, setCopyrightSource] = useState("owned");
  const [status, setStatus] = useState("Chưa có video");
  const [phase, setPhase] = useState("prepare");
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState<"inspect" | "analyze" | "dub" | "">("");
  const [resultUrl, setResultUrl] = useState("");
  const [error, setError] = useState("");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const auditStopTimerRef = useRef<number | null>(null);
  const cloneSelectionRevisionRef = useRef(0);

  useEffect(() => () => {
    if (auditStopTimerRef.current !== null) window.clearTimeout(auditStopTimerRef.current);
    if (previewUrl.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const handleCloneClipReady = useCallback((clip: PreparedAudioClip | null) => {
    cloneSelectionRevisionRef.current += 1;
    setPreparedCloneClip(clip);
    setCloneReferencePath("");
    if (clip) setError("");
  }, []);

  const handleCloneClipError = useCallback((message: string) => {
    setError(message);
  }, []);

  const canProcess = Boolean(
    inspection &&
      inspection.profile.route === "short_video" &&
      copyrightConfirmed &&
      (voiceMode === "system" || preparedCloneClip || cloneReferencePath),
  );
  const controlsLocked = Boolean(busy);
  const profileLabel = useMemo(() => {
    if (!inspection) return "Chưa phân loại";
    return `${inspection.profile.name} · ${formatTime(inspection.duration_seconds)} · ${inspection.profile.asr_model} int8`;
  }, [inspection]);

  function resetForFile(nextFile: File | null) {
    if (nextFile && !nextFile.type.startsWith("video/") && !/\.(mp4|mov|mkv|webm)$/i.test(nextFile.name)) {
      setError("Vui lòng chọn file video hợp lệ (MP4, MOV, MKV hoặc WEBM).");
      return;
    }
    if (previewUrl.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
    setFile(nextFile);
    setInspection(null);
    setSegments([]);
    setPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : "");
    setResultUrl("");
    setCloneReferencePath("");
    setError("");
    setProgress(0);
    setStatus(nextFile ? "Sẵn sàng kiểm tra video" : "Chưa có video");
  }

  function handleVideoDrop(event: React.DragEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
    if (controlsLocked) return;
    resetForFile(event.dataTransfer.files?.[0] || null);
  }

  function handleVideoDragOver(event: React.DragEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = controlsLocked ? "none" : "copy";
  }

  function updateSegment(id: number, patch: Partial<Segment>) {
    setSegments((current) => current.map((segment) => {
      if (segment.id !== id) return segment;
      const nextStart = patch.start === undefined ? segment.start : Math.max(0, Number(patch.start));
      const nextEnd = patch.end === undefined ? segment.end : Math.max(nextStart + 0.1, Number(patch.end));
      return { ...segment, ...patch, start: nextStart, end: nextEnd };
    }));
  }

  function splitText(text: string): [string, string] {
    const clean = text.trim();
    const words = clean.split(/\s+/).filter(Boolean);
    if (words.length > 1) {
      const pivot = Math.ceil(words.length / 2);
      return [words.slice(0, pivot).join(" "), words.slice(pivot).join(" ")];
    }
    const pivot = Math.max(1, Math.ceil(clean.length / 2));
    return [clean.slice(0, pivot), clean.slice(pivot)];
  }

  function splitSegment(id: number) {
    setSegments((current) => {
      const index = current.findIndex((segment) => segment.id === id);
      if (index < 0) return current;
      const segment = current[index];
      const midpoint = segment.start + (segment.end - segment.start) / 2;
      const [originalLeft, originalRight] = splitText(segment.original_text);
      const [translatedLeft, translatedRight] = splitText(segment.translated_text || segment.original_text);
      const left = { ...segment, start: segment.start, end: midpoint, original_text: originalLeft, translated_text: translatedLeft };
      const right = { ...segment, start: midpoint, end: segment.end, original_text: originalRight, translated_text: translatedRight };
      return [...current.slice(0, index), left, right, ...current.slice(index + 1)].map((item, itemIndex) => ({ ...item, id: itemIndex }));
    });
  }

  function removeSegment(id: number) {
    setSegments((current) => current.filter((segment) => segment.id !== id).map((segment, index) => ({ ...segment, id: index })));
  }

  function playAuditSegment(segment: Segment) {
    if (!videoRef.current) return;
    if (auditStopTimerRef.current !== null) window.clearTimeout(auditStopTimerRef.current);
    const player = videoRef.current;
    player.currentTime = Math.max(0, segment.start);
    void player.play();
    auditStopTimerRef.current = window.setTimeout(() => {
      player.pause();
      auditStopTimerRef.current = null;
    }, Math.max(100, (segment.end - segment.start) * 1000));
  }

  async function inspect() {
    if (!file) return;
    setBusy("inspect");
    setError("");
    setStatus("Đang probe media và phân loại Short Video...");
    try {
      const form = new FormData();
      form.append("video", file);
      const response = await fetch(INSPECT_URL, { method: "POST", body: form });
      if (!response.ok) throw new Error(await errorMessage(response));
      const payload = (await response.json()) as Inspection;
      setInspection(payload);
      if (previewUrl.startsWith("blob:")) URL.revokeObjectURL(previewUrl);
      setPreviewUrl(mediaUrl(payload.input_url));
      setStatus(payload.profile.route === "short_video" ? "Đã chọn pipeline Short Video" : "Video dài — nên chuyển Clone Video");
      setProgress(10);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Không inspect được video.");
      setStatus("Inspect thất bại");
    } finally {
      setBusy("");
    }
  }

  async function consumeStream(response: Response, operation: "analyze" | "dub") {
    if (!response.ok) throw new Error(await errorMessage(response));
    if (!response.body) throw new Error("Backend không trả event stream.");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const consumeEvent = (event: string) => {
      for (const line of event.split(/\r?\n/)) {
        if (!line.startsWith("data:")) continue;
        const raw = line.replace(/^data:\s*/, "").trim();
        if (!raw || raw === "[DONE]") continue;
        const payload = JSON.parse(raw) as Record<string, unknown>;
        if (typeof payload.error === "string") throw new Error(payload.error);
        if (typeof payload.phase === "string") setPhase(payload.phase);
        if (typeof payload.progress === "number") setProgress(Math.max(0, Math.min(100, Math.round(payload.progress))));
        if (typeof payload.step === "string") setStatus(String(payload.step));
        if (Array.isArray(payload.segments)) setSegments(payload.segments as Segment[]);
        if (typeof payload.video_url === "string") setResultUrl(mediaUrl(payload.video_url));
        if (payload.status === "success") {
          setProgress(100);
          setStatus(operation === "dub" ? "Hoàn tất — có thể tải video" : "Đã tạo timeline để kiểm tra");
        }
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop() || "";
      events.forEach(consumeEvent);
    }
    if (buffer.trim()) consumeEvent(buffer);
  }

  function makeForm(clonePathOverride = cloneReferencePath) {
    if (!inspection) throw new Error("Chưa có media profile.");
    const form = new FormData();
    form.append("media_id", inspection.media_id);
    form.append("source_language", sourceLanguage);
    form.append("target_language", targetLanguage);
    form.append("asr_engine", asrEngine);
    form.append("asr_model", asrModel);
    form.append("whisper_model", whisperModel);
    form.append("whisper_beam_size", String(whisperBeamSize));
    form.append("compute_type", computeType);
    form.append("translation_provider", translationProvider);
    form.append("translation_model", translationModel);
    form.append("voice_model", voiceModel);
    form.append("voice_mode", voiceMode);
    if (clonePathOverride) form.append("clone_reference_audio_path", clonePathOverride);
    form.append("ocr_fallback", String(ocrEnabled));
    form.append("ocr_force", "false");
    form.append("ocr_interval_seconds", "0.5");
    form.append("ocr_model", ocrModel);
    form.append("segment_language_detection", String(segmentLanguageDetection));
    form.append("translate_batch_size", "40");
    form.append("translate_analysis", "true");
    form.append("translate_review", "true");
    form.append("translate_cps_budget", "12.5");
    form.append("soft_timing_fit", String(softTimingFit));
    form.append("timing_max_drift_s", String(timingMaxDrift));
    form.append("timing_min_gap_s", "0.08");
    form.append("timing_max_atempo", String(timingMaxAtempo));
    form.append("voice_speed", "1.0");
    form.append("vocal_separation", "false");
    form.append("copyright_confirmed", String(copyrightConfirmed));
    form.append("copyright_source", copyrightSource);
    form.append("copyright_notes", "Short Video workflow");
    return form;
  }

  function assertCloneSelectionUnchanged(expectedRevision: number) {
    if (cloneSelectionRevisionRef.current !== expectedRevision) {
      throw new Error("Đoạn giọng mẫu đã thay đổi trong lúc tải lên. Hãy chạy lại với đoạn 3 giây mới.");
    }
  }

  async function uploadCloneReference(expectedRevision: number) {
    if (voiceMode === "system") return "";
    assertCloneSelectionUnchanged(expectedRevision);
    if (cloneReferencePath) return cloneReferencePath;
    if (!preparedCloneClip) throw new Error("Hãy chọn và chuẩn bị đúng đoạn giọng mẫu 3 giây để clone.");
    const form = new FormData();
    form.append("audio", preparedCloneClip.file);
    form.append("clip_duration_seconds", String(VOICE_REFERENCE_SECONDS));
    form.append("selection_start_seconds", preparedCloneClip.start.toFixed(6));
    form.append("selection_end_seconds", preparedCloneClip.end.toFixed(6));
    const response = await fetch(VOICE_REFERENCE_URL, { method: "POST", body: form });
    if (!response.ok) throw new Error(await errorMessage(response));
    const payload = (await response.json()) as { path?: string };
    if (!payload.path) throw new Error("Backend không trả đường dẫn giọng mẫu.");
    assertCloneSelectionUnchanged(expectedRevision);
    setCloneReferencePath(payload.path);
    return payload.path;
  }

  async function analyze() {
    if (!canProcess) return;
    setBusy("analyze");
    setError("");
    setStatus("Đang nhận dạng thoại và subtitle...");
    setPhase("recognize");
    try {
      const cloneSelectionRevision = cloneSelectionRevisionRef.current;
      const selectedClonePath = await uploadCloneReference(cloneSelectionRevision);
      assertCloneSelectionUnchanged(cloneSelectionRevision);
      await consumeStream(await fetch(ANALYZE_URL, { method: "POST", body: makeForm(selectedClonePath), headers: { Accept: "text/event-stream" } }), "analyze");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Analyze thất bại.");
      setStatus("Analyze thất bại");
    } finally {
      setBusy("");
    }
  }

  async function dub() {
    if (!canProcess) return;
    setBusy("dub");
    setError("");
    setResultUrl("");
    setStatus("Đang dịch, lồng tiếng và render Short Video...");
    try {
      const cloneSelectionRevision = cloneSelectionRevisionRef.current;
      const selectedClonePath = await uploadCloneReference(cloneSelectionRevision);
      assertCloneSelectionUnchanged(cloneSelectionRevision);
      if (segments.length > 0) {
        const renderSegments = segments.map((segment) => ({
          ...segment,
          voice_model: segment.voice_model || voiceModel,
          subtitle_style: segment.subtitle_style || { x: 12, y: 72, width: 76, height: 16, font_size: 42, color: "#FFFFFF", outline_color: "#000000", outline_width: 3, align: "center" },
          blur_style: segment.blur_style || { enabled: false, x: 18, y: 10, width: 44, height: 18, blur: 16, opacity: 0.26 },
        }));
        const response = await fetch(RENDER_URL, {
          method: "POST",
          headers: { Accept: "text/event-stream", "Content-Type": "application/json" },
          body: JSON.stringify({
            media_id: inspection?.media_id,
            source_language: sourceLanguage === "auto" ? null : sourceLanguage,
            target_language: targetLanguage,
            translation_provider: translationProvider,
            translation_model: translationModel,
            voice_model: voiceModel,
            voice_mode: voiceMode,
            clone_reference_audio_path: selectedClonePath || null,
            copyright_confirmed: copyrightConfirmed,
            copyright_source: copyrightSource,
            copyright_notes: "Short Video workflow",
            vocal_separation: false,
            ocr_fallback: ocrEnabled,
            ocr_force: false,
            ocr_model: ocrModel,
            ocr_interval_seconds: 0.5,
            ocr_crop_bottom_ratio: 0.35,
            asr_engine: asrEngine,
            whisper_model: whisperModel,
            whisper_beam_size: whisperBeamSize,
            segment_language_detection: segmentLanguageDetection,
            translate_batch_size: 40,
            translate_analysis: true,
            translate_review: true,
            translate_cps_budget: 12.5,
            soft_timing_fit: softTimingFit,
            timing_max_drift_s: timingMaxDrift,
            timing_min_gap_s: 0.08,
            timing_max_atempo: timingMaxAtempo,
            voice_speed: 1.0,
            segments: renderSegments,
          }),
        });
        await consumeStream(response, "dub");
      } else {
        await consumeStream(await fetch(DUB_URL, { method: "POST", body: makeForm(selectedClonePath), headers: { Accept: "text/event-stream" } }), "dub");
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Dubbing thất bại.");
      setStatus("Dubbing thất bại");
    } finally {
      setBusy("");
    }
  }

  async function downloadResult() {
    if (!resultUrl) return;
    try {
      const { savedInFolder } = await downloadUrlToDestination(resultUrl, "video", "short_video.mp4");
      if (savedInFolder) {
        setStatus("Đã lưu video vào thư mục đã cấu hình.");
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Không tải được video.");
    }
  }

  return (
    <main
      className="grid min-h-[calc(100vh-64px)] min-w-0 gap-0 lg:grid-cols-[360px_minmax(0,1fr)]"
      aria-label="Short Video workspace"
      onDragOver={handleVideoDragOver}
      onDrop={handleVideoDrop}
    >
      <aside className="min-w-0 border-r border-slate-200 bg-white p-5">
        <header className="flex items-center justify-between">
          <h2 className="text-xs font-bold uppercase tracking-wide text-slate-500">Short Video</h2>
          <span className="rounded bg-blue-50 px-2 py-1 text-xs font-semibold text-blue-700">≤ 2 phút</span>
        </header>
        <figure
          className="relative mt-4 aspect-[9/14] overflow-hidden rounded-lg border border-dashed border-blue-200 bg-slate-50"
          onDragOver={handleVideoDragOver}
          onDrop={handleVideoDrop}
        >
          {previewUrl ? (
            <div className="relative h-full w-full">
              <video ref={videoRef} className="h-full w-full object-cover" src={previewUrl} controls />
              <label className={`absolute inset-x-3 bottom-3 rounded-md bg-slate-950/75 px-3 py-2 text-center text-xs font-semibold text-white ${controlsLocked ? "cursor-not-allowed opacity-60" : "cursor-pointer"}`}>
                Thả video mới vào đây hoặc chọn file
                <input className="hidden" type="file" accept="video/*,.mkv" disabled={controlsLocked} onChange={(event: ChangeEvent<HTMLInputElement>) => resetForFile(event.target.files?.[0] || null)} />
              </label>
            </div>
          ) : (
            <label className={`flex h-full flex-col items-center justify-center p-8 text-center text-slate-600 ${controlsLocked ? "cursor-not-allowed opacity-60" : "cursor-pointer"}`}>
              <UploadCloud className="mb-3 text-blue-600" size={34} />
              <span className="text-sm font-semibold">Thả video hoặc chọn file</span>
              <span className="mt-2 text-xs">MP4, MOV, MKV, WEBM · tối đa 2 phút</span>
              <input className="hidden" type="file" accept="video/*,.mkv" disabled={controlsLocked} onChange={(event: ChangeEvent<HTMLInputElement>) => resetForFile(event.target.files?.[0] || null)} />
            </label>
          )}
        </figure>
        {file && <p className="mt-3 flex items-center justify-between gap-2 text-sm font-semibold"><span className="truncate">{file.name}</span><label className={`shrink-0 text-xs font-semibold text-blue-700 ${controlsLocked ? "cursor-not-allowed opacity-60" : "cursor-pointer"}`}>Đổi<input className="hidden" type="file" accept="video/*,.mkv" disabled={controlsLocked} onChange={(event: ChangeEvent<HTMLInputElement>) => resetForFile(event.target.files?.[0] || null)} /></label></p>}
        <button type="button" disabled={!file || Boolean(busy)} onClick={() => void inspect()} className="mt-4 inline-flex h-10 w-full items-center justify-center gap-2 rounded-md border border-blue-200 bg-blue-50 text-sm font-semibold text-blue-700 disabled:opacity-50">
          {busy === "inspect" ? <Loader2 className="animate-spin" size={16} /> : <FileVideo size={16} />} Kiểm tra & phân loại
        </button>
        <fieldset className="mt-5 grid gap-3 border-t border-slate-200 pt-5">
          <legend className="text-sm font-semibold text-slate-700">Cấu hình nhanh</legend>
          <label className="grid gap-1 text-sm font-medium">Ngôn ngữ gốc<select value={sourceLanguage} disabled={controlsLocked} onChange={(event) => setSourceLanguage(event.target.value)} className="rounded border border-slate-200 bg-white px-2 py-2 disabled:cursor-not-allowed disabled:opacity-60"><option value="auto">Tự động</option><option value="zh">Tiếng Trung</option><option value="en">Tiếng Anh</option><option value="vi">Tiếng Việt</option><option value="ja">Tiếng Nhật</option><option value="ko">Tiếng Hàn</option></select></label>
           <label className="grid gap-1 text-sm font-medium">Ngôn ngữ đích<select value={targetLanguage} disabled={controlsLocked} onChange={(event) => setTargetLanguage(event.target.value)} className="rounded border border-slate-200 bg-white px-2 py-2 disabled:cursor-not-allowed disabled:opacity-60"><option value="vi">Tiếng Việt</option><option value="en">Tiếng Anh</option><option value="zh">Tiếng Trung</option></select></label>
           <details open className="rounded border border-violet-200 bg-violet-50/40 p-3">
             <summary className="cursor-pointer text-sm font-semibold text-violet-800">Thiết lập model Short Video</summary>
             <div className="mt-3 grid gap-3 text-sm">
               <label className="grid gap-1 font-medium">Engine ASR<select value={asrEngine} disabled={controlsLocked} onChange={(event) => setAsrEngine(event.target.value as typeof asrEngine)} className="rounded border border-violet-200 bg-white px-2 py-2"><option value="auto">Tự động</option><option value="whisper">Whisper</option><option value="paraformer">Paraformer</option></select></label>
               <label className="grid gap-1 font-medium">Model ASR<select value={asrModel} disabled={controlsLocked} onChange={(event) => setAsrModel(event.target.value)} className="rounded border border-violet-200 bg-white px-2 py-2">{SHORT_ASR_MODELS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
               <div className="grid grid-cols-2 gap-2">
                 <label className="grid gap-1 text-xs font-medium">Whisper model<select value={whisperModel} disabled={controlsLocked} onChange={(event) => setWhisperModel(event.target.value)} className="rounded border border-violet-200 bg-white px-2 py-2 text-xs"><option value="auto">Theo ASR</option><option value="tiny">tiny</option><option value="base">base</option><option value="small">small</option><option value="medium">medium</option><option value="large-v3">large-v3</option></select></label>
                 <label className="grid gap-1 text-xs font-medium">Beam size<select value={whisperBeamSize} disabled={controlsLocked} onChange={(event) => setWhisperBeamSize(Number(event.target.value))} className="rounded border border-violet-200 bg-white px-2 py-2 text-xs">{[1, 3, 5, 7, 10].map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
                 <label className="grid gap-1 text-xs font-medium">Compute<select value={computeType} disabled={controlsLocked} onChange={(event) => setComputeType(event.target.value as typeof computeType)} className="rounded border border-violet-200 bg-white px-2 py-2 text-xs"><option value="int8">int8 · tiết kiệm</option><option value="float16">float16 · GPU</option></select></label>
               </div>
               <label className="grid gap-1 font-medium">Model dịch<select value={translationModel} disabled={controlsLocked} onChange={(event) => setTranslationModel(event.target.value)} className="rounded border border-violet-200 bg-white px-2 py-2">{SHORT_TRANSLATION_MODELS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
               <label className="grid gap-1 text-xs font-medium">Provider dịch<select value={translationProvider} disabled={controlsLocked} onChange={(event) => setTranslationProvider(event.target.value as typeof translationProvider)} className="rounded border border-violet-200 bg-white px-2 py-2"><option value="9router">9Router</option><option value="google">Google</option><option value="mock">Mock (test)</option></select></label>
               <label className="flex items-center gap-2 text-xs font-medium"><input type="checkbox" checked={ocrEnabled} disabled={controlsLocked} onChange={(event) => setOcrEnabled(event.target.checked)} /> Bật OCR phụ đề (tắt nếu video chỉ có giọng)</label>
               <label className={`grid gap-1 font-medium ${!ocrEnabled ? "opacity-50" : ""}`}>Model OCR<select value={ocrModel} disabled={controlsLocked || !ocrEnabled} onChange={(event) => setOcrModel(event.target.value)} className="rounded border border-violet-200 bg-white px-2 py-2">{SHORT_OCR_MODELS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label>
               <fieldset className="grid gap-2 rounded border border-emerald-200 bg-emerald-50/50 p-2">
                 <legend className="px-1 text-xs font-semibold text-emerald-800">Khớp giọng gốc theo từng đoạn</legend>
                 <label className="flex items-center gap-2 text-xs font-medium"><input type="checkbox" checked={softTimingFit} disabled={controlsLocked} onChange={(event) => setSoftTimingFit(event.target.checked)} /> Fit audio vào đúng khoảng thoại gốc</label>
                 <label className="grid gap-1 text-xs font-medium">Độ lệch cho phép (giây)<input type="number" min="0" max="2" step="0.05" value={timingMaxDrift} disabled={controlsLocked} onChange={(event) => setTimingMaxDrift(Math.max(0, Math.min(2, Number(event.target.value) || 0)))} className="rounded border border-emerald-200 bg-white px-2 py-1" /></label>
                 <label className="grid gap-1 text-xs font-medium">Nén tốc độ tối đa<select value={timingMaxAtempo} disabled={controlsLocked} onChange={(event) => setTimingMaxAtempo(Number(event.target.value))} className="rounded border border-emerald-200 bg-white px-2 py-1"><option value="1.05">1.05x · tự nhiên</option><option value="1.08">1.08x · cân bằng</option><option value="1.1">1.10x · linh hoạt</option></select></label>
               </fieldset>
               <label className="flex items-center gap-2 text-xs font-medium"><input type="checkbox" checked={segmentLanguageDetection} disabled={controlsLocked} onChange={(event) => setSegmentLanguageDetection(event.target.checked)} /> Nhận diện ngôn ngữ theo từng segment</label>
               <p className="text-[11px] leading-4 text-violet-700">Short ≤2 phút dùng timestamp chi tiết, không gộp segment/TTS. Có thể chọn Paraformer khi nguồn là tiếng Trung.</p>
             </div>
           </details>
           <fieldset className="grid gap-2 rounded border border-blue-100 bg-blue-50/40 p-3">
            <legend className="text-sm font-semibold text-blue-800">Nguồn giọng TTS</legend>
            <div className="grid grid-cols-2 gap-2 text-xs font-semibold">
              <label className={`rounded border bg-white px-2 py-2 ${controlsLocked ? "cursor-not-allowed opacity-60" : ""}`}><input className="mr-1" type="radio" checked={voiceMode === "system"} disabled={controlsLocked} onChange={() => setVoiceMode("system")} />Hệ thống</label>
              <label className={`rounded border bg-white px-2 py-2 ${controlsLocked ? "cursor-not-allowed opacity-60" : ""}`}><input className="mr-1" type="radio" checked={voiceMode === "clone"} disabled={controlsLocked} onChange={() => setVoiceMode("clone")} />Clone giọng</label>
            </div>
            {voiceMode === "system" ? (
              <label className="grid gap-1 text-sm font-medium">Giọng hệ thống<select value={voiceModel} disabled={controlsLocked} onChange={(event) => setVoiceModel(event.target.value)} className="rounded border border-slate-200 bg-white px-2 py-2 disabled:cursor-not-allowed disabled:opacity-60"><option>Trúc Ly</option><option>Ngọc Linh</option><option>Phạm Tuyên</option></select></label>
            ) : (
              <div className="grid gap-2 text-xs font-medium text-slate-700">
                <label className="grid gap-1">
                  File giọng mẫu (chọn file dài ít nhất 3 giây)
                  <input
                    type="file"
                    accept="audio/*,.wav,.flac,.mp3,.m4a,.ogg,.opus"
                    disabled={controlsLocked}
                    onChange={(event) => {
                      cloneSelectionRevisionRef.current += 1;
                      setCloneReferenceFile(event.target.files?.[0] || null);
                      setPreparedCloneClip(null);
                      setCloneReferencePath("");
                      setError("");
                    }}
                    className="rounded border border-blue-200 bg-white px-2 py-2 text-xs disabled:cursor-not-allowed disabled:opacity-60"
                  />
                </label>
                {cloneReferenceFile && (
                  <AudioClipSelector
                    file={cloneReferenceFile}
                    disabled={controlsLocked}
                    onClipReady={handleCloneClipReady}
                    onError={handleCloneClipError}
                  />
                )}
                <span className={cloneReferencePath ? "text-emerald-700" : "text-slate-500"}>
                  {cloneReferencePath
                    ? "Backend đã nhận đúng clip 3 giây."
                    : preparedCloneClip
                      ? `Sẽ chỉ gửi đoạn ${preparedCloneClip.start.toFixed(2)}–${preparedCloneClip.end.toFixed(2)} giây; không gửi toàn bộ file.`
                      : "Kéo khung xanh để chọn đúng đoạn giọng muốn clone."}
                </span>
              </div>
            )}
          </fieldset>
          <label className="grid gap-1 text-sm font-medium">Nguồn quyền<select value={copyrightSource} disabled={controlsLocked} onChange={(event) => setCopyrightSource(event.target.value)} className="rounded border border-slate-200 bg-white px-2 py-2 disabled:cursor-not-allowed disabled:opacity-60"><option value="owned">Tự sở hữu / tự quay</option><option value="licensed">Đã mua license</option><option value="permission">Được cho phép</option><option value="public_domain">Public domain</option></select></label>
          <label className={`flex items-start gap-2 text-xs text-slate-600 ${controlsLocked ? "cursor-not-allowed opacity-60" : ""}`}><input type="checkbox" checked={copyrightConfirmed} disabled={controlsLocked} onChange={(event) => setCopyrightConfirmed(event.target.checked)} className="mt-0.5" /> Tôi xác nhận có quyền sử dụng nội dung.</label>
        </fieldset>
      </aside>
      <section className="min-w-0 p-6">
        <nav aria-label="Short Video checkpoints"><ol className="flex flex-wrap gap-3 text-sm font-semibold text-slate-500"><li className={phase === "prepare" ? "text-blue-700" : ""}>1. Media</li><li className={phase === "recognize" ? "text-blue-700" : ""}>2. Nhận dạng</li><li className={phase === "translate" ? "text-blue-700" : ""}>3. Dịch</li><li className={phase === "voice" ? "text-blue-700" : ""}>4. Lồng tiếng</li><li className={phase === "render" || progress === 100 ? "text-blue-700" : ""}>5. Xuất bản</li></ol></nav>
        <header className="mt-5 flex items-end justify-between gap-3"><hgroup><p className="text-xs font-bold uppercase tracking-wide text-blue-700">Pipeline bổ sung</p><h1 className="mt-1 text-2xl font-semibold">Short Video dubbing</h1><p className="mt-1 text-sm text-slate-600">{profileLabel}</p></hgroup>{inspection?.profile.route === "clone_video" && <button type="button" disabled={controlsLocked} onClick={onOpenClone} className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm font-semibold text-amber-800 disabled:cursor-not-allowed disabled:opacity-60">Mở Clone Video</button>}</header>
        <section className="mt-4 rounded-lg border border-blue-100 bg-white p-4 shadow-sm" aria-live="polite"><header className="flex items-center justify-between"><h2 className="font-semibold">{status}</h2><output className="font-bold text-blue-700">{progress}%</output></header><progress className="mt-3 h-2 w-full" max={100} value={progress} />{error && <p className="mt-3 rounded bg-red-50 p-3 text-sm text-red-700">{error}</p>}</section>
        <section className="mt-4 rounded-lg border border-slate-200 bg-white p-4" aria-labelledby="short-script-title">
          <header className="flex flex-wrap items-center justify-between gap-2">
            <div><h2 id="short-script-title" className="font-semibold">Review timeline & audit giọng gốc</h2><p className="mt-1 text-xs text-slate-500">Sửa text, chỉnh mốc thời gian, nghe từng đoạn hoặc chia nhỏ trước khi lồng tiếng.</p></div>
            <span className="text-sm font-semibold text-slate-500">{segments.length} đoạn thoại</span>
          </header>
          {segments.length ? (
            <ol className="mt-3 grid gap-3">
              {segments.map((segment, index) => (
                <li key={segment.id} className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <header className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-xs font-bold uppercase tracking-wide text-slate-500">Đoạn {index + 1} · {formatTime(segment.start)}–{formatTime(segment.end)}</span>
                    <div className="flex flex-wrap gap-1">
                      <button type="button" disabled={Boolean(busy)} onClick={() => playAuditSegment(segment)} className="inline-flex items-center gap-1 rounded border border-blue-200 bg-white px-2 py-1 text-xs font-semibold text-blue-700 disabled:opacity-50"><Headphones size={13} /> Nghe</button>
                      <button type="button" disabled={Boolean(busy) || segment.end - segment.start < 0.3} onClick={() => splitSegment(segment.id)} className="inline-flex items-center gap-1 rounded border border-amber-200 bg-white px-2 py-1 text-xs font-semibold text-amber-700 disabled:opacity-50"><Scissors size={13} /> Chia đôi</button>
                      <button type="button" disabled={Boolean(busy) || segments.length <= 1} onClick={() => removeSegment(segment.id)} className="inline-flex items-center gap-1 rounded border border-red-200 bg-white px-2 py-1 text-xs font-semibold text-red-700 disabled:opacity-50"><Trash2 size={13} /> Xóa</button>
                    </div>
                  </header>
                  <div className="mt-2 grid gap-2 md:grid-cols-[120px_120px_minmax(0,1fr)]">
                    <label className="grid gap-1 text-xs font-medium">Bắt đầu<input type="number" min="0" step="0.01" value={segment.start.toFixed(2)} disabled={Boolean(busy)} onChange={(event) => updateSegment(segment.id, { start: Number(event.target.value) || 0 })} className="rounded border border-slate-300 bg-white px-2 py-1.5" /></label>
                    <label className="grid gap-1 text-xs font-medium">Kết thúc<input type="number" min="0.1" step="0.01" value={segment.end.toFixed(2)} disabled={Boolean(busy)} onChange={(event) => updateSegment(segment.id, { end: Number(event.target.value) || segment.start + 0.1 })} className="rounded border border-slate-300 bg-white px-2 py-1.5" /></label>
                    <label className="grid gap-1 text-xs font-medium">Thời lượng fit<span className="rounded border border-emerald-200 bg-emerald-50 px-2 py-1.5 text-emerald-800">{Math.max(0.1, segment.end - segment.start).toFixed(2)} giây · fit từng đoạn</span></label>
                  </div>
                  <div className="mt-2 grid gap-2 md:grid-cols-2">
                    <label className="grid gap-1 text-xs font-medium text-rose-700">Voice gốc<textarea value={segment.original_text} disabled={Boolean(busy)} onChange={(event) => updateSegment(segment.id, { original_text: event.target.value })} rows={2} className="resize-y rounded border border-rose-200 bg-rose-50/40 px-2 py-1.5 text-sm text-slate-900" /></label>
                    <label className="grid gap-1 text-xs font-medium text-blue-700">Bản dịch sẽ đọc<textarea value={segment.translated_text} disabled={Boolean(busy)} onChange={(event) => updateSegment(segment.id, { translated_text: event.target.value })} rows={2} className="resize-y rounded border border-blue-200 bg-blue-50/40 px-2 py-1.5 text-sm text-slate-900" /></label>
                  </div>
                </li>
              ))}
            </ol>
          ) : <p className="mt-3 rounded bg-slate-50 p-3 text-sm text-slate-500">Chạy “Nhận dạng & dịch” trước. Sau đó review từng đoạn rồi mới lồng tiếng.</p>}
        </section>
        <menu className="mt-4 flex flex-wrap gap-3"><button type="button" disabled={!canProcess || Boolean(busy)} onClick={() => void analyze()} className="inline-flex h-10 items-center gap-2 rounded-md border border-blue-200 bg-blue-50 px-4 text-sm font-semibold text-blue-700 disabled:opacity-50">{busy === "analyze" ? <Loader2 className="animate-spin" size={16} /> : <CheckCircle2 size={16} />} Nhận dạng & dịch</button><button type="button" disabled={!canProcess || segments.length === 0 || Boolean(busy)} onClick={() => void dub()} className="inline-flex h-10 items-center gap-2 rounded-md bg-blue-600 px-4 text-sm font-semibold text-white disabled:bg-slate-300">{busy === "dub" ? <Loader2 className="animate-spin" size={16} /> : <Play size={16} />} Lồng tiếng timeline đã duyệt</button>{resultUrl && <button type="button" onClick={() => void downloadResult()} className="inline-flex h-10 items-center gap-2 rounded-md bg-slate-950 px-4 text-sm font-semibold text-white"><Download size={16} /> Tải video</button>}</menu>
      </section>
    </main>
  );
}
