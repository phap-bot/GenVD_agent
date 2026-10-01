"use client";

import {
  AlertCircle,
  CheckCircle2,
  Clock3,
  Copy,
  Download,
  FileVideo,
  Languages,
  Loader2,
  Maximize2,
  Mic2,
  Move,
  Play,
  RefreshCw,
  Settings,
  Type,
  UploadCloud,
  Video,
  Wand2,
  X,
} from "lucide-react";
import { ChangeEvent, DragEvent, PointerEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import AudioClipSelector, { PreparedAudioClip, VOICE_REFERENCE_SECONDS } from "./audio-clip-selector";
import GenVideoPipeline from "./genvideo-pipeline";
import ShortVideoWorkspace from "./short-video-workspace";
import DownloadSetup from "./download-setup";
import { deleteSessionFiles, readSessionFile, writeSessionFile } from '@/lib/session-files';
import { downloadUrlToDestination } from '@/lib/download-destination';

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
const ANALYZE_URL = `${BACKEND_URL}/api/analyze-stream`;
const RENDER_SCRIPT_URL = `${BACKEND_URL}/api/render-script`;
const RENDER_SCRIPT_UPLOAD_URL = `${BACKEND_URL}/api/render-script-upload`;
const RENDER_JOB_URL = `${BACKEND_URL}/api/render-jobs`;
const CANCEL_URL = `${BACKEND_URL}/api/cancel`;
const VOICE_REFERENCE_URL = `${BACKEND_URL}/api/voice-reference`;
const SHORTEN_TEXT_URL = `${BACKEND_URL}/api/shorten-text`;
const TRANSLATION_MODELS_URL = `${BACKEND_URL}/api/translation/models`;
const STT_MODELS_URL = `${BACKEND_URL}/api/stt/models`;
const PIPELINE_SETTINGS_URL = `${BACKEND_URL}/api/pipeline/settings`;
const DEFAULT_TRANSLATION_MODEL = "ag/gemini-3-flash-agent";
const DEFAULT_OCR_MODEL = "gemini/gemini-2.5-flash";
const DEFAULT_ASR_MODEL = "base";
const VI_WORDS_PER_SECOND = 3;
const STUDIO_SESSION_KEY = 'video-clone:studio-session:v1';
const STUDIO_SESSION_VERSION = 4;

type CopyrightSource = "unknown" | "owned" | "licensed" | "public_domain" | "permission" | "platform_library";
type VoiceMode = "system" | "clone";

type StudioConfig = {
  sourceLanguage: string;
  targetLanguage: string;
  translationProvider: "9router" | "google" | "mock";
  translationModel: string;
  asrModel: string;
  asrEngine: "auto" | "whisper" | "paraformer";
  whisperModel: string;
  whisperBeamSize: number;
  segmentLanguageDetection: boolean;
  voiceModel: string;
  voiceMode: VoiceMode;
  ttsDevice: "cuda";
  copyrightAcknowledged: boolean;
  copyrightSource: CopyrightSource;
  copyrightNotes: string;
  ocrFallback: boolean;
  ocrForce: boolean;
  ocrModel: string;
  ocrIntervalSeconds: number;
  ocrCropBottomRatio: number;
  flashTextEnabled: boolean;
  flashTextMode: "balanced" | "strict";
  flashTextMinConfidence: number;
  flashTextMaxDuration: number;
  vocalSeparation: boolean;
  originalVocalGain: number;
  accompanimentGain: number;
  softTimingFit: boolean;
  timingMaxDrift: number;
  timingMinGap: number;
  timingMaxAtempo: number;
  voiceSpeed: number;
};

type ModelOption = {
  label: string;
  value: string;
};

const copyrightSourceOptions: ModelOption[] = [
  { label: "Chưa xác định", value: "unknown" },
  { label: "Tự sở hữu / tự quay", value: "owned" },
  { label: "Đã mua license", value: "licensed" },
  { label: "Public domain", value: "public_domain" },
  { label: "Được chủ sở hữu cho phép", value: "permission" },
  { label: "Thư viện âm thanh/video của nền tảng", value: "platform_library" },
];

type SubtitleStyle = {
  x: number;
  y: number;
  width: number;
  height: number;
  font_size: number;
  color: string;
  outline_color: string;
  outline_width: number;
  align: "left" | "center" | "right";
};

type BlurStyle = {
  enabled: boolean;
  x: number;
  y: number;
  width: number;
  height: number;
  blur: number;
  opacity: number;
};

type FlashTextBox = {
  timestamp: number;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
};

type FlashTextTrack = {
  id: number;
  start: number;
  end: number;
  confidence: number;
  enabled: boolean;
  boxes: FlashTextBox[];
  blur: number;
  padding: number;
  feather: number;
};

type ScriptSegment = {
  id: number;
  start: number;
  end: number;
  original_text: string;
  translated_text: string;
  voice_model: string;
  subtitle_style?: SubtitleStyle;
  blur_style?: BlurStyle;
};

type ToastState = {
  type: "error" | "success" | "info";
  message: string;
} | null;

type ProcessingStats = {
  segments?: number;
  groups?: number;
  speech_duration?: number;
  video_duration?: number;
  characters?: number;
  chunks?: number;
  segment_id?: number;
  engine?: string;
};

type ProcessingActivity = {
  id: number;
  key: string;
  phase: string;
  label: string;
  detail: string;
  progress: number;
  stats: ProcessingStats | null;
};

type StudioSessionSnapshot = {
  version: number;
  sessionId: string;
  activeWorkspace: 'clone' | 'short' | 'gen' | 'setup';
  sourceVideoUrl: string;
  segments: ScriptSegment[];
  captionSuggestions: string[];
  flashTextTracks: FlashTextTrack[];
  config: StudioConfig;
  activeStep: number;
  selectedSegmentId: number | null;
  progress: number;
  statusText: string;
  processingPhase: string;
  processingDetail: string;
  processingStats: ProcessingStats | null;
  processingActivities: ProcessingActivity[];
  resultVideoUrl: string;
  resultSubtitleUrl: string;
  renderJobId: string;
  textLayerEnabled: boolean;
  savedDemoSegments: ScriptSegment[] | null;
  savedTextLayerEnabled: boolean | null;
  cloneClipSelection: Pick<PreparedAudioClip, 'start' | 'end' | 'sourceDuration'> | null;
};

type SessionSaveStatus = 'loading' | 'saved' | 'error';

const workflowSteps = ["Video", "Nhận dạng", "Dịch thuật", "Lồng tiếng", "Xuất bản"];

const languageOptions = [
  { label: "Tự động nhận dạng", value: "auto" },
  { label: "Tiếng Anh", value: "en" },
  { label: "Tiếng Việt", value: "vi" },
  { label: "Tiếng Trung", value: "zh" },
  { label: "Tiếng Nhật", value: "ja" },
  { label: "Tiếng Hàn", value: "ko" },
];

const targetLanguageOptions = languageOptions.filter((item) => item.value !== "auto");

const fallbackTranslationModels: ModelOption[] = [
  { label: "OpenCode DeepSeek V4 Flash Free", value: "oc/deepseek-v4-flash-free" },
  { label: "OpenCode MiMo 2.5 Free", value: "oc/mimo-v2.5-free" },
  { label: "Antigravity Gemini Pro Agent", value: "ag/gemini-pro-agent" },
  { label: "Antigravity Gemini 3 Flash Agent", value: "ag/gemini-3-flash-agent" },
  { label: "Antigravity Gemini 3.1 Pro Low", value: "ag/gemini-3.1-pro-low" },
  { label: "Antigravity Gemini 3.5 Flash Low", value: "ag/gemini-3.5-flash-low" },
  { label: "Antigravity Gemini 3.5 Flash Extra Low", value: "ag/gemini-3.5-flash-extra-low" },
  { label: "Antigravity Gemini 3 Flash", value: "ag/gemini-3-flash" },
  { label: "Gemini 2.5 Pro Exp Free", value: "openrouter/google/gemini-2.5-pro-exp-03-25:free" },
  { label: "Gemini 2.0 Flash Exp Free", value: "openrouter/google/gemini-2.0-flash-exp:free" },
  { label: "Gemini 2.0 Flash Thinking Free", value: "openrouter/google/gemini-2.0-flash-thinking-exp:free" },
  { label: "Gemini 2.0 Flash Lite Preview Free", value: "openrouter/google/gemini-2.0-flash-lite-preview-02-05:free" },
  { label: "Gemma 4 26B IT Free", value: "openrouter/google/gemma-4-26b-a4b-it:free" },
  { label: "Gemma 4 31B IT Free", value: "openrouter/google/gemma-4-31b-it:free" },
  { label: "Gemini 3 Flash Preview", value: "gemini/gemini-3-flash-preview" },
  { label: "Gemini 3.1 Flash Lite Preview", value: "gemini/gemini-3.1-flash-lite-preview" },
  { label: "Gemini 3.1 Pro Preview", value: "gemini/gemini-3.1-pro-preview" },
  { label: "Gemma 4 31B IT", value: "gemini/gemma-4-31b-it" },
];

const fallbackAsrModelOptions: ModelOption[] = [
  { label: "WhisperX local (tiny int8)", value: "tiny" },
  { label: "WhisperX local (base int8)", value: "base" },
  { label: "WhisperX local (small int8)", value: "small" },
  { label: "WhisperX local (medium int8)", value: "medium" },
  { label: "WhisperX local (large-v3 float16)", value: "large-v3" },
  { label: "Gemini 2.5 Flash STT qua 9Router", value: "gemini/gemini-2.5-flash" },
  { label: "Gemini 2.5 Flash Lite STT qua 9Router", value: "gemini/gemini-2.5-flash-lite" },
  { label: "Gemini 2.5 Pro STT qua 9Router", value: "gemini/gemini-2.5-pro" },
];

const voiceModels = [
  { label: "Trúc Ly - nữ Bắc tự nhiên", value: "Trúc Ly" },
  { label: "Ngọc Linh - nữ Bắc kể chuyện", value: "Ngọc Linh" },
  { label: "Đoan Trang - nữ Bắc tự nhiên", value: "Đoan Trang" },
  { label: "Mai Anh - nữ Bắc tin tức", value: "Mai Anh" },
  { label: "Thục Đoan - nữ Nam kể chuyện", value: "Thục Đoan" },
  { label: "Thùy Dung - nữ Nam tin tức", value: "Thùy Dung" },
  { label: "Ngọc Trân - nữ Trung tự nhiên", value: "Ngọc Trân" },
  { label: "Phạm Tuyên - nam Bắc tự nhiên", value: "Phạm Tuyên" },
  { label: "Thái Sơn - nam Nam kể chuyện", value: "Thái Sơn" },
  { label: "Xuân Vĩnh - nam Nam tự nhiên", value: "Xuân Vĩnh" },
  { label: "Thanh Bình - nam Bắc kể chuyện", value: "Thanh Bình" },
  { label: "Minh Đức - nam Bắc tin tức", value: "Minh Đức" },
  { label: "Minh Triết - nam Nam tin tức", value: "Minh Triết" },
  { label: "Quang Sơn - nam Trung tự nhiên", value: "Quang Sơn" },
];

const defaultSubtitleStyle: SubtitleStyle = {
  x: 12,
  y: 72,
  width: 76,
  height: 16,
  font_size: 42,
  color: "#FFFFFF",
  outline_color: "#000000",
  outline_width: 3,
  align: "center",
};

const defaultBlurStyle: BlurStyle = {
  enabled: false,
  x: 18,
  y: 10,
  width: 44,
  height: 18,
  blur: 16,
  opacity: 0.26,
};

function clampNumber(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function normalizeSubtitleStyle(style?: Partial<SubtitleStyle>): SubtitleStyle {
  return {
    ...defaultSubtitleStyle,
    ...style,
    x: clampNumber(style?.x ?? defaultSubtitleStyle.x, 0, 95),
    y: clampNumber(style?.y ?? defaultSubtitleStyle.y, 0, 95),
    width: clampNumber(style?.width ?? defaultSubtitleStyle.width, 5, 100),
    height: clampNumber(style?.height ?? defaultSubtitleStyle.height, 4, 60),
    font_size: Math.round(clampNumber(style?.font_size ?? defaultSubtitleStyle.font_size, 12, 120)),
    outline_width: Math.round(clampNumber(style?.outline_width ?? defaultSubtitleStyle.outline_width, 0, 12)),
    align: style?.align || defaultSubtitleStyle.align,
  };
}

function normalizeBlurStyle(style?: Partial<BlurStyle>): BlurStyle {
  return {
    ...defaultBlurStyle,
    ...style,
    enabled: Boolean(style?.enabled ?? defaultBlurStyle.enabled),
    x: clampNumber(style?.x ?? defaultBlurStyle.x, 0, 95),
    y: clampNumber(style?.y ?? defaultBlurStyle.y, 0, 95),
    width: clampNumber(style?.width ?? defaultBlurStyle.width, 5, 100),
    height: clampNumber(style?.height ?? defaultBlurStyle.height, 4, 60),
    blur: Math.round(clampNumber(style?.blur ?? defaultBlurStyle.blur, 0, 48)),
    opacity: clampNumber(style?.opacity ?? defaultBlurStyle.opacity, 0, 0.95),
  };
}

function normalizeSegment(segment: ScriptSegment, voiceModel: string): ScriptSegment {
  return {
    ...segment,
    voice_model: segment.voice_model || voiceModel,
    subtitle_style: normalizeSubtitleStyle(segment.subtitle_style),
    blur_style: normalizeBlurStyle(segment.blur_style),
  };
}
function wrapSubtitlePreviewLikeRender(text: string, boxWidthPx: number, fontSize: number) {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) return "";

  if (hasCjkText(normalized)) {
    const charsPerLine = Math.max(2, Math.floor(boxWidthPx / Math.max(1, fontSize)));
    const lines: string[] = [];
    for (let index = 0; index < normalized.length; index += charsPerLine) {
      lines.push(normalized.slice(index, index + charsPerLine));
    }
    return lines.join("\n");
  }

  const charsPerLine = Math.max(8, Math.floor(boxWidthPx / Math.max(1, fontSize * 0.55)));
  const lines: string[] = [];
  let current = "";
  for (const word of normalized.split(" ")) {
    const candidate = current ? `${current} ${word}` : word;
    if (current && candidate.length > charsPerLine) {
      lines.push(current);
      current = word;
    } else {
      current = candidate;
    }
  }
  if (current) lines.push(current);
  return lines.join("\n");
}

function copyDemoSegments(items: ScriptSegment[], voiceModel: string): ScriptSegment[] {
  return items.map((segment) => {
    const normalized = normalizeSegment(segment, voiceModel);
    return {
      ...normalized,
      subtitle_style: normalizeSubtitleStyle(normalized.subtitle_style),
      blur_style: normalizeBlurStyle(normalized.blur_style),
    };
  });
}

function demoSignature(items: ScriptSegment[], voiceModel: string, textLayerEnabled = true) {
  return JSON.stringify({
    text_layer_enabled: textLayerEnabled,
    segments: copyDemoSegments(items, voiceModel),
  });
}

function mediaUrl(value: string) {
  if (!value) return "";
  if (value.startsWith("http://") || value.startsWith("https://") || value.startsWith("blob:")) return value;
  return new URL(value, BACKEND_URL).toString();
}

function formatTime(seconds: number) {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  return `${minutes}:${secs.toString().padStart(2, "0")}`;
}

function hasCjkText(text: string) {
  return /[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]/.test(text);
}

function countSpeechUnits(text: string) {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  if (hasCjkText(trimmed)) return trimmed.match(/[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]/g)?.length ?? 0;
  return trimmed.split(/\s+/).filter(Boolean).length;
}

function estimateVoiceDuration(text: string) {
  const units = countSpeechUnits(text);
  if (units === 0) return 0;
  return Math.max(0.4, units / (hasCjkText(text) ? 5.5 : VI_WORDS_PER_SECOND));
}

function normalizeStage(message: string) {
  const lower = message.toLowerCase();
  if (lower.includes("initial")) return "Đang khởi tạo workspace...";
  if (lower.includes("ocr")) return "Đang đọc phụ đề bằng OCR...";
  if (lower.includes("extract") || lower.includes("transcrib")) return "Đang tách âm thanh, nhận dạng và kiểm tra OCR...";
  if (lower.includes("translat")) return "Đang dịch kịch bản...";
  if (lower.includes("voice") || lower.includes("tts")) return "Đang tạo giọng lồng tiếng...";
  if (lower.includes("render")) return "Đang dựng video cuối...";
  if (lower.includes("complete")) return "Hoàn tất";
  return message;
}

function clampProgress(value: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.min(100, Math.max(0, Math.round(value)));
}

function progressFromStep(message: string) {
  const lower = message.toLowerCase();
  if (lower.includes("initial")) return 8;
  if (lower.includes("ocr")) return 32;
  if (lower.includes("extract") || lower.includes("transcrib")) return 25;
  if (lower.includes("translat")) return 45;
  if (lower.includes("voice") || lower.includes("tts")) return 70;
  if (lower.includes("render")) return 90;
  if (lower.includes("complete")) return 100;
  return null;
}

function phaseToStep(phase: string) {
  if (phase === "prepare" || phase === "recognize") return 1;
  if (phase === "translate") return 2;
  if (phase === "voice") return 3;
  if (phase === "render" || phase === "complete") return 4;
  return null;
}

function phaseLabel(phase: string) {
  switch (phase) {
    case "prepare":
      return "Chuẩn bị media";
    case "recognize":
      return "Nhận dạng nội dung";
    case "translate":
      return "Dịch kịch bản";
    case "voice":
      return "Tạo giọng";
    case "render":
      return "Dựng video";
    case "complete":
      return "Hoàn tất";
    default:
      return "Đang xử lý";
  }
}

function segmentProgress(stats: ProcessingStats | null) {
  const total = stats?.groups ?? stats?.segments;
  if (!stats || typeof stats.chunks !== "number" || typeof total !== "number" || total <= 0) return null;
  const done = Math.min(total, Math.max(0, stats.chunks));
  return { done, total, percent: clampProgress((done / total) * 100) };
}

function statsSummary(stats: ProcessingStats | null) {
  if (!stats) return "";
  const voiceProgress = segmentProgress(stats);
  if (voiceProgress) return `Đã tạo ${voiceProgress.done}/${voiceProgress.total} đoạn voice`;

  const items = [];
  if (typeof stats.engine === "string" && stats.engine) items.push(stats.engine);
  if (typeof stats.segments === "number") items.push(`${stats.segments} đoạn`);
  if (typeof stats.speech_duration === "number" && stats.speech_duration > 0) items.push(`${formatTime(stats.speech_duration)} thoại`);
  if (typeof stats.video_duration === "number" && stats.video_duration > 0) items.push(`${formatTime(stats.video_duration)} video`);
  if (typeof stats.characters === "number" && stats.characters > 0) items.push(`${stats.characters} ký tự`);
  return items.join(" · ");
}

function activityProgressLabel(activity: ProcessingActivity) {
  const voiceProgress = segmentProgress(activity.stats);
  if (voiceProgress) return `${voiceProgress.done}/${voiceProgress.total}`;
  return `${activity.progress}%`;
}

function modelOptionsFromPayload(payload: { models?: Array<{ id?: string; label?: string }> }) {
  return (payload.models ?? [])
    .map((model) => ({
      label: model.label || model.id || "",
      value: model.id || model.label || "",
    }))
    .filter((model) => model.value);
}

function preferredTranslationModel(
  currentModel: string,
  models: ModelOption[],
  payloadDefaultModel?: string,
  preserveCurrent = false,
) {
  const hasModel = (value: string | undefined) => Boolean(value && models.some((model) => model.value === value));
  if (preserveCurrent && hasModel(currentModel)) return currentModel;
  if (hasModel(DEFAULT_TRANSLATION_MODEL)) return DEFAULT_TRANSLATION_MODEL;
  if (hasModel(payloadDefaultModel)) return payloadDefaultModel || models[0]?.value || DEFAULT_TRANSLATION_MODEL;
  return models[0]?.value || DEFAULT_TRANSLATION_MODEL;
}

export default function VideoDubbingStudio() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const translationModelTouchedRef = useRef(false);
  const persistenceReadyRef = useRef(false);
  const sessionIdRef = useRef('');
  const lastStoredVideoFileRef = useRef<File | null>(null);
  const lastStoredCloneFileRef = useRef<File | null>(null);
  const pendingFileSavesRef = useRef(0);
  const cloneSelectionRevisionRef = useRef(0);
  const activeRequestControllerRef = useRef<AbortController | null>(null);
  const activeOperationIdRef = useRef("");
  const activeRenderJobIdRef = useRef("");
  const cancelRequestedRef = useRef(false);
  const [activeWorkspace, setActiveWorkspace] = useState<"clone" | "short" | "gen" | "setup">("clone");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [sourceVideoUrl, setSourceVideoUrl] = useState("");
  const [segments, setSegments] = useState<ScriptSegment[]>([]);
  const [captionSuggestions, setCaptionSuggestions] = useState<string[]>([]);
  const [flashTextTracks, setFlashTextTracks] = useState<FlashTextTrack[]>([]);
  const [config, setConfig] = useState<StudioConfig>({
    sourceLanguage: "auto",
    targetLanguage: "vi",
    translationProvider: "9router",
    translationModel: DEFAULT_TRANSLATION_MODEL,
    asrModel: DEFAULT_ASR_MODEL,
    asrEngine: "auto",
    whisperModel: "auto",
    whisperBeamSize: 5,
    segmentLanguageDetection: true,
    voiceModel: "Trúc Ly",
    voiceMode: "system",
    ttsDevice: "cuda",
    copyrightAcknowledged: false,
    copyrightSource: "unknown",
    copyrightNotes: "",
    ocrFallback: false,
    ocrForce: false,
    ocrModel: DEFAULT_OCR_MODEL,
    ocrIntervalSeconds: 0.75,
    ocrCropBottomRatio: 0.35,
    flashTextEnabled: false,
    flashTextMode: "balanced",
    flashTextMinConfidence: 0.58,
    flashTextMaxDuration: 3.0,
    vocalSeparation: false,
    originalVocalGain: 0,
    accompanimentGain: 1,
    softTimingFit: true,
    timingMaxDrift: 1.5,
    timingMinGap: 0.12,
    timingMaxAtempo: 1.1,
    voiceSpeed: 1.0,
  });
  const [translationModels, setTranslationModels] = useState<ModelOption[]>(fallbackTranslationModels);
  const [asrModels, setAsrModels] = useState<ModelOption[]>(fallbackAsrModelOptions);
  const [pipelineSourceLanguages, setPipelineSourceLanguages] = useState<ModelOption[]>(languageOptions);
  const [pipelineAsrEngines, setPipelineAsrEngines] = useState<ModelOption[]>([
    { label: "Tự động", value: "auto" },
    { label: "Whisper", value: "whisper" },
    { label: "Paraformer", value: "paraformer" },
  ]);
  const [pipelineWhisperModels, setPipelineWhisperModels] = useState<ModelOption[]>(["auto", "tiny", "base", "small", "medium", "large-v3"].map((value) => ({ label: value, value })));
  const [pipelineOcrModels, setPipelineOcrModels] = useState<ModelOption[]>([
    { label: "Gemini 2.5 Flash OCR", value: DEFAULT_OCR_MODEL },
    { label: "Gemini 2.5 Pro OCR", value: "gemini/gemini-2.5-pro" },
  ]);
  const [activeStep, setActiveStep] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isRendering, setIsRendering] = useState(false);
  const [cloneReferenceFile, setCloneReferenceFile] = useState<File | null>(null);
  const [preparedCloneClip, setPreparedCloneClip] = useState<PreparedAudioClip | null>(null);
  const [initialCloneSelection, setInitialCloneSelection] = useState<{ start: number; end: number } | null>(null);
  const [shorteningSegmentId, setShorteningSegmentId] = useState<number | null>(null);
  const [selectedSegmentId, setSelectedSegmentId] = useState<number | null>(null);
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState("Chưa có video");
  const [processingPhase, setProcessingPhase] = useState("");
  const [processingDetail, setProcessingDetail] = useState("");
  const [processingStats, setProcessingStats] = useState<ProcessingStats | null>(null);
  const [processingActivities, setProcessingActivities] = useState<ProcessingActivity[]>([]);
  const [resultVideoUrl, setResultVideoUrl] = useState("");
  const [resultSubtitleUrl, setResultSubtitleUrl] = useState("");
  const [renderJobId, setRenderJobId] = useState("");
  const [textLayerEnabled, setTextLayerEnabled] = useState(true);
  const [savedDemoSegments, setSavedDemoSegments] = useState<ScriptSegment[] | null>(null);
  const [savedTextLayerEnabled, setSavedTextLayerEnabled] = useState<boolean | null>(null);
  const [toast, setToast] = useState<ToastState>(null);
  const [sessionSaveStatus, setSessionSaveStatus] = useState<SessionSaveStatus>('loading');
  const [sessionHydrated, setSessionHydrated] = useState(false);

  const handleCloneClipError = useCallback((message: string) => {
    setToast({ type: "error", message });
    window.setTimeout(() => setToast(null), 5200);
  }, []);

  const handleCloneClipReady = useCallback((clip: PreparedAudioClip | null) => {
    cloneSelectionRevisionRef.current += 1;
    setPreparedCloneClip(clip);
  }, []);

  const voiceControlsLocked = isAnalyzing || isRendering;

  const totalDuration = useMemo(() => {
    if (segments.length === 0) return 0;
    return Math.max(...segments.map((segment) => segment.end));
  }, [segments]);

  const currentVideoUrl = sourceVideoUrl || previewUrl || resultVideoUrl;
  const selectedSegment = useMemo(
    () => segments.find((segment) => segment.id === selectedSegmentId) || segments[0] || null,
    [segments, selectedSegmentId],
  );
  const savedDemoSignature = useMemo(
    () => (savedDemoSegments ? demoSignature(savedDemoSegments, config.voiceModel, savedTextLayerEnabled ?? true) : ""),
    [savedDemoSegments, savedTextLayerEnabled, config.voiceModel],
  );
  const currentDemoSignature = useMemo(
    () => (segments.length > 0 ? demoSignature(segments, config.voiceModel, textLayerEnabled) : ""),
    [segments, textLayerEnabled, config.voiceModel],
  );
  const hasSavedDemo = Boolean(savedDemoSegments?.length);
  const hasUnsavedDemoChanges = segments.length > 0 && currentDemoSignature !== savedDemoSignature;
  const sessionSnapshot = useMemo<StudioSessionSnapshot>(() => ({
    version: STUDIO_SESSION_VERSION,
    sessionId: sessionIdRef.current,
    activeWorkspace,
    sourceVideoUrl: sourceVideoUrl.startsWith('blob:') ? '' : sourceVideoUrl,
    segments,
    captionSuggestions,
    flashTextTracks,
    config,
    activeStep,
    selectedSegmentId,
    progress,
    statusText,
    processingPhase,
    processingDetail,
    processingStats,
    processingActivities,
    resultVideoUrl: resultVideoUrl.startsWith('blob:') ? '' : resultVideoUrl,
    resultSubtitleUrl: resultSubtitleUrl.startsWith('blob:') ? '' : resultSubtitleUrl,
    renderJobId,
    textLayerEnabled,
    savedDemoSegments,
    savedTextLayerEnabled,
    cloneClipSelection: preparedCloneClip
      ? {
          start: preparedCloneClip.start,
          end: preparedCloneClip.end,
          sourceDuration: preparedCloneClip.sourceDuration,
        }
      : initialCloneSelection
        ? { ...initialCloneSelection, sourceDuration: 0 }
        : null,
  }), [
    activeStep,
    activeWorkspace,
    config,
    initialCloneSelection,
    preparedCloneClip,
    processingActivities,
    processingDetail,
    processingPhase,
    processingStats,
    progress,
    resultSubtitleUrl,
    resultVideoUrl,
    renderJobId,
    savedDemoSegments,
    savedTextLayerEnabled,
    segments,
    captionSuggestions,
    selectedSegmentId,
    sessionHydrated,
    sourceVideoUrl,
    statusText,
    textLayerEnabled,
    flashTextTracks,
  ]);

  useEffect(() => {
    let cancelled = false;

    async function restoreSession() {
      let snapshot: StudioSessionSnapshot | null = null;
      let fileStoreFailed = false;

      try {
        const stored = window.sessionStorage.getItem(STUDIO_SESSION_KEY);
        if (stored) {
          const parsed = JSON.parse(stored) as StudioSessionSnapshot;
          // Version 1 stored the old OCR default as enabled. Versions 1/2 also
          // used the old always-on vocal-separation default; migrate that
          // legacy default to the new opt-in behavior below.
          if ((parsed.version === STUDIO_SESSION_VERSION || parsed.version === 3 || parsed.version === 2 || parsed.version === 1) && parsed.sessionId) snapshot = parsed;
        }
      } catch {
        window.sessionStorage.removeItem(STUDIO_SESSION_KEY);
      }

      const sessionId = snapshot?.sessionId || window.crypto.randomUUID();
      sessionIdRef.current = sessionId;

      if (snapshot) {
        const restoredConfig = { ...config, ...snapshot.config };
        if (snapshot.version < STUDIO_SESSION_VERSION) {
          restoredConfig.ocrFallback = false;
          restoredConfig.vocalSeparation = false;
          restoredConfig.originalVocalGain = 0;
          restoredConfig.accompanimentGain = 1;
        }
        if (snapshot.version < 4) {
          restoredConfig.flashTextEnabled = false;
          restoredConfig.flashTextMode = "balanced";
          restoredConfig.flashTextMinConfidence = 0.58;
          restoredConfig.flashTextMaxDuration = 3.0;
        }
        const restoredSegments = Array.isArray(snapshot.segments)
          ? snapshot.segments.map((segment) => normalizeSegment(segment, restoredConfig.voiceModel))
          : [];
        const restoredSavedSegments = Array.isArray(snapshot.savedDemoSegments)
          ? snapshot.savedDemoSegments.map((segment) => normalizeSegment(segment, restoredConfig.voiceModel))
          : null;

        translationModelTouchedRef.current = Boolean(snapshot.config?.translationModel);
        setActiveWorkspace(snapshot.activeWorkspace === 'gen' ? 'gen' : snapshot.activeWorkspace === 'short' ? 'short' : snapshot.activeWorkspace === 'setup' ? 'setup' : 'clone');
        setSourceVideoUrl(snapshot.sourceVideoUrl || '');
        setSegments(restoredSegments);
        setCaptionSuggestions(Array.isArray(snapshot.captionSuggestions) ? snapshot.captionSuggestions.slice(0, 3) : []);
        setFlashTextTracks(Array.isArray(snapshot.flashTextTracks) ? snapshot.flashTextTracks : []);
        setConfig(restoredConfig);
        setActiveStep(clampNumber(snapshot.activeStep, 0, workflowSteps.length - 1));
        setSelectedSegmentId(snapshot.selectedSegmentId ?? restoredSegments[0]?.id ?? null);
        setProgress(clampProgress(snapshot.progress));
        setStatusText(snapshot.statusText || 'Đã khôi phục session');
        setProcessingPhase(snapshot.processingPhase || '');
        setProcessingDetail(snapshot.processingDetail || '');
        setProcessingStats(snapshot.processingStats || null);
        setProcessingActivities(Array.isArray(snapshot.processingActivities) ? snapshot.processingActivities : []);
        setResultVideoUrl(snapshot.resultVideoUrl || '');
        setResultSubtitleUrl(snapshot.resultSubtitleUrl || '');
        setRenderJobId(snapshot.renderJobId || '');
        setTextLayerEnabled(snapshot.textLayerEnabled ?? true);
        setSavedDemoSegments(restoredSavedSegments);
        setSavedTextLayerEnabled(snapshot.savedTextLayerEnabled ?? null);
        setInitialCloneSelection(snapshot.cloneClipSelection
          ? { start: snapshot.cloneClipSelection.start, end: snapshot.cloneClipSelection.end }
          : null);
      }

      const safelyReadFile = async (key: string) => {
        try {
          return await readSessionFile(key);
        } catch {
          fileStoreFailed = true;
          return null;
        }
      };
      const [restoredVideoFile, restoredCloneFile] = await Promise.all([
        safelyReadFile(`${sessionId}:video`),
        safelyReadFile(`${sessionId}:clone-reference`),
      ]);

      if (!cancelled) {
        if (restoredVideoFile) {
          lastStoredVideoFileRef.current = restoredVideoFile;
          setFile(restoredVideoFile);
          setPreviewUrl(URL.createObjectURL(restoredVideoFile));
        }
        if (restoredCloneFile) {
          lastStoredCloneFileRef.current = restoredCloneFile;
          setCloneReferenceFile(restoredCloneFile);
        }
        persistenceReadyRef.current = true;
        setSessionHydrated(true);
        setSessionSaveStatus(fileStoreFailed ? 'error' : 'saved');
      }
    }

    void restoreSession();
    return () => {
      cancelled = true;
    };
    // The initial config is intentionally captured once, before model discovery updates it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!persistenceReadyRef.current || !sessionSnapshot.sessionId) return;

    const persistSnapshot = () => {
      try {
        window.sessionStorage.setItem(STUDIO_SESSION_KEY, JSON.stringify(sessionSnapshot));
      } catch {
        setSessionSaveStatus('error');
      }
    };
    const timer = window.setTimeout(persistSnapshot, 180);
    window.addEventListener('pagehide', persistSnapshot);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener('pagehide', persistSnapshot);
    };
  }, [sessionSnapshot]);

  useEffect(() => {
    if (!sessionHydrated || !renderJobId) return;
    let cancelled = false;
    let timer: number | null = null;

    const poll = async () => {
      if (cancelled) return;
      try {
        const response = await fetch(`${RENDER_JOB_URL}/${renderJobId}`, { cache: "no-store" });
        if (response.status === 404) {
          setRenderJobId("");
          return;
        }
        if (!response.ok) throw new Error(await response.text());
        const job = (await response.json()) as {
          status?: string;
          phase?: string;
          progress?: number;
          error?: string;
          output_video_url?: string;
          output_subtitle_url?: string;
          completed_groups?: number;
          total_groups?: number;
        };
        const jobProgress = typeof job.progress === "number" ? clampProgress(job.progress) : null;
        const completedGroups = typeof job.completed_groups === "number" ? Math.max(0, job.completed_groups) : null;
        const totalGroups = typeof job.total_groups === "number" && job.total_groups > 0 ? job.total_groups : null;
        const voiceStats: ProcessingStats | null = completedGroups !== null && totalGroups !== null
          ? { chunks: Math.min(completedGroups, totalGroups), groups: totalGroups }
          : null;
        const voiceDetail = voiceStats ? statsSummary(voiceStats) : "";

        if (jobProgress !== null) setProgress((current) => Math.max(current, jobProgress));
        if (job.phase) {
          setProcessingPhase(job.phase);
          const nextStep = phaseToStep(job.phase);
          if (nextStep !== null) setActiveStep(nextStep);
        }
        if (voiceStats) {
          setProcessingStats((current) => ({ ...current, ...voiceStats }));
          setProcessingDetail(job.phase === "render" ? `${voiceDetail} \u00b7 \u0110ang d\u1ef1ng video` : voiceDetail);
          pushProcessingActivity("voice", phaseLabel("voice"), voiceDetail, segmentProgress(voiceStats)?.percent ?? 0, voiceStats);
        }
        if (job.phase && job.phase !== "voice" && jobProgress !== null) {
          const phaseDetail = job.phase === "render" ? "\u0110ang tr\u1ed9n \u00e2m thanh v\u00e0 d\u1ef1ng video" : phaseLabel(job.phase);
          pushProcessingActivity(job.phase, phaseLabel(job.phase), phaseDetail, jobProgress, null);
        }
        if (job.status === "complete") {
          setIsRendering(false);
          setActiveStep(4);
          setProgress(100);
          setStatusText("Ho\u00e0n t\u1ea5t - xem demo tr\u01b0\u1edbc khi t\u1ea3i xu\u1ed1ng");
          if (job.output_video_url) setResultVideoUrl(mediaUrl(job.output_video_url));
          if (job.output_subtitle_url) setResultSubtitleUrl(mediaUrl(job.output_subtitle_url));
          return;
        }
        if (job.status === "failed") {
          setIsRendering(false);
          setStatusText("Render th\u1ea5t b\u1ea1i");
          setProcessingDetail(job.error || "Render worker failed");
          return;
        }
        if (job.status === "queued" || job.status === "running") {
          setIsRendering(true);
          setStatusText(job.status === "queued" ? "Render \u0111ang ch\u1edd worker" : phaseLabel(job.phase || ""));
        }
      } catch {
        // A transient API restart must not discard the durable job ID.
      }
      timer = window.setTimeout(poll, 2000);
    };

    void poll();
    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
    };
  }, [renderJobId, sessionHydrated]);

  useEffect(() => {
    if (!persistenceReadyRef.current || !sessionIdRef.current || file === lastStoredVideoFileRef.current) return;
    const nextFile = file;
    pendingFileSavesRef.current += 1;
    setSessionSaveStatus('loading');
    void writeSessionFile(`${sessionIdRef.current}:video`, nextFile)
      .then(() => {
        lastStoredVideoFileRef.current = nextFile;
        pendingFileSavesRef.current -= 1;
        if (pendingFileSavesRef.current === 0) setSessionSaveStatus('saved');
      })
      .catch(() => {
        pendingFileSavesRef.current -= 1;
        setSessionSaveStatus('error');
      });
  }, [file]);

  useEffect(() => {
    if (!persistenceReadyRef.current || !sessionIdRef.current || cloneReferenceFile === lastStoredCloneFileRef.current) return;
    const nextFile = cloneReferenceFile;
    pendingFileSavesRef.current += 1;
    setSessionSaveStatus('loading');
    void writeSessionFile(`${sessionIdRef.current}:clone-reference`, nextFile)
      .then(() => {
        lastStoredCloneFileRef.current = nextFile;
        pendingFileSavesRef.current -= 1;
        if (pendingFileSavesRef.current === 0) setSessionSaveStatus('saved');
      })
      .catch(() => {
        pendingFileSavesRef.current -= 1;
        setSessionSaveStatus('error');
      });
  }, [cloneReferenceFile]);

  useEffect(() => () => {
    if (previewUrl.startsWith('blob:')) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);
  const copyrightReady = config.copyrightAcknowledged && config.copyrightSource !== "unknown";

  useEffect(() => {
    let mounted = true;

    async function loadTranslationModels() {
      try {
        const response = await fetch(TRANSLATION_MODELS_URL);
        if (!response.ok) return;
        const payload = (await response.json()) as {
          default_model?: string;
          models?: Array<{ id?: string; label?: string }>;
        };
        const nextModels = modelOptionsFromPayload(payload);

        if (!mounted || nextModels.length === 0) return;
        setTranslationModels(nextModels);
        setConfig((current) => ({
          ...current,
          translationModel: preferredTranslationModel(
            current.translationModel,
            nextModels,
            payload.default_model,
            translationModelTouchedRef.current,
          ),
        }));
      } catch {
        if (mounted) {
          setTranslationModels(fallbackTranslationModels);
          setConfig((current) => ({
            ...current,
            translationModel: preferredTranslationModel(
              current.translationModel,
              fallbackTranslationModels,
              DEFAULT_TRANSLATION_MODEL,
              translationModelTouchedRef.current,
            ),
          }));
        }
      }
    }

    async function loadSttModels() {
      try {
        const response = await fetch(STT_MODELS_URL);
        if (!response.ok) return;
        const payload = (await response.json()) as {
          default_model?: string;
          models?: Array<{ id?: string; label?: string }>;
        };
        const nextModels = modelOptionsFromPayload(payload);

        if (!mounted || nextModels.length === 0) return;
        setAsrModels(nextModels);
        setConfig((current) => ({
          ...current,
          asrModel: nextModels.some((model) => model.value === current.asrModel)
            ? current.asrModel
            : payload.default_model || nextModels[0].value,
        }));
      } catch {
        if (mounted) setAsrModels(fallbackAsrModelOptions);
      }
    }

    async function loadPipelineSettings() {
      try {
        const response = await fetch(PIPELINE_SETTINGS_URL, { cache: "no-store" });
        if (!response.ok) return;
        const payload = (await response.json()) as {
          source_languages?: Array<{ value?: string; label?: string }>;
          asr_engines?: string[];
          whisper_models?: string[];
          timing?: { soft_timing_fit?: boolean; max_drift_s?: number; min_gap_s?: number; max_atempo?: number; voice_speed?: number };
          ocr?: { interval_seconds?: number; crop_bottom_ratio?: number; enabled_by_default?: boolean; models?: Array<{ id?: string; label?: string }> };
        };
        if (!mounted) return;
        const sourceOptions = (payload.source_languages ?? []).filter((item) => item.value).map((item) => ({ value: item.value as string, label: item.label || item.value as string }));
        const asrEngineOptions = (payload.asr_engines ?? []).map((value) => ({ value, label: value === "auto" ? "Tự động" : value === "paraformer" ? "Paraformer" : "Whisper" }));
        const whisperOptions = (payload.whisper_models ?? []).map((value) => ({ value, label: value }));
        const ocrOptions = (payload.ocr?.models ?? []).filter((item) => item.id).map((item) => ({ value: item.id as string, label: item.label || item.id as string }));
        if (sourceOptions.length) setPipelineSourceLanguages(sourceOptions);
        if (asrEngineOptions.length) setPipelineAsrEngines(asrEngineOptions);
        if (whisperOptions.length) setPipelineWhisperModels(whisperOptions);
        if (ocrOptions.length) setPipelineOcrModels(ocrOptions);
        setConfig((current) => ({
          ...current,
          softTimingFit: payload.timing?.soft_timing_fit ?? current.softTimingFit,
          timingMaxDrift: payload.timing?.max_drift_s ?? current.timingMaxDrift,
          timingMinGap: payload.timing?.min_gap_s ?? current.timingMinGap,
          timingMaxAtempo: payload.timing?.max_atempo ?? current.timingMaxAtempo,
          voiceSpeed: payload.timing?.voice_speed ?? current.voiceSpeed,
          ocrIntervalSeconds: payload.ocr?.interval_seconds ?? current.ocrIntervalSeconds,
          ocrCropBottomRatio: payload.ocr?.crop_bottom_ratio ?? current.ocrCropBottomRatio,
        }));
      } catch {
        // Local fallback remains valid when an older backend is running.
      }
    }

    loadTranslationModels();
    loadSttModels();
    loadPipelineSettings();
    return () => {
      mounted = false;
    };
  }, []);

  function showToast(nextToast: ToastState) {
    setToast(nextToast);
    window.setTimeout(() => setToast(null), 5200);
  }

  async function copyCaption(caption: string) {
    try {
      await navigator.clipboard.writeText(caption);
      showToast({ type: "success", message: "Đã sao chép caption để bạn dùng ngay." });
    } catch {
      showToast({ type: "error", message: "Không thể sao chép caption trên trình duyệt này." });
    }
  }

  async function persistVideoFileForRetry(videoFile: File | null) {
    if (!sessionIdRef.current || !videoFile) return;
    try {
      await writeSessionFile(`${sessionIdRef.current}:video`, videoFile);
      lastStoredVideoFileRef.current = videoFile;
      setSessionSaveStatus('saved');
    } catch {
      setSessionSaveStatus('error');
    }
  }

  async function startNewSession() {
    if ((file || sourceVideoUrl || segments.length > 0) && !window.confirm('Tạo session mới? Dữ liệu của session hiện tại sẽ bị xóa khỏi trình duyệt.')) {
      return;
    }

    const previousSessionId = sessionIdRef.current;
    persistenceReadyRef.current = false;
    setSessionSaveStatus('loading');
    window.sessionStorage.removeItem(STUDIO_SESSION_KEY);
    if (previewUrl.startsWith('blob:')) URL.revokeObjectURL(previewUrl);

    setActiveWorkspace('clone');
    setFile(null);
    setPreviewUrl('');
    setSourceVideoUrl('');
    setSegments([]);
    setCaptionSuggestions([]);
    setFlashTextTracks([]);
    setConfig({
      sourceLanguage: 'auto',
      targetLanguage: 'vi',
      translationProvider: '9router',
      translationModel: DEFAULT_TRANSLATION_MODEL,
      asrModel: DEFAULT_ASR_MODEL,
      asrEngine: 'auto',
      whisperModel: 'auto',
      whisperBeamSize: 5,
      segmentLanguageDetection: true,
      voiceModel: 'Trúc Ly',
      voiceMode: 'system',
      ttsDevice: 'cuda',
      copyrightAcknowledged: false,
      copyrightSource: 'unknown',
      copyrightNotes: '',
      ocrFallback: false,
      ocrForce: false,
      ocrModel: DEFAULT_OCR_MODEL,
      ocrIntervalSeconds: 0.75,
      ocrCropBottomRatio: 0.35,
      flashTextEnabled: false,
      flashTextMode: 'balanced',
      flashTextMinConfidence: 0.58,
      flashTextMaxDuration: 3.0,
      vocalSeparation: false,
      originalVocalGain: 0,
      accompanimentGain: 1,
      softTimingFit: true,
      timingMaxDrift: 1.5,
      timingMinGap: 0.12,
      timingMaxAtempo: 1.1,
      voiceSpeed: 1.0,
    });
    setActiveStep(0);
    setCloneReferenceFile(null);
    cloneSelectionRevisionRef.current += 1;
    setPreparedCloneClip(null);
    setInitialCloneSelection(null);
    setSelectedSegmentId(null);
    setProgress(0);
    setStatusText('Chưa có video');
    setProcessingPhase('');
    setProcessingDetail('');
    setProcessingStats(null);
    setProcessingActivities([]);
    setResultVideoUrl('');
    setResultSubtitleUrl('');
    setTextLayerEnabled(true);
    setSavedDemoSegments(null);
    setSavedTextLayerEnabled(null);
    if (inputRef.current) inputRef.current.value = '';

    try {
      if (previousSessionId) {
        await deleteSessionFiles([
          `${previousSessionId}:video`,
          `${previousSessionId}:clone-reference`,
        ]);
      }
      setSessionSaveStatus('saved');
    } catch {
      setSessionSaveStatus('error');
    } finally {
      sessionIdRef.current = window.crypto.randomUUID();
      lastStoredVideoFileRef.current = null;
      lastStoredCloneFileRef.current = null;
      pendingFileSavesRef.current = 0;
      persistenceReadyRef.current = true;
      setSessionHydrated((current) => !current);
    }
  }

  function resetProcessing(label: string, nextProgress = 0, phase = "") {
    setProgress(nextProgress);
    setStatusText(label);
    setProcessingPhase(phase);
    setProcessingDetail("");
    setProcessingStats(null);
    setProcessingActivities([]);
  }

  function pushProcessingActivity(phase: string, label: string, detail: string, nextProgress: number, stats: ProcessingStats | null) {
    const key = phase || label;
    setProcessingActivities((current) => {
      const nextActivity: ProcessingActivity = {
        id: Date.now() + Math.random(),
        key,
        phase,
        label,
        detail,
        progress: nextProgress,
        stats,
      };
      const existingIndex = current.findIndex((activity) => activity.key === key);
      if (existingIndex === -1) return [nextActivity, ...current].slice(0, 5);

      const existing = current[existingIndex];
      const updated = { ...nextActivity, id: existing.id };
      return [updated, ...current.filter((_, index) => index !== existingIndex)].slice(0, 5);
    });
  }

  function handleFile(nextFile: File | undefined) {
    if (!nextFile) return;
    if (!nextFile.type.startsWith("video/") && !/\.(mp4|mov|mkv|webm)$/i.test(nextFile.name)) {
      showToast({ type: "error", message: "Vui lòng chọn file video hợp lệ." });
      return;
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(nextFile);
    void persistVideoFileForRetry(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setFlashTextTracks([]);
    if (segments.length > 0) {
      setSourceVideoUrl("");
      setResultVideoUrl("");
      setResultSubtitleUrl("");
      resetProcessing("Đã khôi phục video gốc; script hiện tại được giữ nguyên", 0, "prepare");
      showToast({ type: "success", message: "Đã khôi phục video gốc và giữ nguyên script hiện tại." });
      return;
    }
    setSourceVideoUrl("");
    setResultVideoUrl("");
    setResultSubtitleUrl("");
    setSegments([]);
    setCaptionSuggestions([]);
    setSavedDemoSegments(null);
    setSelectedSegmentId(null);
    setActiveStep(0);
    resetProcessing("Đã chọn video");
  }

  function onDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
    handleFile(event.dataTransfer.files[0]);
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    handleFile(event.target.files?.[0]);
  }

  function updateSegment(id: number, patch: Partial<ScriptSegment>) {
    setSegments((current) =>
      current.map((segment) => (segment.id === id ? normalizeSegment({ ...segment, ...patch }, config.voiceModel) : segment)),
    );
  }

  function harmonizeLayoutAcrossSegments(sourceSegments: ScriptSegment[]) {
    if (sourceSegments.length === 0) return [];
    const layoutSource = sourceSegments.find((segment) => segment.id === selectedSegmentId) || sourceSegments[0];
    const sharedSubtitleStyle = normalizeSubtitleStyle(layoutSource.subtitle_style);
    const sharedBlurStyle = normalizeBlurStyle(layoutSource.blur_style);
    return sourceSegments.map((segment) =>
      normalizeSegment(
        {
          ...segment,
          subtitle_style: sharedSubtitleStyle,
          blur_style: sharedBlurStyle,
        },
        config.voiceModel,
      ),
    );
  }

  function updateSubtitleStyle(id: number, patch: Partial<SubtitleStyle>) {
    setSegments((current) => {
      const layoutSource = current.find((segment) => segment.id === id) || current[0];
      const sharedSubtitleStyle = normalizeSubtitleStyle({ ...normalizeSubtitleStyle(layoutSource?.subtitle_style), ...patch });
      return current.map((segment) =>
        normalizeSegment(
          {
            ...segment,
            subtitle_style: sharedSubtitleStyle,
          },
          config.voiceModel,
        ),
      );
    });
  }

  function updateBlurStyle(id: number, patch: Partial<BlurStyle>) {
    setSegments((current) => {
      const layoutSource = current.find((segment) => segment.id === id) || current[0];
      const sharedBlurStyle = normalizeBlurStyle({ ...normalizeBlurStyle(layoutSource?.blur_style), ...patch });
      return current.map((segment) =>
        normalizeSegment(
          {
            ...segment,
            blur_style: sharedBlurStyle,
          },
          config.voiceModel,
        ),
      );
    });
  }

  function saveDemoSnapshot() {
    if (segments.length === 0) {
      showToast({ type: "error", message: "Chưa có demo để lưu." });
      return;
    }
    const harmonizedSegments = harmonizeLayoutAcrossSegments(segments);
    setSegments(harmonizedSegments);
    setSavedDemoSegments(copyDemoSegments(harmonizedSegments, config.voiceModel));
    setSavedTextLayerEnabled(textLayerEnabled);
    showToast({ type: "success", message: "Đã lưu demo và áp dụng layout/blur/lớp chữ cho tất cả segment." });
  }

  async function autoShortenSegment(segment: ScriptSegment) {
    const text = segment.translated_text.trim();
    if (!text) {
      showToast({ type: "error", message: "Không có bản dịch để rút gọn." });
      return;
    }

    setShorteningSegmentId(segment.id);
    try {
      const response = await fetch(SHORTEN_TEXT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          source_text: segment.original_text,
          target_duration: Math.max(0.1, segment.end - segment.start),
          target_language: config.targetLanguage,
          translation_provider: config.translationProvider,
          translation_model: config.translationModel,
        }),
      });
      if (!response.ok) throw new Error(await response.text());

      const payload = (await response.json()) as { text?: string; max_words?: number };
      const shortenedText = (payload.text || "").trim();
      if (!shortenedText) throw new Error("Backend không trả câu rút gọn.");

      updateSegment(segment.id, { translated_text: shortenedText });
      showToast({
        type: "success",
        message: `Đã rút gọn còn tối đa ${payload.max_words ?? Math.max(1, Math.floor((segment.end - segment.start) * 3))} từ.`,
      });
    } catch (error) {
      showToast({
        type: "error",
        message: error instanceof Error ? error.message : "Không rút gọn được câu thoại.",
      });
    } finally {
      setShorteningSegmentId((current) => (current === segment.id ? null : current));
    }
  }

  function updateDefaultVoice(voiceModel: string) {
    setConfig((current) => ({ ...current, voiceModel }));
    setSegments((current) => current.map((segment) => ({ ...segment, voice_model: voiceModel })));
  }

  function scriptDefaultVoice() {
    return segments.find((segment) => segment.voice_model?.trim())?.voice_model || config.voiceModel;
  }

  function appendCopyrightPreflight(formData: FormData) {
    formData.append("copyright_confirmed", String(config.copyrightAcknowledged));
    formData.append("copyright_source", config.copyrightSource);
    formData.append("copyright_notes", config.copyrightNotes.trim());
  }

  function ensureCopyrightPreflight() {
    if (copyrightReady) return true;
    showToast({ type: "error", message: "Cần xác nhận quyền sử dụng nội dung trước khi xử lý." });
    return false;
  }

  async function analyzeScript() {
    if (!file) {
      showToast({ type: "error", message: "Chọn video trước khi phân tích." });
      return;
    }
    if (!ensureCopyrightPreflight()) return;

    const formData = new FormData();
    formData.append("video", file);
    formData.append("source_language", config.sourceLanguage);
    formData.append("target_language", config.targetLanguage);
    formData.append("translation_provider", config.translationProvider);
    formData.append("translation_model", config.translationModel);
    formData.append("asr_model", config.asrModel);
    formData.append("asr_engine", config.asrEngine);
    formData.append("whisper_model", config.whisperModel);
    formData.append("whisper_beam_size", String(config.whisperBeamSize));
    formData.append("segment_language_detection", String(config.segmentLanguageDetection));
    formData.append("compute_type", "int8");
    formData.append("voice_model", config.voiceModel);
    formData.append("tts_device", config.ttsDevice);
    formData.append("word_timestamps", "true");
    formData.append("mock_translation", "false");
    formData.append("ocr_fallback", String(config.ocrFallback));
    formData.append("ocr_force", String(config.ocrForce));
    formData.append("ocr_model", config.ocrModel);
    formData.append("ocr_interval_seconds", String(config.ocrIntervalSeconds));
    formData.append("ocr_crop_bottom_ratio", String(config.ocrCropBottomRatio));
    formData.append("flash_text_enabled", String(config.flashTextEnabled));
    formData.append("flash_text_mode", config.flashTextMode);
    formData.append("flash_text_min_confidence", String(config.flashTextMinConfidence));
    formData.append("flash_text_max_duration_s", String(config.flashTextMaxDuration));
    formData.append("soft_timing_fit", String(config.softTimingFit));
    formData.append("timing_max_drift_s", String(config.timingMaxDrift));
    formData.append("timing_min_gap_s", String(config.timingMinGap));
    formData.append("timing_max_atempo", String(config.timingMaxAtempo));
    formData.append("voice_speed", String(config.voiceSpeed));
    formData.append("vocal_separation", String(config.vocalSeparation));
    formData.append("original_vocal_gain", String(config.originalVocalGain));
    formData.append("accompaniment_gain", String(config.accompanimentGain));
    appendCopyrightPreflight(formData);

    setIsAnalyzing(true);
    resetProcessing("Đang nhận dạng và tách timeline...", 5, "recognize");
    setActiveStep(1);
    setToast(null);
    const controller = new AbortController();
    activeRequestControllerRef.current = controller;
    cancelRequestedRef.current = false;
    activeOperationIdRef.current = "";
    activeRenderJobIdRef.current = "";

    try {
      const response = await fetch(ANALYZE_URL, {
        method: "POST",
        body: formData,
        headers: { Accept: "text/event-stream" },
        signal: controller.signal,
      });
      await readEventStream(response);
      if (cancelRequestedRef.current) return;
      showToast({ type: "success", message: "Đã phân tích timeline kịch bản." });
    } catch (error) {
      showToast({
        type: "error",
        message: error instanceof Error ? error.message : "Không phân tích được video.",
      });
      setStatusText("Phân tích thất bại");
    } finally {
      if (activeRequestControllerRef.current === controller) activeRequestControllerRef.current = null;
      setIsAnalyzing(false);
    }
  }

  async function handleStreamEvent(eventText: string) {
    const lines = eventText
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    const messages = lines
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.replace(/^data:\s*/, ""));

    for (const message of messages.length ? messages : [eventText.trim()]) {
      if (!message || message === "[DONE]") continue;

      const parsed = JSON.parse(message) as Record<string, unknown>;
      const parsedJobId = typeof parsed.job_id === "string" ? parsed.job_id : "";
      if (parsedJobId) {
        activeRenderJobIdRef.current = parsedJobId;
        setRenderJobId(parsedJobId);
      }
      const parsedRequestId = typeof parsed.request_id === "string" ? parsed.request_id : "";
      if (parsedRequestId) activeOperationIdRef.current = parsedRequestId;
      const error = typeof parsed.error === "string" ? parsed.error : "";
      if (error) throw new Error(error);

      const step = typeof parsed.step === "string" ? parsed.step : "";
      const status = typeof parsed.status === "string" ? parsed.status : "";
      const phase = typeof parsed.phase === "string" ? parsed.phase : "";
      const detail = typeof parsed.detail === "string" ? parsed.detail : "";
      const stats = parsed.stats && typeof parsed.stats === "object" ? (parsed.stats as ProcessingStats) : null;
      const parsedProgress =
        typeof parsed.progress === "number"
          ? parsed.progress
          : typeof parsed.progress === "string"
            ? Number(parsed.progress)
            : null;
      const videoUrl = typeof parsed.video_url === "string" ? parsed.video_url : "";
      const subtitleUrl = typeof parsed.subtitle_url === "string" ? parsed.subtitle_url : "";
      const sourceVideoPath = typeof parsed.source_video_path === "string" ? parsed.source_video_path : "";
      const nextSegments = Array.isArray(parsed.segments) ? (parsed.segments as ScriptSegment[]) : null;
      const nextFlashTextTracks = Array.isArray(parsed.flash_text_tracks) ? (parsed.flash_text_tracks as FlashTextTrack[]) : null;
      const nextCaptionSuggestions = Array.isArray(parsed.caption_suggestions)
        ? parsed.caption_suggestions.filter((item): item is string => typeof item === "string").slice(0, 3)
        : null;
      const nextProgress = parsedProgress !== null ? clampProgress(parsedProgress) : progress;
      const statusLabel = phase ? phaseLabel(phase) : step ? normalizeStage(step) : statusText;
      const nextDetail = detail || statsSummary(stats) || (step && phase ? normalizeStage(step) : "");

      if (phase) {
        setProcessingPhase(phase);
        const nextStep = phaseToStep(phase);
        if (nextStep !== null) setActiveStep(nextStep);
      }
      if (stats) setProcessingStats(stats);
      if (nextDetail) setProcessingDetail(nextDetail);

      if (step) {
        setStatusText(statusLabel);
        const inferredProgress = progressFromStep(step);
        if (parsedProgress === null && inferredProgress !== null) setProgress(inferredProgress);
      }
      if (parsedProgress !== null) setProgress(nextProgress);
      if (step || phase || detail || stats) pushProcessingActivity(phase, statusLabel, nextDetail || normalizeStage(step), nextProgress, stats);
      if (status === "success") {
        setActiveStep(4);
        setProgress(100);
        setProcessingPhase("complete");
        setProcessingDetail("Backend đã xử lý xong. Có thể xem lại kết quả trước khi tải xuống.");
        setStatusText("Hoàn tất - xem demo trước khi tải xuống");
      }
      if (videoUrl) setResultVideoUrl(mediaUrl(videoUrl));
      if (subtitleUrl) setResultSubtitleUrl(mediaUrl(subtitleUrl));
      if (sourceVideoPath) setSourceVideoUrl(mediaUrl(sourceVideoPath));
      if (nextSegments) {
        const normalizedSegments = nextSegments.map((segment) => normalizeSegment(segment, config.voiceModel));
        setSegments(normalizedSegments);
        setSavedDemoSegments(null);
        setSelectedSegmentId(normalizedSegments[0]?.id ?? null);
        setActiveStep(3);
        setProgress(100);
        setProcessingPhase("complete");
        setProcessingStats({ segments: nextSegments.length });
        setProcessingDetail("Timeline đã sẵn sàng để chỉnh sửa và chọn giọng.");
        setStatusText(`Đã tách ${nextSegments.length} đoạn thoại`);
      }
      if (nextFlashTextTracks) setFlashTextTracks(nextFlashTextTracks);
      if (nextCaptionSuggestions) setCaptionSuggestions(nextCaptionSuggestions);
    }
  }

  async function httpErrorMessage(response: Response) {
    const text = await response.text();
    try {
      const parsed = JSON.parse(text) as { detail?: unknown };
      if (typeof parsed.detail === "string") return parsed.detail;
    } catch {
      // Use the raw body below when the backend does not return JSON.
    }
    return text || `HTTP ${response.status}`;
  }

  async function readEventStream(response: Response) {
    if (!response.ok) throw new Error(await httpErrorMessage(response));
    if (!response.body) throw new Error("Backend khong tra stream.");
    const responseJobId = response.headers.get("X-Render-Job-ID");
    if (responseJobId) {
      activeRenderJobIdRef.current = responseJobId;
      setRenderJobId(responseJobId);
    }
    const responseOperationId = response.headers.get("X-Operation-ID");
    if (responseOperationId) activeOperationIdRef.current = responseOperationId;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split(/\r?\n\r?\n/);
      buffer = events.pop() ?? "";
      for (const event of events) await handleStreamEvent(event);
    }
    if (buffer.trim()) await handleStreamEvent(buffer);
  }

  async function cancelCurrentOperation() {
    if (!isAnalyzing && !isRendering) return;
    cancelRequestedRef.current = true;
    const controller = activeRequestControllerRef.current;
    const operationId = activeOperationIdRef.current;
    const jobId = activeRenderJobIdRef.current || renderJobId;
    try {
      if (isRendering && jobId) {
        await fetch(`${RENDER_JOB_URL}/${jobId}/cancel`, { method: "POST" });
      } else {
        await fetch(CANCEL_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ request_id: operationId || null, hard: true }),
        });
      }
    } catch {
      // Hard cancellation may terminate the BE before the response reaches the browser.
    } finally {
      controller?.abort();
      activeRequestControllerRef.current = null;
      activeOperationIdRef.current = "";
      activeRenderJobIdRef.current = "";
      setIsAnalyzing(false);
      setIsRendering(false);
      setStatusText("Đã hủy và gửi lệnh dừng process BE");
      showToast({ type: "success", message: "Đã gửi lệnh hủy cưỡng bức tới backend." });
    }
  }

  function assertCloneSelectionUnchanged(expectedRevision: number) {
    if (cloneSelectionRevisionRef.current !== expectedRevision) {
      throw new Error("Đoạn giọng mẫu đã thay đổi trong lúc tải lên. Hãy render lại với đoạn 3 giây mới.");
    }
  }

  async function uploadCloneReference(expectedRevision: number): Promise<string | null> {
    if (config.voiceMode === "system") return null;
    assertCloneSelectionUnchanged(expectedRevision);
    if (!preparedCloneClip) {
      throw new Error("Đã chọn luồng clone nhưng chưa có đoạn giọng mẫu hợp lệ.");
    }

    const formData = new FormData();
    formData.append("audio", preparedCloneClip.file);
    formData.append("clip_duration_seconds", String(VOICE_REFERENCE_SECONDS));
    formData.append("selection_start_seconds", preparedCloneClip.start.toFixed(6));
    formData.append("selection_end_seconds", preparedCloneClip.end.toFixed(6));
    const response = await fetch(VOICE_REFERENCE_URL, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) throw new Error(await response.text());

    const payload = (await response.json()) as { path?: string };
    if (!payload.path) throw new Error("Backend không trả đường dẫn giọng mẫu.");
    assertCloneSelectionUnchanged(expectedRevision);
    return payload.path;
  }

  async function renderFinalVideo() {
    if (!file && !sourceVideoUrl) {
      showToast({ type: "error", message: "Chọn video trước khi render." });
      return;
    }
    if (!ensureCopyrightPreflight()) return;
    if (segments.length === 0) {
      showToast({ type: "error", message: "Bấm Tách script trước để vào màn demo/review, rồi Lưu demo mới render được." });
      return;
    }
    if (!savedDemoSegments?.length || hasUnsavedDemoChanges) {
      showToast({ type: "error", message: "Bấm Lưu demo trước khi render để giữ đúng layout/blur đã chỉnh." });
      return;
    }
    if (config.voiceMode === "clone" && !preparedCloneClip) {
      showToast({ type: "error", message: "Chọn và cắt đoạn giọng mẫu trước khi render bằng giọng clone." });
      return;
    }
    setIsRendering(true);
    setRenderJobId("");
    const controller = new AbortController();
    activeRequestControllerRef.current = controller;
    cancelRequestedRef.current = false;
    activeOperationIdRef.current = "";
    activeRenderJobIdRef.current = "";
    setActiveStep(3);
    resetProcessing("Đang bắt đầu render bản demo...", 5, "voice");
    setResultVideoUrl("");
    setResultSubtitleUrl("");

    try {
      if (segments.length > 0) {
        const baseSegments = savedDemoSegments ?? [];
        if (config.voiceMode === "system" && baseSegments.some((segment) => !segment.voice_model?.trim())) {
          throw new Error("Mỗi đoạn thoại phải có một giọng hệ thống hợp lệ.");
        }
        const systemVoiceModel = config.voiceModel.trim() || scriptDefaultVoice();
        const cloneSelectionRevision = cloneSelectionRevisionRef.current;
        const cloneReferenceAudioPath = await uploadCloneReference(cloneSelectionRevision);
        assertCloneSelectionUnchanged(cloneSelectionRevision);
        const renderSegments = baseSegments.map((segment) => ({
          ...segment,
          // Send the effective UI voice on every segment.  The backend keeps
          // per-segment voice selection, but must never receive an empty
          // setup when the user selected a default voice in the panel.
          voice_model: segment.voice_model?.trim() || systemVoiceModel,
          subtitle_style: normalizeSubtitleStyle(segment.subtitle_style),
          blur_style: normalizeBlurStyle(segment.blur_style),
        }));
        const renderPayload = {
          source_video_path: sourceVideoUrl || `/media/${sessionIdRef.current}_source.mp4`,
          source_language: config.sourceLanguage === "auto" ? null : config.sourceLanguage,
          target_language: config.targetLanguage,
          translation_provider: config.translationProvider,
          translation_model: config.translationModel,
          voice_model: systemVoiceModel,
          voice_mode: config.voiceMode,
          clone_reference_audio_path: cloneReferenceAudioPath,
          tts_device: config.ttsDevice,
          background_volume: 0,
          tts_volume: 1,
          burn_subtitles: savedTextLayerEnabled ?? true,
          mock_tts: false,
          copyright_confirmed: config.copyrightAcknowledged,
          copyright_source: config.copyrightSource,
          copyright_notes: config.copyrightNotes.trim(),
          vocal_separation: config.vocalSeparation,
          original_vocal_gain: config.originalVocalGain,
          accompaniment_gain: config.accompanimentGain,
          ocr_fallback: config.ocrFallback,
          ocr_force: config.ocrForce,
          ocr_model: config.ocrModel,
          ocr_interval_seconds: config.ocrIntervalSeconds,
          ocr_crop_bottom_ratio: config.ocrCropBottomRatio,
          flash_text_enabled: config.flashTextEnabled,
          flash_text_mode: config.flashTextMode,
          flash_text_min_confidence: config.flashTextMinConfidence,
          flash_text_max_duration_s: config.flashTextMaxDuration,
          flash_text_tracks: flashTextTracks,
          asr_engine: config.asrEngine,
          whisper_model: config.whisperModel,
          whisper_beam_size: config.whisperBeamSize,
          segment_language_detection: config.segmentLanguageDetection,
          soft_timing_fit: config.softTimingFit,
          timing_max_drift_s: config.timingMaxDrift,
          timing_min_gap_s: config.timingMinGap,
          timing_max_atempo: config.timingMaxAtempo,
          voice_speed: config.voiceSpeed,
          segments: renderSegments,
        };

        const readStoredVideoFile = async () => {
          if (!sessionIdRef.current) return null;
          try {
            return await readSessionFile(`${sessionIdRef.current}:video`);
          } catch {
            return null;
          }
        };
        const postRenderWithUpload = async (videoFile: File) => {
          await persistVideoFileForRetry(videoFile);
          const formData = new FormData();
          formData.append("video", videoFile);
          formData.append("payload", JSON.stringify(renderPayload));
          return fetch(RENDER_SCRIPT_UPLOAD_URL, {
            method: "POST",
            headers: { Accept: "text/event-stream" },
            body: formData,
            signal: controller.signal,
          });
        };
        const postRenderWithSourcePath = () =>
          fetch(RENDER_SCRIPT_URL, {
            method: "POST",
            headers: {
              Accept: "text/event-stream",
              "Content-Type": "application/json",
            },
            body: JSON.stringify(renderPayload),
            signal: controller.signal,
          });

        const renderVideoFile = file || (await readStoredVideoFile());
        if (renderVideoFile && renderVideoFile !== file) {
          setFile(renderVideoFile);
          if (!previewUrl) setPreviewUrl(URL.createObjectURL(renderVideoFile));
        }

        if (renderVideoFile) {
          await readEventStream(await postRenderWithUpload(renderVideoFile));
        } else {
          try {
            await readEventStream(await postRenderWithSourcePath());
          } catch (error) {
            const message = error instanceof Error ? error.message : "";
            const fallbackFile = message.includes("Source video not found")
              ? await readStoredVideoFile()
              : null;
            if (!fallbackFile) {
              throw new Error("Backend đã mất source video. Chọn lại video gốc để render lại pipeline và mix voice.");
            }

            setFile(fallbackFile);
            if (!previewUrl) setPreviewUrl(URL.createObjectURL(fallbackFile));
            setStatusText("File nguồn trên backend đã mất, đang gửi lại video gốc để render lại từ đầu...");
            await readEventStream(await postRenderWithUpload(fallbackFile));
          }
        }
        setSavedDemoSegments(copyDemoSegments(renderSegments, systemVoiceModel));
      }
      showToast({ type: "success", message: "Đã xuất video hoàn chỉnh." });
    } catch (error) {
      showToast({
        type: "error",
        message: error instanceof Error ? error.message : "Render thất bại.",
      });
      setStatusText("Render thất bại");
    } finally {
      if (activeRequestControllerRef.current === controller) activeRequestControllerRef.current = null;
      setIsRendering(false);
    }
  }

  async function downloadResultFile(url: string, kind: "video" | "subtitle", fallbackName: string, successLabel: string) {
    if (!url) return;
    try {
      const { savedInFolder } = await downloadUrlToDestination(url, kind, fallbackName);
      if (savedInFolder) {
        showToast({ type: "success", message: `${successLabel} vào thư mục đã cấu hình.` });
      }
    } catch (error) {
      showToast({
        type: "error",
        message: error instanceof Error ? error.message : `Không tải được file ${kind === "video" ? "video" : "SRT"}.`,
      });
    }
  }

  async function downloadResultSubtitle() {
    await downloadResultFile(resultSubtitleUrl, "subtitle", "capcut_subtitles.srt", "Đã lưu SRT");
  }

  async function downloadResultVideo() {
    await downloadResultFile(resultVideoUrl, "video", "dubbed_video.mp4", "Đã lưu video");
  }

  return (
    <main className="min-h-screen bg-slate-50 text-slate-950">
      <header className="flex h-16 items-center justify-between border-b border-slate-200 bg-white px-6">
        <section className="flex items-center gap-3" aria-label="Thương hiệu ứng dụng">
          <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-600 text-white" aria-hidden="true">
            <Video size={20} />
          </span>
          <hgroup>
            <h1 className="text-base font-semibold">Video Clone</h1>
            <p className="text-xs uppercase tracking-wide text-slate-500">Studio dịch thuật & lồng tiếng AI</p>
          </hgroup>
        </section>

        <nav className="hidden items-center gap-2 text-sm font-medium text-slate-600 lg:flex" aria-label="Chức năng chính">
          <button type="button" onClick={() => setActiveWorkspace("clone")} className={`rounded-md px-4 py-2 ${activeWorkspace === "clone" ? "border border-blue-200 bg-blue-50 text-blue-700" : "hover:bg-slate-100"}`}>Clone Video</button>
          <button type="button" onClick={() => setActiveWorkspace("short")} className={`rounded-md px-4 py-2 ${activeWorkspace === "short" ? "border border-blue-200 bg-blue-50 text-blue-700" : "hover:bg-slate-100"}`}>Short Video ≤2p</button>
          <button type="button" onClick={() => setActiveWorkspace("gen")} className={`rounded-md px-4 py-2 ${activeWorkspace === "gen" ? "border border-blue-200 bg-blue-50 text-blue-700" : "hover:bg-slate-100"}`}>Gen Video</button>
          <button type="button" onClick={() => setActiveWorkspace("setup")} className={`rounded-md px-4 py-2 ${activeWorkspace === "setup" ? "border border-blue-200 bg-blue-50 text-blue-700" : "hover:bg-slate-100"}`}>Thiết lập tải</button>
          <button className="rounded-md px-4 py-2 hover:bg-slate-100">Hướng Dẫn</button>
        </nav>

        <p
          className={`hidden items-center gap-2 rounded-md px-3 py-2 text-xs font-semibold md:flex ${
            sessionSaveStatus === 'error' ? 'bg-amber-50 text-amber-700' : 'bg-blue-50 text-blue-700'
          }`}
          aria-live='polite'
        >
          {sessionSaveStatus === 'error' ? <AlertCircle size={14} /> : <CheckCircle2 size={14} />}
          {sessionSaveStatus === 'loading'
            ? 'Đang lưu/khôi phục session'
            : sessionSaveStatus === 'saved'
              ? 'Session đã tự lưu'
              : 'Không thể tự lưu session'}
        </p>

        <p className="flex items-center gap-2 rounded-md bg-emerald-50 px-3 py-2 text-xs font-semibold text-emerald-700" aria-label="Trạng thái phần cứng CUDA RTX">
          <span className="h-2 w-2 rounded-full bg-emerald-500" aria-hidden="true" />
          CUDA / RTX
        </p>
      </header>

      {activeWorkspace === "setup" ? (
        <DownloadSetup onBack={() => setActiveWorkspace("clone")} />
      ) : activeWorkspace === "gen" ? (
        <GenVideoPipeline />
      ) : activeWorkspace === "short" ? (
        <ShortVideoWorkspace onOpenClone={() => setActiveWorkspace("clone")} />
      ) : (
        <section
          className="grid min-h-[calc(100vh-64px)] min-w-0 lg:grid-cols-[360px_minmax(0,1fr)]"
          aria-label="Không gian làm việc lồng tiếng video"
          onDragOver={(event) => event.preventDefault()}
          onDrop={onDrop}
        >
        <aside className="min-w-0 overflow-hidden border-r border-slate-200 bg-white p-5">
          <header className="flex items-center justify-between">
            <h2 className="text-xs font-bold uppercase tracking-wide text-slate-500">Dự án</h2>
            <button type='button' onClick={() => void startNewSession()} className="inline-flex items-center gap-1 text-xs font-semibold text-blue-700">
              <RefreshCw size={13} />
              Mới
            </button>
          </header>

          <figure
            className="relative mt-4 overflow-hidden rounded-lg border border-slate-200 bg-slate-950"
            onDragOver={(event) => {
              event.preventDefault();
              event.stopPropagation();
              event.dataTransfer.dropEffect = "copy";
            }}
            onDrop={onDrop}
          >
            {currentVideoUrl ? (
              <div className="relative">
                <video className="aspect-[9/14] w-full object-cover" src={currentVideoUrl} controls />
                <label className="absolute inset-x-3 bottom-3 cursor-pointer rounded-md bg-slate-950/75 px-3 py-2 text-center text-xs font-semibold text-white">
                  Thả video mới vào đây hoặc chọn file
                  <input ref={inputRef} className="hidden" type="file" accept="video/*,.mkv" onChange={onFileChange} />
                </label>
              </div>
            ) : (
              <label
                className="flex aspect-[9/14] cursor-pointer flex-col items-center justify-center bg-slate-100 p-8 text-center text-slate-600"
              >
                <input ref={inputRef} className="hidden" type="file" accept="video/*,.mkv" onChange={onFileChange} />
                <UploadCloud className="mb-3 text-blue-600" size={34} />
                <span className="text-sm font-semibold">Thả video hoặc chọn file</span>
                <span className="mt-2 text-xs">MP4, MOV, MKV, WEBM</span>
              </label>
            )}
          </figure>

          {file && (
            <article className="mt-3 flex items-center justify-between rounded-md border border-slate-200 px-3 py-2">
              <header className="min-w-0">
                <p className="truncate text-sm font-semibold">{file.name}</p>
                <p className="text-xs text-slate-500">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
              </header>
              <button
                className="flex h-8 w-8 items-center justify-center rounded-md hover:bg-slate-100"
                onClick={() => {
                  setFile(null);
                  setPreviewUrl("");
                  setSourceVideoUrl("");
                  setSegments([]);
                  setFlashTextTracks([]);
                  resetProcessing("Chưa có video");
                }}
              >
                <X size={16} />
              </button>
            </article>
          )}

          <form className="mt-5 grid min-w-0 gap-4 border-t border-slate-200 pt-5" aria-label="Cấu hình lồng tiếng">
            <SelectField label="Nhận dạng giọng nói (ASR)" value={config.asrModel} options={asrModels} onChange={(value) => setConfig((current) => ({ ...current, asrModel: value }))} />
            <fieldset className="grid gap-3 rounded-md border border-violet-200 bg-violet-50/40 p-3">
              <legend className="px-1 text-sm font-semibold text-violet-800">Thiết lập ASR chi tiết</legend>
              <div className="grid grid-cols-2 gap-2">
                <SelectField label="Engine" value={config.asrEngine} options={pipelineAsrEngines} onChange={(value) => setConfig((current) => ({ ...current, asrEngine: value as StudioConfig["asrEngine"] }))} />
                <SelectField label="Whisper model" value={config.whisperModel} options={pipelineWhisperModels} onChange={(value) => setConfig((current) => ({ ...current, whisperModel: value }))} />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <label className="grid gap-1 text-xs font-medium">Beam size<select value={config.whisperBeamSize} onChange={(event) => setConfig((current) => ({ ...current, whisperBeamSize: Number(event.target.value) }))} className="rounded border border-violet-200 bg-white px-2 py-2">{[1, 3, 5, 7, 10].map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
                <label className="flex items-center gap-2 pt-5 text-xs font-medium"><input type="checkbox" checked={config.segmentLanguageDetection} onChange={(event) => setConfig((current) => ({ ...current, segmentLanguageDetection: event.target.checked }))} /> Nhận diện ngôn ngữ từng đoạn</label>
              </div>
            </fieldset>
            <SelectField label="Ngôn ngữ gốc" value={config.sourceLanguage} options={pipelineSourceLanguages} onChange={(value) => setConfig((current) => ({ ...current, sourceLanguage: value }))} />
            <SelectField label="Ngôn ngữ dịch" value={config.targetLanguage} options={targetLanguageOptions} onChange={(value) => setConfig((current) => ({ ...current, targetLanguage: value }))} />
            <SelectField
              label="Model dịch thuật (9Router)"
              value={config.translationModel}
              options={translationModels}
              onChange={(value) => {
                translationModelTouchedRef.current = true;
                setConfig((current) => ({ ...current, translationModel: value }));
              }}
            />
            <SelectField label="Model OCR" value={config.ocrModel} options={pipelineOcrModels} onChange={(value) => setConfig((current) => ({ ...current, ocrModel: value }))} />
            <fieldset className="grid gap-3 rounded-md border border-slate-200 bg-slate-50 p-3">
              <legend className="px-1 text-sm font-semibold text-slate-700">OCR phụ đề / video silent</legend>
              <label className="flex items-start gap-3 text-sm font-medium text-slate-700">
                <input
                  type="checkbox"
                  checked={config.ocrFallback}
                  onChange={(event) => setConfig((current) => ({ ...current, ocrFallback: event.target.checked }))}
                  className="mt-1 h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                />
                <span>Tự dùng OCR khi ASR không có thoại hoặc script quá ít</span>
              </label>
              <label className="flex items-start gap-3 text-sm font-medium text-slate-700">
                <input
                  type="checkbox"
                  checked={config.ocrForce}
                  onChange={(event) => setConfig((current) => ({ ...current, ocrForce: event.target.checked }))}
                  className="mt-1 h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                />
                <span>Ép dùng OCR cho video có chữ/phụ đề trên màn hình</span>
              </label>
              <div className="grid grid-cols-2 gap-2">
                <label className="grid gap-1 text-xs font-medium">Khoảng quét OCR (giây)<input type="number" min="0.25" max="5" step="0.05" value={config.ocrIntervalSeconds} disabled={!config.ocrFallback && !config.ocrForce} onChange={(event) => setConfig((current) => ({ ...current, ocrIntervalSeconds: Math.max(0.25, Math.min(5, Number(event.target.value) || 0.75)) }))} className="rounded border border-slate-300 bg-white px-2 py-2" /></label>
                <label className="grid gap-1 text-xs font-medium">Vùng phụ đề phía dưới (%)<input type="number" min="12" max="85" step="1" value={Math.round(config.ocrCropBottomRatio * 100)} disabled={!config.ocrFallback && !config.ocrForce} onChange={(event) => setConfig((current) => ({ ...current, ocrCropBottomRatio: Math.max(0.12, Math.min(0.85, (Number(event.target.value) || 35) / 100)) }))} className="rounded border border-slate-300 bg-white px-2 py-2" /></label>
              </div>
            </fieldset>
            <fieldset className="grid gap-3 rounded-md border border-amber-200 bg-amber-50/50 p-3">
              <legend className="px-1 text-sm font-semibold text-amber-800">Che chữ hiệu ứng chớp nhoáng</legend>
              <label className="flex items-start gap-3 text-sm font-medium text-slate-700">
                <input
                  type="checkbox"
                  checked={config.flashTextEnabled}
                  onChange={(event) => setConfig((current) => ({ ...current, flashTextEnabled: event.target.checked }))}
                  className="mt-1 h-4 w-4 rounded border-amber-300 text-amber-600 focus:ring-amber-500"
                />
                <span>
                  Tự detect và blur chữ lớn trên màn hình
                  <span className="mt-1 block text-xs font-normal text-amber-800">Chỉ quét vùng giữa/phía trên, bỏ qua subtitle phía dưới và không gọi OCR dịch thuật.</span>
                </span>
              </label>
              {config.flashTextEnabled && (
                <div className="grid grid-cols-2 gap-2">
                  <label className="grid gap-1 text-xs font-medium">Độ quét<select value={config.flashTextMode} onChange={(event) => setConfig((current) => ({ ...current, flashTextMode: event.target.value as StudioConfig["flashTextMode"] }))} className="rounded border border-amber-200 bg-white px-2 py-2"><option value="balanced">Cân bằng (mỗi 2 frame)</option><option value="strict">Khắt khe (mọi frame)</option></select></label>
                  <label className="grid gap-1 text-xs font-medium">Thời lượng tối đa (giây)<input type="number" min="0.1" max="10" step="0.1" value={config.flashTextMaxDuration} onChange={(event) => setConfig((current) => ({ ...current, flashTextMaxDuration: Math.max(0.1, Math.min(10, Number(event.target.value) || 3)) }))} className="rounded border border-amber-200 bg-white px-2 py-2" /></label>
                </div>
              )}
              {flashTextTracks.length > 0 && <p className="text-xs font-semibold text-amber-800">Đã phát hiện {flashTextTracks.length} cụm chữ overlay; sẽ che đúng timeline khi render.</p>}
            </fieldset>
            <fieldset className="grid gap-3 rounded-md border border-emerald-200 bg-emerald-50/40 p-3">
              <legend className="px-1 text-sm font-semibold text-emerald-800">Khớp voice theo timeline gốc</legend>
              <label className="flex items-center gap-2 text-sm font-medium"><input type="checkbox" checked={config.softTimingFit} onChange={(event) => setConfig((current) => ({ ...current, softTimingFit: event.target.checked }))} /> Dồn nhẹ vào khoảng lặng kế tiếp</label>
              <p className="text-xs text-emerald-800">Gap tối thiểu chỉ áp dụng cho khoảng lặng thật giữa hai câu; hai cue chạm nhau sẽ được nối voice liên tục.</p>
              <div className="grid grid-cols-2 gap-2">
                <label className="grid gap-1 text-xs font-medium">Drift tối đa (giây)<input type="number" min="0" max="10" step="0.05" value={config.timingMaxDrift} onChange={(event) => setConfig((current) => ({ ...current, timingMaxDrift: Math.max(0, Math.min(10, Number(event.target.value) || 0)) }))} className="rounded border border-emerald-200 bg-white px-2 py-2" /></label>
                <label className="grid gap-1 text-xs font-medium">Gap tối thiểu (giây)<input type="number" min="0" max="2" step="0.01" value={config.timingMinGap} onChange={(event) => setConfig((current) => ({ ...current, timingMinGap: Math.max(0, Math.min(2, Number(event.target.value) || 0)) }))} className="rounded border border-emerald-200 bg-white px-2 py-2" /></label>
                <label className="grid gap-1 text-xs font-medium">Nén tốc độ tối đa<select value={config.timingMaxAtempo} onChange={(event) => setConfig((current) => ({ ...current, timingMaxAtempo: Number(event.target.value) }))} className="rounded border border-emerald-200 bg-white px-2 py-2"><option value="1.05">1.05x</option><option value="1.08">1.08x</option><option value="1.1">1.10x</option></select></label>
                <label className="grid gap-1 text-xs font-medium">Tốc độ giọng đọc<select value={config.voiceSpeed} onChange={(event) => setConfig((current) => ({ ...current, voiceSpeed: Number(event.target.value) }))} className="rounded border border-emerald-200 bg-white px-2 py-2"><option value="0.9">0.90x</option><option value="1">1.00x</option><option value="1.1">1.10x</option></select></label>
              </div>
            </fieldset>
            <fieldset className="grid gap-3 rounded-md border border-purple-200 bg-purple-50/50 p-3">
              <legend className="px-1 text-sm font-semibold text-purple-800">Âm thanh gốc</legend>
              <label className="flex items-start gap-3 text-sm font-medium text-slate-700">
                <input
                  type="checkbox"
                  checked={config.vocalSeparation}
                  onChange={(event) => setConfig((current) => ({ ...current, vocalSeparation: event.target.checked }))}
                  className="mt-1 h-4 w-4 rounded border-purple-300 text-purple-600 focus:ring-purple-500"
                />
                <div>
                  <span className="font-semibold text-purple-900">Bật tách giọng và nhạc nền</span>
                  <p className="mt-0.5 text-xs text-purple-700">Mặc định tắt. Demucs chỉ chạy khi bạn bật tùy chọn này; các mức bên dưới điều khiển bản trộn cuối.</p>
                </div>
              </label>
              <div className="grid gap-3 rounded-md border border-purple-200 bg-white/70 p-3">
                <label className="grid gap-1 text-xs font-medium text-slate-700">
                  <span className="flex items-center justify-between gap-2">
                    <span>Giữ lại giọng nói gốc</span>
                    <output className="font-semibold text-purple-800">{Math.round(config.originalVocalGain * 100)}%</output>
                  </span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={config.originalVocalGain}
                    disabled={!config.vocalSeparation}
                    onChange={(event) => setConfig((current) => ({ ...current, originalVocalGain: Number(event.target.value) }))}
                    className="accent-purple-600"
                  />
                  <span className="text-[11px] font-normal text-slate-500">0% = loại bỏ giọng gốc, 100% = giữ nguyên giọng gốc.</span>
                </label>
                <label className="grid gap-1 text-xs font-medium text-slate-700">
                  <span className="flex items-center justify-between gap-2">
                    <span>Giữ lại nhạc nền & hiệu ứng</span>
                    <output className="font-semibold text-purple-800">{Math.round(config.accompanimentGain * 100)}%</output>
                  </span>
                  <input
                    type="range"
                    min="0"
                    max="1.2"
                    step="0.05"
                    value={config.accompanimentGain}
                    disabled={!config.vocalSeparation}
                    onChange={(event) => setConfig((current) => ({ ...current, accompanimentGain: Number(event.target.value) }))}
                    className="accent-purple-600"
                  />
                  <span className="text-[11px] font-normal text-slate-500">100% là mức nền chuẩn; có thể tăng tối đa 120%.</span>
                </label>
                <div className="flex flex-wrap gap-2">
                  <button type="button" disabled={!config.vocalSeparation} onClick={() => setConfig((current) => ({ ...current, originalVocalGain: 1, accompanimentGain: 1 }))} className="rounded border border-purple-200 px-2 py-1 text-[11px] text-purple-800 disabled:cursor-not-allowed disabled:opacity-50">Giữ nguyên</button>
                  <button type="button" disabled={!config.vocalSeparation} onClick={() => setConfig((current) => ({ ...current, originalVocalGain: 0.3, accompanimentGain: 1 }))} className="rounded border border-purple-200 px-2 py-1 text-[11px] text-purple-800 disabled:cursor-not-allowed disabled:opacity-50">Giảm giọng</button>
                  <button type="button" disabled={!config.vocalSeparation} onClick={() => setConfig((current) => ({ ...current, originalVocalGain: 0, accompanimentGain: 1 }))} className="rounded border border-purple-200 px-2 py-1 text-[11px] text-purple-800 disabled:cursor-not-allowed disabled:opacity-50">Xóa giọng</button>
                  <button type="button" disabled={!config.vocalSeparation} onClick={() => setConfig((current) => ({ ...current, originalVocalGain: 1, accompanimentGain: 0 }))} className="rounded border border-purple-200 px-2 py-1 text-[11px] text-purple-800 disabled:cursor-not-allowed disabled:opacity-50">Chỉ lấy giọng</button>
                </div>
              </div>
            </fieldset>
            <fieldset className="grid gap-3 rounded-md border border-amber-200 bg-amber-50 p-3">
              <legend className="px-1 text-sm font-semibold text-amber-800">Kiểm tra quyền sử dụng</legend>
              <SelectField label="Nguồn quyền nội dung" value={config.copyrightSource} options={copyrightSourceOptions} onChange={(value) => setConfig((current) => ({ ...current, copyrightSource: value as CopyrightSource }))} />
              <label className="flex items-start gap-3 text-sm font-medium text-amber-900">
                <input
                  type="checkbox"
                  checked={config.copyrightAcknowledged}
                  onChange={(event) => setConfig((current) => ({ ...current, copyrightAcknowledged: event.target.checked }))}
                  className="mt-1 h-4 w-4 rounded border-amber-300 text-amber-600 focus:ring-amber-500"
                />
                <span>Tôi xác nhận mình có quyền sử dụng video, âm thanh, hình ảnh và phụ đề để đăng lên nền tảng mạng xã hội.</span>
              </label>
              <textarea
                value={config.copyrightNotes}
                onChange={(event) => setConfig((current) => ({ ...current, copyrightNotes: event.target.value.slice(0, 500) }))}
                placeholder="Ghi chú license / nguồn nội dung / link xác nhận quyền"
                rows={3}
                className="min-h-20 resize-y rounded-md border border-amber-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-amber-400"
              />
              {!copyrightReady && <p className="text-xs font-semibold text-amber-800">Cần chọn nguồn quyền rõ ràng và tick xác nhận trước khi xử lý.</p>}
            </fieldset>
            <fieldset className="grid gap-3 rounded-md border border-blue-200 bg-blue-50/50 p-3">
              <legend className="px-1 text-sm font-semibold text-blue-800">Nguồn giọng TTS</legend>
              <div className="grid grid-cols-2 gap-2">
                <label className={`${voiceControlsLocked ? "cursor-not-allowed opacity-60" : "cursor-pointer"} rounded-md border px-3 py-2 text-sm font-semibold ${config.voiceMode === "system" ? "border-blue-400 bg-white text-blue-700" : "border-slate-200 bg-slate-50 text-slate-600"}`}>
                  <input
                    type="radio"
                    name="voice-mode"
                    value="system"
                    checked={config.voiceMode === "system"}
                    disabled={voiceControlsLocked}
                    onChange={() => setConfig((current) => ({ ...current, voiceMode: "system" }))}
                    className="mr-2 disabled:cursor-not-allowed"
                  />
                  Giọng hệ thống
                </label>
                <label className={`${voiceControlsLocked ? "cursor-not-allowed opacity-60" : "cursor-pointer"} rounded-md border px-3 py-2 text-sm font-semibold ${config.voiceMode === "clone" ? "border-blue-400 bg-white text-blue-700" : "border-slate-200 bg-slate-50 text-slate-600"}`}>
                  <input
                    type="radio"
                    name="voice-mode"
                    value="clone"
                    checked={config.voiceMode === "clone"}
                    disabled={voiceControlsLocked}
                    onChange={() => setConfig((current) => ({ ...current, voiceMode: "clone" }))}
                    className="mr-2 disabled:cursor-not-allowed"
                  />
                  Giọng clone
                </label>
              </div>
              {config.voiceMode === "clone" && (
                <div className="grid gap-2 text-sm font-medium text-slate-700">
                  <label className="grid gap-2">
                    File giọng mẫu (chọn file dài ít nhất 3 giây, sau đó chọn đúng một đoạn sạch 3 giây)
                    <input
                      type="file"
                      accept="audio/*,.wav,.flac,.mp3,.m4a,.ogg,.opus"
                      disabled={voiceControlsLocked}
                      onChange={(event) => {
                        cloneSelectionRevisionRef.current += 1;
                        setCloneReferenceFile(event.target.files?.[0] ?? null);
                        setPreparedCloneClip(null);
                        setInitialCloneSelection(null);
                      }}
                      className="block w-full rounded-md border border-blue-200 bg-white px-3 py-2 text-sm file:mr-3 file:rounded file:border-0 file:bg-blue-50 file:px-3 file:py-1 file:font-semibold file:text-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
                    />
                  </label>
                  {cloneReferenceFile && (
                    <AudioClipSelector
                      file={cloneReferenceFile}
                      initialSelection={initialCloneSelection}
                      disabled={voiceControlsLocked}
                      onClipReady={handleCloneClipReady}
                      onError={handleCloneClipError}
                    />
                  )}
                  <span className="text-xs text-slate-500">
                    {preparedCloneClip
                      ? `Chỉ gửi đoạn ${preparedCloneClip.start.toFixed(2)}–${preparedCloneClip.end.toFixed(2)} giây từ ${cloneReferenceFile?.name}; không gửi toàn bộ file.`
                      : cloneReferenceFile
                        ? "Đang chuẩn bị đoạn âm thanh đã chọn..."
                        : "Bắt buộc chọn file. Nếu clone lỗi, tác vụ sẽ dừng; không đổi sang giọng hệ thống."}
                  </span>
                </div>
              )}
            </fieldset>
            {config.voiceMode === "system" && (
              <SelectField label="Giọng hệ thống mặc định" value={config.voiceModel} options={voiceModels} onChange={updateDefaultVoice} />
            )}
            <SelectField label="Thiết bị TTS" value={config.ttsDevice} options={[{ label: "RTX / CUDA", value: "cuda" }]} onChange={(value) => setConfig((current) => ({ ...current, ttsDevice: value as "cuda" }))} />
          </form>

          <menu className="mt-5 grid min-w-0 grid-cols-2 gap-3">
            <button
              disabled={!file || isAnalyzing || !copyrightReady}
              onClick={analyzeScript}
              className="inline-flex h-10 min-w-0 items-center justify-center gap-2 rounded-md border border-blue-200 bg-blue-50 px-2 text-sm font-semibold text-blue-700 disabled:opacity-50"
            >
              {isAnalyzing ? <Loader2 className="animate-spin" size={16} /> : <Wand2 size={16} />}
              <span className="truncate">Tách script</span>
            </button>
            <button
              disabled={(!file && !sourceVideoUrl) || isRendering || !copyrightReady || (config.voiceMode === "clone" && !preparedCloneClip)}
              onClick={renderFinalVideo}
              className="inline-flex h-10 min-w-0 items-center justify-center gap-2 rounded-md bg-blue-600 px-2 text-sm font-semibold text-white disabled:bg-slate-300"
            >
              {isRendering ? <Loader2 className="animate-spin" size={16} /> : <Play size={16} />}
              <span className="truncate">{segments.length > 0 && sourceVideoUrl ? "Render bản đã lưu" : "Render"}</span>
            </button>
            {(isAnalyzing || isRendering) && (
              <button
                type="button"
                onClick={() => void cancelCurrentOperation()}
                className="col-span-2 inline-flex h-10 items-center justify-center gap-2 rounded-md border border-red-300 bg-red-50 px-2 text-sm font-semibold text-red-700"
              >
                <X size={16} />
                <span className="truncate">Hủy & dừng process BE</span>
              </button>
            )}
          </menu>
        </aside>

        <section className="min-w-0 p-6" aria-labelledby="script-heading">
          <nav className="flex flex-wrap items-center gap-3" aria-label="Tiến trình xử lý video">
            <ol className="flex flex-wrap items-center gap-3">
            {workflowSteps.map((step, index) => {
              const done = index < activeStep;
              const active = index === activeStep;
              return (
                <li key={step} className="flex items-center gap-2 text-sm font-semibold">
                  <span className={`flex h-7 w-7 items-center justify-center rounded-full ${done ? "bg-emerald-500 text-white" : active ? "bg-blue-600 text-white" : "bg-white text-slate-500 ring-1 ring-slate-200"}`}>
                    {done ? <CheckCircle2 size={15} /> : index + 1}
                  </span>
                  <span className={active ? "text-blue-700" : "text-slate-500"}>{step}</span>
                  {index < workflowSteps.length - 1 && <span className="text-slate-300" aria-hidden="true">›</span>}
                </li>
              );
            })}
            </ol>
          </nav>

          <header className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <hgroup>
              <p className="text-xs font-bold uppercase tracking-wide text-blue-700">Kịch bản lồng tiếng</p>
              <h2 id="script-heading" className="mt-1 text-2xl font-semibold">{segments.length || 0} đoạn thoại</h2>
            </hgroup>
            <aside className="flex items-center gap-3" aria-label="Thông tin xuất bản">
              <time dateTime={`PT${Math.round(totalDuration)}S`} className="inline-flex items-center gap-2 rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600">
                <Clock3 size={15} aria-hidden="true" />
                {formatTime(totalDuration)}
              </time>
              {resultVideoUrl && (
                <button type="button" onClick={() => void downloadResultVideo()} className="inline-flex h-10 items-center gap-2 rounded-md bg-slate-950 px-4 text-sm font-semibold text-white">
                  <Download size={16} aria-hidden="true" />
                  Tải video
                </button>
              )}
              {resultSubtitleUrl && (
                <button
                  type="button"
                  onClick={downloadResultSubtitle}
                  className="inline-flex h-10 items-center gap-2 rounded-md border border-blue-200 bg-blue-50 px-4 text-sm font-semibold text-blue-700"
                >
                  <Download size={16} aria-hidden="true" />
                  T&#7843;i SRT CapCut
                </button>
              )}
            </aside>
          </header>

          <section className="mt-4 rounded-lg border border-blue-100 bg-white px-4 py-4 shadow-sm" aria-labelledby="processing-status-title">
            <header className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <hgroup className="min-w-0">
                <p className="text-xs font-bold uppercase tracking-wide text-blue-700">Tiến trình backend</p>
                <h3 id="processing-status-title" className="mt-1 text-base font-semibold text-slate-950">{statusText}</h3>
                <p className="mt-1 text-sm text-slate-600">
                  {processingDetail || "Chờ backend gửi trạng thái xử lý thực tế."}
                </p>
              </hgroup>
              <output className="inline-flex min-w-[4rem] items-center justify-center rounded-md bg-blue-50 px-3 py-2 text-lg font-bold text-blue-700" aria-label="Phần trăm xử lý">
                {progress}%
              </output>
            </header>

            <section
              className="mt-4 h-2.5 w-full overflow-hidden rounded-full bg-slate-200"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={progress}
              aria-label="Tiến độ xử lý video"
            >
              <span
                className="block h-full rounded-full bg-gradient-to-r from-blue-600 via-cyan-500 to-emerald-500 transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </section>

            <dl className="mt-4 grid gap-2 text-sm sm:grid-cols-4">
              <div className="rounded-md bg-slate-50 px-3 py-2">
                <dt className="text-xs font-semibold uppercase text-slate-500">Bước</dt>
                <dd className="mt-1 font-semibold text-slate-950">{phaseLabel(processingPhase)}</dd>
              </div>
              <div className="rounded-md bg-slate-50 px-3 py-2">
                <dt className="text-xs font-semibold uppercase text-slate-500">Engine</dt>
                <dd className="mt-1 truncate font-semibold text-slate-950">{processingStats?.engine || config.asrModel}</dd>
              </div>
              <div className="rounded-md bg-slate-50 px-3 py-2">
                <dt className="text-xs font-semibold uppercase text-slate-500">Timeline</dt>
                <dd className="mt-1 font-semibold text-slate-950">{processingStats?.segments ?? segments.length} đoạn</dd>
              </div>
              <div className="rounded-md bg-slate-50 px-3 py-2">
                <dt className="text-xs font-semibold uppercase text-slate-500">Thời lượng</dt>
                <dd className="mt-1 font-semibold text-slate-950">
                  {formatTime(processingStats?.speech_duration || processingStats?.video_duration || totalDuration)}
                </dd>
              </div>
            </dl>

            {processingActivities.length > 0 && (
              <ol className="mt-4 grid gap-2" aria-label="Hoạt động xử lý gần nhất">
                {processingActivities.map((activity) => {
                  const itemProgress = segmentProgress(activity.stats);
                  return (
                    <li key={activity.id} className="rounded-md border border-slate-100 bg-slate-50 px-3 py-3 text-sm">
                      <div className="flex items-start justify-between gap-3">
                        <span className="min-w-0">
                          <span className="block font-semibold text-slate-900">{activity.label}</span>
                          <span className="block truncate text-slate-500">{activity.detail || "Đã nhận event từ backend"}</span>
                        </span>
                        <span className="shrink-0 rounded bg-white px-2 py-0.5 text-xs font-bold text-slate-700 shadow-sm">{activityProgressLabel(activity)}</span>
                      </div>
                      {itemProgress && (
                        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-200" aria-label={`Tiến độ ${activity.label}`}>
                          <span className="block h-full rounded-full bg-blue-600 transition-all duration-500" style={{ width: `${itemProgress.percent}%` }} />
                        </div>
                      )}
                    </li>
                  );
                })}
              </ol>
            )}
          </section>

          {captionSuggestions.length > 0 && (
            <section className="mt-4 rounded-lg border border-amber-200 bg-amber-50/60 p-4 shadow-sm" aria-labelledby="caption-suggestions-title">
              <header className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-xs font-bold uppercase tracking-wide text-amber-700">Caption bám sát nội dung</p>
                  <h3 id="caption-suggestions-title" className="mt-1 text-base font-semibold text-slate-950">3 đề xuất từ kịch bản đã dịch</h3>
                  <p className="mt-1 text-sm text-slate-600">Các câu được tạo từ transcript gốc và bản dịch, không lấy ý tưởng rời khỏi nội dung video.</p>
                </div>
              </header>
              <ol className="mt-3 grid gap-2">
                {captionSuggestions.map((caption, index) => (
                  <li key={`${index}-${caption}`} className="flex items-start gap-3 rounded-md border border-amber-100 bg-white p-3">
                    <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-amber-100 text-xs font-bold text-amber-800">{index + 1}</span>
                    <p className="min-w-0 flex-1 text-sm font-medium leading-6 text-slate-800">{caption}</p>
                    <button type="button" onClick={() => void copyCaption(caption)} className="inline-flex shrink-0 items-center gap-1 rounded border border-amber-200 bg-amber-50 px-2 py-1 text-xs font-semibold text-amber-800 hover:bg-amber-100">
                      <Copy size={13} aria-hidden="true" /> Dùng
                    </button>
                  </li>
                ))}
              </ol>
            </section>
          )}

          {segments.length > 0 && currentVideoUrl && selectedSegment && (
            <SubtitleLayoutEditor
              videoUrl={currentVideoUrl}
              segments={segments}
              flashTextTracks={flashTextTracks}
              selectedSegment={selectedSegment}
              hasSavedDemo={hasSavedDemo}
              hasUnsavedDemoChanges={hasUnsavedDemoChanges}
              textLayerEnabled={textLayerEnabled}
              onTextLayerEnabledChange={setTextLayerEnabled}
              onSaveDemo={saveDemoSnapshot}
              onSelect={setSelectedSegmentId}
              onStyleChange={(id, patch) => updateSubtitleStyle(id, patch)}
              onBlurStyleChange={(id, patch) => updateBlurStyle(id, patch)}
              onSegmentChange={(id, patch) => updateSegment(id, patch)}
            />
          )}

          {resultVideoUrl && (
            <section className="mt-5 overflow-hidden rounded-lg border border-emerald-200 bg-white shadow-sm" aria-labelledby="final-preview-heading">
              <header className="flex flex-col gap-3 border-b border-emerald-100 bg-emerald-50/60 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
                <hgroup>
                  <p className="text-xs font-bold uppercase tracking-wide text-emerald-700">Demo hoàn chỉnh</p>
                  <h3 id="final-preview-heading" className="mt-1 text-base font-semibold text-slate-950">
                    Xem lại bản lồng tiếng trước khi tải xuống
                  </h3>
                </hgroup>
                <nav className="flex flex-wrap items-center gap-2" aria-label="Thao tác với demo hoàn chỉnh">

                  <button
                    type="button"
                    onClick={renderFinalVideo}
                    disabled={isRendering}
                    className="inline-flex h-10 items-center gap-2 rounded-md border border-emerald-200 bg-white px-4 text-sm font-semibold text-emerald-700 disabled:opacity-50"
                  >
                    {isRendering ? <Loader2 className="animate-spin" size={16} /> : <RefreshCw size={16} />}
                    Render lại bản đã lưu
                  </button>
                  <button type="button" onClick={() => void downloadResultVideo()} className="inline-flex h-10 items-center gap-2 rounded-md bg-slate-950 px-4 text-sm font-semibold text-white">
                    <Download size={16} aria-hidden="true" />
                    Tải xuống
                  </button>
                  {resultSubtitleUrl && (
                    <button
                      type="button"
                      onClick={downloadResultSubtitle}
                      className="inline-flex h-10 items-center gap-2 rounded-md bg-blue-600 px-4 text-sm font-semibold text-white"
                    >
                      <Download size={16} aria-hidden="true" />
                      T&#7843;i SRT CapCut
                    </button>
                  )}
                </nav>
              </header>
              <figure className="flex justify-center bg-slate-100 p-2">
                <video className="max-h-[560px] max-w-full rounded-md bg-black" src={resultVideoUrl} controls preload="metadata" />
                <figcaption className="sr-only">Video demo hoàn chỉnh đã ghép giọng đọc, âm nền và phụ đề.</figcaption>
              </figure>
              <footer className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 text-sm text-slate-600">
                <p>Bản demo này là file cuối đã render. Kiểm tra âm thanh, phụ đề và nhịp thoại trước khi tải.</p>
                <time dateTime={`PT${Math.round(totalDuration)}S`} className="font-semibold text-slate-800">
                  Thời lượng {formatTime(totalDuration)}
                </time>
              </footer>
            </section>
          )}

          <section className="mt-5 grid gap-4" aria-labelledby="timeline-heading">
            {segments.length === 0 ? (
              <section className="flex min-h-72 flex-col items-center justify-center rounded-lg border border-dashed border-slate-300 bg-white p-8 text-center" aria-labelledby="timeline-heading">
                <FileVideo className="text-blue-600" size={34} aria-hidden="true" />
                <h3 id="timeline-heading" className="mt-4 text-base font-semibold">Chưa có timeline thoại</h3>
                <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">
                  Chọn video rồi bấm “Tách script” để nhận dạng lời thoại, dịch kịch bản và tạo các đoạn timeline có thể chỉnh sửa.
                </p>
              </section>
            ) : (
              <ol className="grid gap-4" aria-label="Danh sách đoạn thoại có thể chỉnh sửa">
                {segments.map((segment, index) => (
                  <li key={segment.id}>
                    <TimelineRow
                      index={index}
                      segment={segment}
                      voiceMode={config.voiceMode}
                      isSelected={segment.id === selectedSegment?.id}
                      onSelect={() => setSelectedSegmentId(segment.id)}
                      onChange={(patch) => updateSegment(segment.id, patch)}
                      onShorten={() => autoShortenSegment(segment)}
                      isShortening={shorteningSegmentId === segment.id}
                    />
                  </li>
                ))}
              </ol>
            )}
          </section>
        </section>
        </section>
      )}

      {toast && (
        <aside className="fixed bottom-5 right-5 z-50 flex max-w-md items-start gap-3 rounded-lg border border-slate-200 bg-white p-4 shadow-soft" role="status" aria-live="polite">
          <span className={toast.type === "error" ? "text-red-600" : toast.type === "success" ? "text-emerald-600" : "text-blue-600"} aria-hidden="true">
            {toast.type === "error" ? <AlertCircle size={20} /> : <CheckCircle2 size={20} />}
          </span>
          <p className="text-sm font-medium leading-6 text-slate-800">{toast.message}</p>
        </aside>
      )}
    </main>
  );
}

function SubtitleLayoutEditor({
  videoUrl,
  segments,
  flashTextTracks,
  selectedSegment,
  hasSavedDemo,
  hasUnsavedDemoChanges,
  textLayerEnabled,
  onTextLayerEnabledChange,
  onSaveDemo,
  onSelect,
  onStyleChange,
  onBlurStyleChange,
  onSegmentChange,
}: {
  videoUrl: string;
  segments: ScriptSegment[];
  flashTextTracks: FlashTextTrack[];
  selectedSegment: ScriptSegment;
  hasSavedDemo: boolean;
  hasUnsavedDemoChanges: boolean;
  textLayerEnabled: boolean;
  onTextLayerEnabledChange: (enabled: boolean) => void;
  onSaveDemo: () => void;
  onSelect: (id: number) => void;
  onStyleChange: (id: number, patch: Partial<SubtitleStyle>) => void;
  onBlurStyleChange: (id: number, patch: Partial<BlurStyle>) => void;
  onSegmentChange: (id: number, patch: Partial<ScriptSegment>) => void;
}) {
  const frameRef = useRef<HTMLDivElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [previewTime, setPreviewTime] = useState(selectedSegment.start);
  const [videoAspectRatio, setVideoAspectRatio] = useState(16 / 9);
  const selectedStyle = normalizeSubtitleStyle(selectedSegment.subtitle_style);
  const selectedBlur = normalizeBlurStyle(selectedSegment.blur_style);
  const [frameSize, setFrameSize] = useState({ width: 0, height: 0 });
  const [videoNaturalSize, setVideoNaturalSize] = useState({ width: 0, height: 0 });
  const layoutSegments = useMemo(() => {
    const orderedSegments = [...segments].sort((left, right) => left.start - right.start || left.end - right.end);
    const runningSegment = orderedSegments.findLast((segment) => previewTime >= segment.start);
    return runningSegment ? [runningSegment] : [selectedSegment];
  }, [previewTime, segments, selectedSegment]);
  const activeFlashTextBoxes = useMemo(
    () => flashTextTracks
      .filter((track) => track.enabled && previewTime >= track.start && previewTime <= track.end && track.boxes.length > 0)
      .map((track) => ({
        track,
        box: track.boxes.reduce((closest, current) =>
          Math.abs(current.timestamp - previewTime) < Math.abs(closest.timestamp - previewTime) ? current : closest
        ),
      })),
    [flashTextTracks, previewTime],
  );

  useEffect(() => {
    const video = videoRef.current;
    if (video && !video.paused && !video.ended) return;

    const nextTime = Math.max(0, selectedSegment.start + 0.01);
    setPreviewTime(nextTime);
    if (video && Math.abs(video.currentTime - nextTime) > 0.05) {
      video.currentTime = nextTime;
    }
  }, [selectedSegment.id, selectedSegment.start]);

  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return;

    const updateFrameSize = () => {
      const rect = frame.getBoundingClientRect();
      setFrameSize({ width: rect.width, height: rect.height });
    };

    updateFrameSize();
    const observer = new ResizeObserver(updateFrameSize);
    observer.observe(frame);
    return () => observer.disconnect();
  }, []);

  function selectSegment(segment: ScriptSegment) {
    onSelect(segment.id);
    const video = videoRef.current;
    const nextTime = Math.max(0, segment.start + 0.01);
    setPreviewTime(nextTime);
    if (video) {
      video.currentTime = nextTime;
    }
  }

  function startBoxPointerEdit(
    event: PointerEvent<HTMLElement>,
    segment: ScriptSegment,
    mode: "move" | "resize",
    base: Pick<SubtitleStyle, "x" | "y" | "width" | "height">,
    applyPatch: (patch: Partial<Pick<SubtitleStyle, "x" | "y" | "width" | "height">>) => void,
  ) {
    const frame = frameRef.current;
    if (!frame) return;
    event.preventDefault();
    event.stopPropagation();
    selectSegment(segment);

    const rect = frame.getBoundingClientRect();
    const startX = event.clientX;
    const startY = event.clientY;

    function onMove(moveEvent: globalThis.PointerEvent) {
      const dx = ((moveEvent.clientX - startX) / rect.width) * 100;
      const dy = ((moveEvent.clientY - startY) / rect.height) * 100;
      if (mode === "move") {
        applyPatch({
          x: clampNumber(base.x + dx, 0, 100 - base.width),
          y: clampNumber(base.y + dy, 0, 100 - base.height),
        });
      } else {
        applyPatch({
          width: clampNumber(base.width + dx, 8, 100 - base.x),
          height: clampNumber(base.height + dy, 5, 100 - base.y),
        });
      }
    }

    function onUp() {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    }

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  function startBlurPointerEdit(
    event: PointerEvent<HTMLElement>,
    segment: ScriptSegment,
    mode: "move" | "resize",
    base: Pick<BlurStyle, "x" | "y" | "width" | "height">,
  ) {
    const frame = frameRef.current;
    if (!frame) return;
    event.preventDefault();
    event.stopPropagation();
    selectSegment(segment);

    const rect = frame.getBoundingClientRect();
    const startX = event.clientX;
    const startY = event.clientY;

    function onMove(moveEvent: globalThis.PointerEvent) {
      const dx = ((moveEvent.clientX - startX) / rect.width) * 100;
      const dy = ((moveEvent.clientY - startY) / rect.height) * 100;
      if (mode === "move") {
        onBlurStyleChange(segment.id, {
          x: clampNumber(base.x + dx, 0, 100 - base.width),
          y: clampNumber(base.y + dy, 0, 100 - base.height),
        });
      } else {
        onBlurStyleChange(segment.id, {
          width: clampNumber(base.width + dx, 5, 100 - base.x),
          height: clampNumber(base.height + dy, 4, 100 - base.y),
        });
      }
    }

    function onUp() {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    }

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }
  function numberPatch(key: keyof SubtitleStyle, value: string) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return;
    onStyleChange(selectedSegment.id, { [key]: parsed } as Partial<SubtitleStyle>);
  }

  function blurNumberPatch(key: keyof BlurStyle, value: string) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return;
    onBlurStyleChange(selectedSegment.id, { [key]: parsed } as Partial<BlurStyle>);
  }

  const previewScale = videoNaturalSize.width > 0 && frameSize.width > 0 ? frameSize.width / videoNaturalSize.width : 1;

  return (
    <section className="mt-5 overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm" aria-labelledby="subtitle-layout-heading">
      <header className="flex flex-col gap-2 border-b border-slate-100 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
        <hgroup>
          <p className="text-xs font-bold uppercase tracking-wide text-blue-700">Subtitle layout</p>
          <h3 id="subtitle-layout-heading" className="mt-1 text-base font-semibold text-slate-950">
            Click text, drag box, resize, adjust timing
          </h3>
        </hgroup>
        <div className="flex flex-col items-start gap-2 text-sm sm:items-end">
          <p className="text-slate-500">
            Đang chọn #{selectedSegment.id + 1} - {formatTime(selectedSegment.start)} - {formatTime(selectedSegment.end)}
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <span className={`rounded-full px-2.5 py-1 text-xs font-bold ${
              hasSavedDemo && !hasUnsavedDemoChanges ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"
            }`}>
              {hasSavedDemo ? (hasUnsavedDemoChanges ? "Có thay đổi chưa lưu" : "Demo đã lưu") : "Chưa lưu demo"}
            </span>
            <label className="inline-flex h-9 items-center gap-2 rounded-md border border-slate-200 bg-white px-3 text-xs font-bold text-slate-700 shadow-sm">
              <input
                type="checkbox"
                checked={textLayerEnabled}
                onChange={(event) => onTextLayerEnabledChange(event.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
              />
              Lớp chữ
            </label>
            <button
              type="button"
              onClick={onSaveDemo}
              className="inline-flex h-9 items-center gap-2 rounded-md bg-slate-950 px-3 text-xs font-bold text-white shadow-sm transition hover:bg-slate-800"
            >
              <CheckCircle2 size={14} aria-hidden="true" />
              Lưu demo
            </button>
          </div>
        </div>
      </header>

      <section className="grid gap-4 p-4 xl:grid-cols-[minmax(0,1fr)_300px]">
        <div className="flex justify-center rounded-md bg-slate-100 p-2">
          <div
            ref={frameRef}
            className="relative overflow-hidden rounded-md bg-black shadow-sm"
            style={{
              aspectRatio: `${videoAspectRatio}`,
              width: videoAspectRatio >= 1 ? "100%" : `min(100%, calc(min(72vh, 620px) * ${videoAspectRatio}))`,
              maxHeight: "min(72vh, 620px)",
            }}
          >
          <video
            ref={videoRef}
            className="h-full w-full object-cover"
            src={videoUrl}
            controls
            preload="metadata"
            onLoadedMetadata={(event) => {
              const { videoWidth, videoHeight } = event.currentTarget;
              if (videoWidth > 0 && videoHeight > 0) {
                setVideoAspectRatio(videoWidth / videoHeight);
                setVideoNaturalSize({ width: videoWidth, height: videoHeight });
              }
              const nextTime = Math.max(0, selectedSegment.start + 0.01);
              event.currentTarget.currentTime = nextTime;
              setPreviewTime(nextTime);
            }}
            onTimeUpdate={(event) => {
              setPreviewTime(event.currentTarget.currentTime);
            }}
            onSeeked={(event) => setPreviewTime(event.currentTarget.currentTime)}
          />
          <div className="pointer-events-none absolute inset-0 z-20">
            {activeFlashTextBoxes.map(({ track, box }) => (
              <div
                key={`flash-text-${track.id}`}
                className="absolute rounded-md border-2 border-amber-400/90 bg-black/25 shadow-[0_0_0_2px_rgba(251,191,36,0.25)]"
                style={{
                  left: `${box.x}%`,
                  top: `${box.y}%`,
                  width: `${box.width}%`,
                  height: `${box.height}%`,
                  backdropFilter: `blur(${Math.max(0, box.height * track.blur / 18)}px)`,
                  WebkitBackdropFilter: `blur(${Math.max(0, box.height * track.blur / 18)}px)`,
                }}
              />
            ))}
            {selectedBlur.enabled && (
              <button
                type="button"
                onPointerDown={(event) => startBlurPointerEdit(event, selectedSegment, "move", selectedBlur)}
                onClick={() => selectSegment(selectedSegment)}
                className="pointer-events-auto absolute flex touch-none select-none items-start justify-start rounded-md border-2 border-amber-300 bg-black/25 text-[10px] font-bold uppercase tracking-wide text-amber-100 shadow-[0_0_0_2px_rgba(251,191,36,0.35)] ring-2 ring-amber-300/60"
                style={{
                  left: `${selectedBlur.x}%`,
                  top: `${selectedBlur.y}%`,
                  width: `${selectedBlur.width}%`,
                  height: `${selectedBlur.height}%`,
                  backgroundColor: `rgba(0,0,0,${clampNumber(selectedBlur.opacity, 0.05, 0.95)})`,
                  backdropFilter: `blur(${Math.max(0, selectedBlur.blur * previewScale)}px)`,
                  WebkitBackdropFilter: `blur(${Math.max(0, selectedBlur.blur * previewScale)}px)`,
                }}
                title="Drag to move blur box"
              >
                <span className="m-1 rounded bg-amber-400/95 px-1.5 py-0.5 text-slate-950">Blur</span>
                <span
                  aria-hidden="true"
                  onPointerDown={(event) => startBlurPointerEdit(event, selectedSegment, "resize", selectedBlur)}
                  className="absolute bottom-0 right-0 flex h-6 w-6 translate-x-1/2 translate-y-1/2 items-center justify-center rounded bg-amber-400 text-slate-950 shadow"
                >
                  <Maximize2 size={12} />
                </span>
              </button>
            )}
          </div>
          <div className="pointer-events-none absolute inset-0 z-30">
            {textLayerEnabled && layoutSegments.map((segment) => {
              const style = selectedStyle;
              const active = segment.id === selectedSegment.id;
              const scaledFontSize = clampNumber(style.font_size * previewScale, 8, 120);
              const scaledOutlineWidth = clampNumber(style.outline_width * previewScale, 0, 12);
              const renderBoxWidthPx = Math.max(40, Math.round((videoNaturalSize.width || frameSize.width || 1) * style.width / 100));
              const previewText = wrapSubtitlePreviewLikeRender(segment.translated_text || segment.original_text, renderBoxWidthPx, style.font_size);
              return (
                <div key={segment.id} className="contents">

                  <button
                    type="button"
                    onPointerDown={(event) => startBoxPointerEdit(event, selectedSegment, "move", style, (patch) => onStyleChange(selectedSegment.id, patch))}
                    onClick={() => selectSegment(segment)}
                    className={`pointer-events-auto absolute flex touch-none select-none items-center justify-center border text-center font-bold leading-tight ${
                      active ? "border-cyan-300 bg-cyan-400/10 ring-2 ring-cyan-300" : "border-white/35 bg-black/10 hover:border-white"
                    }`}
                    style={{
                      left: `${style.x}%`,
                      top: `${style.y}%`,
                      width: `${style.width}%`,
                      height: `${style.height}%`,
                      color: style.color,
                      fontSize: `${scaledFontSize}px`,
                      textShadow: `0 0 ${scaledOutlineWidth + 1}px ${style.outline_color}, 0 1px ${scaledOutlineWidth + 2}px ${style.outline_color}`,
                      justifyContent: style.align === "left" ? "flex-start" : style.align === "right" ? "flex-end" : "center",
                      fontFamily: "Arial, sans-serif",
                      fontWeight: 700,
                      lineHeight: 1,
                      padding: "0.25rem",
                    }}
                    title="Drag to move subtitle box"
                  >
                    <span className="block max-w-full whitespace-pre-wrap break-words">{previewText}</span>
                    {active && (
                      <span
                        aria-hidden="true"
                        onPointerDown={(event) => startBoxPointerEdit(event, selectedSegment, "resize", style, (patch) => onStyleChange(selectedSegment.id, patch))}
                        className="absolute bottom-0 right-0 flex h-6 w-6 translate-x-1/2 translate-y-1/2 items-center justify-center rounded bg-cyan-500 text-white shadow"
                      >
                        <Maximize2 size={12} />
                      </span>
                    )}
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        </div>
        <aside className="grid content-start gap-3">
          <label className="grid gap-1 text-sm font-semibold text-slate-700">
            Text layer
            <textarea
              value={selectedSegment.translated_text}
              disabled={!textLayerEnabled}
              placeholder={textLayerEnabled ? "Nhập text hiển thị trên demo" : "Lớp chữ đang tắt; render chỉ chèn voice/blur"}
              onChange={(event) => onSegmentChange(selectedSegment.id, { translated_text: event.target.value })}
              className="min-h-24 resize-none rounded-md border border-slate-200 px-3 py-2 text-sm font-medium outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100 disabled:bg-slate-100 disabled:text-slate-400"
            />
          </label>

          <div className="grid grid-cols-2 gap-2">
            <NumberField label="Start" value={selectedSegment.start} step={0.05} onChange={(value) => onSegmentChange(selectedSegment.id, { start: Math.max(0, value) })} />
            <NumberField label="End" value={selectedSegment.end} step={0.05} onChange={(value) => onSegmentChange(selectedSegment.id, { end: Math.max(selectedSegment.start + 0.1, value) })} />
            <NumberField label="X %" value={selectedStyle.x} onChange={(value) => numberPatch("x", String(value))} />
            <NumberField label="Y %" value={selectedStyle.y} onChange={(value) => numberPatch("y", String(value))} />
            <NumberField label="W %" value={selectedStyle.width} onChange={(value) => numberPatch("width", String(value))} />
            <NumberField label="H %" value={selectedStyle.height} onChange={(value) => numberPatch("height", String(value))} />
          </div>

          <label className="grid gap-1 text-sm font-semibold text-slate-700">
            <span className="inline-flex items-center gap-2"><Type size={14} /> Font size</span>
            <input
              type="range"
              min={12}
              max={120}
              value={selectedStyle.font_size}
              onChange={(event) => numberPatch("font_size", event.target.value)}
            />
            <span className="text-xs text-slate-500">{selectedStyle.font_size}px render size</span>
          </label>

          <div className="grid grid-cols-2 gap-2">
            <label className="grid gap-1 text-sm font-semibold text-slate-700">
              Color
              <input type="color" value={selectedStyle.color} onChange={(event) => onStyleChange(selectedSegment.id, { color: event.target.value })} className="h-10 w-full rounded border border-slate-200" />
            </label>
            <label className="grid gap-1 text-sm font-semibold text-slate-700">
              Outline
              <input type="color" value={selectedStyle.outline_color} onChange={(event) => onStyleChange(selectedSegment.id, { outline_color: event.target.value })} className="h-10 w-full rounded border border-slate-200" />
            </label>
          </div>

          <label className="grid gap-1 text-sm font-semibold text-slate-700">
            Align
            <select
              value={selectedStyle.align}
              onChange={(event) => onStyleChange(selectedSegment.id, { align: event.target.value as SubtitleStyle["align"] })}
              className="h-10 rounded-md border border-slate-200 bg-white px-3 text-sm outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
            >
              <option value="left">Left</option>
              <option value="center">Center</option>
              <option value="right">Right</option>
            </select>
          </label>


          <section className="grid gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3">
            <label className="flex items-start gap-3 text-sm font-semibold text-slate-700">
              <input
                type="checkbox"
                checked={selectedBlur.enabled}
                onChange={(event) => onBlurStyleChange(selectedSegment.id, { enabled: event.target.checked })}
                className="mt-1 h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
              />
              <span>
                Làm mờ chữ gốc
                <span className="mt-1 block text-xs font-normal text-slate-500">Bật khi video có subtitle/logo cũ cần che phía sau bản dịch.</span>
              </span>
            </label>

            {selectedBlur.enabled && (
              <div className="grid gap-3">
                <div className="grid grid-cols-2 gap-2">
                  <NumberField label="Blur X %" value={selectedBlur.x} onChange={(value) => blurNumberPatch("x", String(value))} />
                  <NumberField label="Blur Y %" value={selectedBlur.y} onChange={(value) => blurNumberPatch("y", String(value))} />
                  <NumberField label="Blur W %" value={selectedBlur.width} onChange={(value) => blurNumberPatch("width", String(value))} />
                  <NumberField label="Blur H %" value={selectedBlur.height} onChange={(value) => blurNumberPatch("height", String(value))} />
                </div>

                <label className="grid gap-1 text-sm font-semibold text-slate-700">
                  Độ mờ
                  <input
                    type="range"
                    min={0}
                    max={48}
                    value={selectedBlur.blur}
                    onChange={(event) => blurNumberPatch("blur", event.target.value)}
                  />
                  <span className="text-xs text-slate-500">{selectedBlur.blur}px blur</span>
                </label>

                <label className="grid gap-1 text-sm font-semibold text-slate-700">
                  Nền đen
                  <input
                    type="range"
                    min={0}
                    max={0.95}
                    step={0.01}
                    value={selectedBlur.opacity}
                    onChange={(event) => blurNumberPatch("opacity", event.target.value)}
                  />
                  <span className="text-xs text-slate-500">{Math.round(selectedBlur.opacity * 100)}% opacity</span>
                </label>
              </div>
            )}
          </section>
          <button
            type="button"
            onClick={() => onStyleChange(selectedSegment.id, defaultSubtitleStyle)}
            className="inline-flex h-10 items-center justify-center gap-2 rounded-md border border-slate-200 bg-slate-50 text-sm font-semibold text-slate-700"
          >
            <Move size={14} />
            Reset box
          </button>
        </aside>
      </section>
    </section>
  );
}
function NumberField({
  label,
  value,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  step?: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="grid gap-1 text-xs font-bold uppercase tracking-wide text-slate-500">
      {label}
      <input
        type="number"
        step={step}
        value={Number(value.toFixed(step < 1 ? 2 : 0))}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-9 rounded-md border border-slate-200 px-2 text-sm font-semibold text-slate-900 outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
      />
    </label>
  );
}

function TimelineRow({
  index,
  segment,
  voiceMode,
  isSelected,
  onSelect,
  onChange,
  onShorten,
  isShortening,
}: {
  index: number;
  segment: ScriptSegment;
  voiceMode: VoiceMode;
  isSelected: boolean;
  onSelect: () => void;
  onChange: (patch: Partial<ScriptSegment>) => void;
  onShorten: () => void;
  isShortening: boolean;
}) {
  const duration = Math.max(0.1, segment.end - segment.start);
  const translationFieldId = `segment-${segment.id}-translation`;
  const speechUnits = countSpeechUnits(segment.translated_text);
  const estimatedDuration = estimateVoiceDuration(segment.translated_text);
  const needsShortening = estimatedDuration > duration * 1.12;
  const speechUnitLabel = hasCjkText(segment.translated_text) ? "ký tự" : "từ";

  return (
    <article
      onClick={onSelect}
      className={`rounded-lg border p-4 shadow-sm ${
        isSelected
          ? "border-cyan-300 bg-cyan-50/40 ring-2 ring-cyan-100"
          : needsShortening
            ? "border-rose-200 bg-rose-50/40"
            : "border-slate-200 bg-white"
      }`}
    >
      <section className="grid min-w-0 gap-4 xl:grid-cols-[110px_minmax(0,1fr)_minmax(0,1fr)_190px]" aria-label={`Đoạn thoại ${index + 1}`}>
        <header className="border-r border-slate-100 pr-3">
          <p className="text-lg font-bold">{String(index + 1).padStart(2, "0")}</p>
          <p className="mt-2 text-sm font-semibold text-blue-700">
            {formatTime(segment.start)} – {formatTime(segment.end)}
          </p>
          <button className="mt-3 flex h-9 w-9 items-center justify-center rounded-full bg-blue-50 text-blue-700">
            <Play size={15} aria-hidden="true" />
          </button>
        </header>

        <label className="grid min-w-0 gap-2">
          <span className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-rose-600">
            <Languages size={13} aria-hidden="true" />
            Ngôn ngữ gốc
          </span>
          <textarea
            value={segment.original_text}
            onChange={(event) => onChange({ original_text: event.target.value })}
            className="min-h-16 w-full min-w-0 resize-none rounded-md border border-rose-200 bg-rose-50/50 px-3 py-2 text-sm outline-none focus:border-rose-400 focus:ring-4 focus:ring-rose-100"
          />
        </label>

        <section className="grid min-w-0 gap-2">
          <div className="flex min-w-0 items-center justify-between gap-2">
            <label htmlFor={translationFieldId} className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-blue-700">
              <Wand2 size={13} aria-hidden="true" />
              Bản dịch
            </label>
            {needsShortening && (
              <button
                type="button"
                onClick={onShorten}
                disabled={isShortening}
                title="AI rút gọn theo thời lượng"
                aria-label="AI rút gọn theo thời lượng"
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-rose-200 bg-white text-rose-600 transition hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isShortening ? <Loader2 className="animate-spin" size={15} aria-hidden="true" /> : <Wand2 size={15} aria-hidden="true" />}
              </button>
            )}
          </div>
          <textarea
            id={translationFieldId}
            value={segment.translated_text}
            onChange={(event) => onChange({ translated_text: event.target.value })}
            className={`min-h-16 w-full min-w-0 resize-none rounded-md border px-3 py-2 text-sm outline-none focus:ring-4 ${
              needsShortening
                ? "border-rose-300 bg-white focus:border-rose-400 focus:ring-rose-100"
                : "border-blue-200 bg-blue-50/40 focus:border-blue-400 focus:ring-blue-100"
            }`}
          />
        </section>

        <label className="grid min-w-0 content-start gap-2">
          <span className="inline-flex items-center gap-2 text-xs font-bold uppercase tracking-wide text-slate-500">
            <Mic2 size={13} aria-hidden="true" />
            Giọng đọc
          </span>
          {voiceMode === "clone" ? (
            <p className="rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-sm font-semibold text-blue-700">
              Giọng clone duy nhất
            </p>
          ) : (
            <select
              value={segment.voice_model}
              onChange={(event) => onChange({ voice_model: event.target.value })}
              className="h-10 w-full min-w-0 truncate rounded-md border border-slate-200 bg-white px-3 text-sm font-medium outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100"
            >
              {voiceModels.map((voice) => (
                <option key={voice.value} value={voice.value}>
                  {voice.label}
                </option>
              ))}
            </select>
          )}
          <p className={`text-right text-xs ${needsShortening ? "font-semibold text-rose-600" : "text-slate-500"}`}>
            {speechUnits} {speechUnitLabel} / ~{estimatedDuration.toFixed(1)}s voice
          </p>
          {needsShortening && <p className="text-right text-xs font-bold text-rose-600">Cần rút gọn</p>}
        </label>
      </section>

      <footer className="mt-4 flex items-center gap-3" aria-label="Vị trí đoạn thoại trên timeline">
        <span className="w-10 text-xs text-slate-500">{formatTime(segment.start)}</span>
        <span className={`h-3 flex-1 overflow-hidden rounded-full ring-1 ${needsShortening ? "bg-rose-50 ring-rose-200" : "bg-blue-50 ring-blue-100"}`} aria-hidden="true">
          <span className={`block h-full rounded-full ${needsShortening ? "bg-rose-200" : "bg-blue-200"}`} style={{ width: "100%" }} />
        </span>
        <span className="w-10 text-right text-xs text-slate-500">{formatTime(segment.end)}</span>
      </footer>
    </article>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: Array<{ label: string; value: string }>;
  onChange: (value: string) => void;
}) {
  return (
    <label className="grid min-w-0 gap-2">
      <span className="inline-flex min-w-0 items-center gap-2 text-sm font-semibold text-slate-700">
        <Settings size={14} />
        <span className="truncate">{label}</span>
      </span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        title={value}
        className="block h-10 w-full min-w-0 max-w-full truncate rounded-md border border-slate-300 bg-white px-3 pr-8 text-sm font-medium text-slate-900 outline-none transition focus:border-blue-500 focus:ring-4 focus:ring-blue-100"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}
