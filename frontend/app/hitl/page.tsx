"use client";

import {
  AlertCircle,
  CheckCircle2,
  FileText,
  FileVideo,
  Loader2,
  PlayCircle,
  Save,
  UploadCloud,
  Wand2,
  X,
} from "lucide-react";
import { ChangeEvent, DragEvent, useRef, useState } from "react";

const BACKEND_URL = "http://localhost:8000";
const UPLOAD_URL = `${BACKEND_URL}/api/upload-and-extract`;
const TRANSCRIBE_URL = `${BACKEND_URL}/api/transcribe`;

type UploadResponse = {
  uuid: string;
  status: "audio_extracted";
  audio_url: string;
  video_url: string;
};

type ScriptSegment = {
  id: number;
  start: number;
  end: number;
  text: string;
  words?: Array<{ word: string; start: number; end: number }>;
};

function mediaUrl(path: string) {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://") || path.startsWith("blob:")) return path;
  return new URL(path, BACKEND_URL).toString();
}

export default function Page() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [workspaceUuid, setWorkspaceUuid] = useState("");
  const [videoUrl, setVideoUrl] = useState("");
  const [segments, setSegments] = useState<ScriptSegment[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [error, setError] = useState("");

  async function uploadVideo(nextFile: File) {
    if (!nextFile.type.startsWith("video/") && !/\.(mp4|mov|mkv|webm)$/i.test(nextFile.name)) {
      setError("Please choose a valid video file.");
      return;
    }

    const formData = new FormData();
    formData.append("video", nextFile);

    setFile(nextFile);
    setError("");
    setIsUploading(true);

    try {
      const response = await fetch(UPLOAD_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error(await response.text());

      const result = (await response.json()) as UploadResponse;
      setWorkspaceUuid(result.uuid);
      setVideoUrl(mediaUrl(result.video_url));
      setSegments([]);
    } catch (caught) {
      setWorkspaceUuid("");
      setVideoUrl("");
      setSegments([]);
      setError(caught instanceof Error ? caught.message : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  }

  async function startTranscription() {
    if (!workspaceUuid) return;

    setError("");
    setIsTranscribing(true);

    try {
      const response = await fetch(TRANSCRIBE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ uuid: workspaceUuid }),
      });

      if (!response.ok) throw new Error(await response.text());

      const result = (await response.json()) as ScriptSegment[];
      setSegments(result);
    } catch (caught) {
      setSegments([]);
      setError(caught instanceof Error ? caught.message : "Transcription failed.");
    } finally {
      setIsTranscribing(false);
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0];
    if (nextFile) void uploadVideo(nextFile);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    const nextFile = event.dataTransfer.files[0];
    if (nextFile) void uploadVideo(nextFile);
  }

  function resetUpload() {
    setFile(null);
    setWorkspaceUuid("");
    setVideoUrl("");
    setSegments([]);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  function updateSegmentText(id: number, text: string) {
    setSegments((current) =>
      current.map((segment) => (segment.id === id ? { ...segment, text } : segment)),
    );
  }

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-5 py-6">
        <header className="flex h-14 items-center justify-between border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-500">
              <Wand2 size={20} />
            </div>
            <div>
              <h1 className="text-base font-semibold">Auto-Dubbing Review</h1>
              <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
                {segments.length > 0 ? "Step 2 / Script Review" : "Step 1 / Upload"}
              </p>
            </div>
          </div>
          {workspaceUuid && (
            <button
              onClick={resetUpload}
              className="inline-flex h-9 items-center gap-2 rounded-md border border-white/10 px-3 text-sm font-semibold text-slate-200 hover:bg-white/10"
            >
              <X size={15} />
              Reset
            </button>
          )}
        </header>

        <section className="grid flex-1 items-center gap-8 py-8 lg:grid-cols-[420px_1fr]">
          <div className="space-y-5">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-indigo-300">Human-in-the-loop</p>
              <h2 className="mt-2 max-w-xl text-4xl font-semibold tracking-tight text-white">
                Upload a video and approve it before ASR starts.
              </h2>
            </div>

            <div className="grid gap-3 text-sm text-slate-300">
              <div className="flex items-center gap-3">
                <CheckCircle2 className="text-emerald-400" size={18} />
                <span>Video is saved into an isolated UUID workspace.</span>
              </div>
              <div className="flex items-center gap-3">
                <CheckCircle2 className="text-emerald-400" size={18} />
                <span>Audio is extracted with CPU FFmpeg before transcription.</span>
              </div>
              <div className="flex items-center gap-3">
                <CheckCircle2 className="text-emerald-400" size={18} />
                <span>Review and fix every transcript segment before TTS.</span>
              </div>
            </div>

            {error && (
              <div className="flex items-start gap-3 rounded-lg border border-red-400/30 bg-red-500/10 p-4 text-sm text-red-100">
                <AlertCircle className="mt-0.5 shrink-0 text-red-300" size={18} />
                <span className="leading-6">{error}</span>
              </div>
            )}
          </div>

          {!workspaceUuid ? (
            <label
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              className={`flex min-h-[460px] cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed p-8 text-center transition ${
                isDragging
                  ? "border-indigo-300 bg-indigo-500/15"
                  : "border-white/15 bg-white/[0.04] hover:border-indigo-300 hover:bg-white/[0.06]"
              }`}
            >
              <input
                ref={inputRef}
                type="file"
                accept="video/*,.mkv"
                className="hidden"
                onChange={handleFileChange}
              />

              <div className="flex h-16 w-16 items-center justify-center rounded-xl bg-indigo-500/15 text-indigo-200 ring-1 ring-indigo-300/20">
                {isUploading ? <Loader2 className="animate-spin" size={30} /> : <UploadCloud size={32} />}
              </div>
              <h3 className="mt-6 text-xl font-semibold">
                {isUploading ? "Extracting audio..." : "Drop your video here"}
              </h3>
              <p className="mt-2 max-w-sm text-sm leading-6 text-slate-400">
                {file ? file.name : "MP4, MOV, MKV, or WEBM"}
              </p>
              <span className="mt-6 inline-flex h-10 items-center justify-center rounded-md bg-white px-4 text-sm font-bold text-slate-950">
                Choose video
              </span>
            </label>
          ) : (
            <div className="space-y-4">
              <div className="rounded-xl border border-white/10 bg-white/[0.04] p-4 shadow-2xl shadow-black/20">
                <div className="overflow-hidden rounded-lg bg-black">
                  <video src={videoUrl} controls className="aspect-video w-full object-contain" />
                </div>

                <div className="mt-4 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 text-sm font-semibold text-emerald-300">
                      <FileVideo size={16} />
                      Audio extracted
                    </div>
                    <p className="mt-1 truncate text-xs text-slate-400">Workspace: {workspaceUuid}</p>
                  </div>

                  <button
                    onClick={startTranscription}
                    disabled={isTranscribing || segments.length > 0}
                    className="inline-flex h-11 shrink-0 items-center justify-center gap-2 rounded-md bg-gradient-to-r from-indigo-500 to-purple-600 px-5 text-sm font-bold text-white shadow-lg shadow-indigo-950/30 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {isTranscribing ? <Loader2 className="animate-spin" size={18} /> : <PlayCircle size={18} />}
                    {isTranscribing
                      ? "Transcribing with WhisperX..."
                      : segments.length > 0
                        ? "Transcript Ready"
                        : "Approve & Start Transcription (ASR)"}
                  </button>
                </div>
              </div>

              {segments.length > 0 && (
                <section className="rounded-xl border border-white/10 bg-slate-900/80 p-4 shadow-2xl shadow-black/20">
                  <div className="flex flex-col gap-3 border-b border-white/10 pb-4 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <div className="flex items-center gap-2 text-sm font-semibold text-indigo-200">
                        <FileText size={16} />
                        Script Editor
                      </div>
                      <p className="mt-1 text-xs text-slate-400">
                        Edit hallucinations, names, punctuation, and timing-sensitive wording before voice generation.
                      </p>
                    </div>
                    <button className="inline-flex h-10 shrink-0 items-center justify-center gap-2 rounded-md bg-white px-4 text-sm font-bold text-slate-950 transition hover:bg-indigo-100">
                      <Save size={16} />
                      Approve Script & Generate Voice (TTS)
                    </button>
                  </div>

                  <div className="mt-4 max-h-[520px] space-y-3 overflow-y-auto pr-1">
                    {segments.map((segment) => (
                      <article key={segment.id} className="rounded-lg border border-white/10 bg-white/[0.04] p-3">
                        <div className="mb-2 flex items-center justify-between gap-3 text-xs">
                          <span className="rounded-md bg-indigo-500/15 px-2 py-1 font-bold text-indigo-200">
                            #{segment.id}
                          </span>
                          <span className="font-medium text-slate-400">
                            {segment.start.toFixed(2)}s - {segment.end.toFixed(2)}s
                          </span>
                        </div>
                        <textarea
                          value={segment.text}
                          onChange={(event) => updateSegmentText(segment.id, event.target.value)}
                          className="min-h-24 w-full resize-y rounded-md border border-white/10 bg-slate-950/70 px-3 py-2 text-sm leading-6 text-white outline-none transition placeholder:text-slate-500 focus:border-indigo-300 focus:ring-4 focus:ring-indigo-500/20"
                        />
                      </article>
                    ))}
                  </div>
                </section>
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
