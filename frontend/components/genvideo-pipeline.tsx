"use client";

import {
  Bot,
  Check,
  Clapperboard,
  Download,
  FileText,
  GripVertical,
  Image as ImageIcon,
  Layers3,
  Link2,
  Loader2,
  Mic2,
  Move,
  Play,
  Plus,
  Settings2,
  Sparkles,
  Subtitles,
  Trash2,
  UploadCloud,
  Wand2,
  Workflow,
} from "lucide-react";
import { DragEvent, PointerEvent, useEffect, useRef, useState } from "react";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
const FLOW_ID = process.env.NEXT_PUBLIC_GENVIDEO_FLOW_ID || "main";
const FLOW_API_URL = `${BACKEND_URL}/api/gen-video`;

type NodeKind = "brief" | "script" | "image" | "veo" | "voice" | "subtitles" | "review" | "export";
type NodeStatus = "idle" | "ready" | "running" | "done" | "error";
type SyncStatus = "loading" | "saved" | "saving" | "error";

type PipelineNode = {
  id: string;
  kind: NodeKind;
  title: string;
  eyebrow: string;
  description: string;
  x: number;
  y: number;
  status: NodeStatus;
  settings: Record<string, string>;
};

type CatalogEntry = {
  kind: NodeKind;
  title: string;
  description: string;
  color: string;
  default_settings?: Record<string, string>;
};

type ProviderConfig = {
  name: string;
  configured: boolean;
  models: string[];
  image_models: string[];
  aspect_ratios: string[];
  resolutions: string[];
  key_source: string;
};

type FlowGraph = {
  flow_id: string;
  version: number;
  nodes: PipelineNode[];
};

type BootstrapPayload = {
  catalog: CatalogEntry[];
  graph: FlowGraph;
  provider: ProviderConfig;
};

const iconByKind: Record<NodeKind, typeof Bot> = {
  brief: FileText,
  script: Wand2,
  image: ImageIcon,
  veo: Clapperboard,
  voice: Mic2,
  subtitles: Subtitles,
  review: Check,
  export: Download,
};

const colorClasses: Record<string, { icon: string; border: string; badge: string }> = {
  amber: { icon: "bg-amber-100 text-amber-700", border: "border-amber-200", badge: "bg-amber-50 text-amber-700" },
  blue: { icon: "bg-blue-100 text-blue-700", border: "border-blue-200", badge: "bg-blue-50 text-blue-700" },
  violet: { icon: "bg-violet-100 text-violet-700", border: "border-violet-200", badge: "bg-violet-50 text-violet-700" },
  cyan: { icon: "bg-cyan-100 text-cyan-700", border: "border-cyan-200", badge: "bg-cyan-50 text-cyan-700" },
  rose: { icon: "bg-rose-100 text-rose-700", border: "border-rose-200", badge: "bg-rose-50 text-rose-700" },
  emerald: { icon: "bg-emerald-100 text-emerald-700", border: "border-emerald-200", badge: "bg-emerald-50 text-emerald-700" },
  orange: { icon: "bg-orange-100 text-orange-700", border: "border-orange-200", badge: "bg-orange-50 text-orange-700" },
  slate: { icon: "bg-slate-100 text-slate-700", border: "border-slate-300", badge: "bg-slate-100 text-slate-700" },
};

export default function GenVideoPipeline() {
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<{ id: string; pointerId: number; startX: number; startY: number; nodeX: number; nodeY: number } | null>(null);
  const versionRef = useRef(1);
  const hydratedRef = useRef(false);
  const [nodes, setNodes] = useState<PipelineNode[]>([]);
  const [catalog, setCatalog] = useState<CatalogEntry[]>([]);
  const [provider, setProvider] = useState<ProviderConfig | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [syncStatus, setSyncStatus] = useState<SyncStatus>("loading");
  const [syncMessage, setSyncMessage] = useState("Đang tải flow từ backend...");
  const [libraryFilter, setLibraryFilter] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [runProgress, setRunProgress] = useState(0);
  const [runMessage, setRunMessage] = useState("Flow chưa sẵn sàng");
  const [generatedVideoUrl, setGeneratedVideoUrl] = useState("");

  const selectedNode = nodes.find((node) => node.id === selectedId) || null;
  const visibleCatalog = catalog.filter((entry) => `${entry.title} ${entry.description}`.toLowerCase().includes(libraryFilter.toLowerCase()));
  const catalogByKind = new Map(catalog.map((entry) => [entry.kind, entry]));
  const edges = nodes.slice(0, -1).map((node, index) => ({ from: node, to: nodes[index + 1] }));

  useEffect(() => {
    let mounted = true;
    async function loadBootstrap() {
      try {
        const response = await fetch(`${FLOW_API_URL}/bootstrap?flow_id=${encodeURIComponent(FLOW_ID)}`);
        const payload = (await response.json()) as BootstrapPayload & { detail?: string };
        if (!response.ok) throw new Error(payload.detail || "Không tải được cấu hình GenVideo từ backend.");
        if (!mounted) return;
        setCatalog(payload.catalog);
        setProvider(payload.provider);
        setNodes(payload.graph.nodes);
        setSelectedId(payload.graph.nodes[0]?.id || "");
        versionRef.current = payload.graph.version;
        hydratedRef.current = true;
        setSyncStatus("saved");
        setSyncMessage("Đã tải và đồng bộ với backend");
        setRunMessage(payload.provider.configured ? "Flow sẵn sàng" : "Backend chưa có Gemini API key");
      } catch (error) {
        if (!mounted) return;
        setSyncStatus("error");
        setSyncMessage(error instanceof Error ? error.message : "Không kết nối được backend");
      }
    }
    loadBootstrap();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (!hydratedRef.current || nodes.length === 0) return;
    setSyncStatus("saving");
    setSyncMessage("Đang lưu thay đổi...");
    const timer = window.setTimeout(() => {
      saveGraph(nodes).catch(() => undefined);
    }, 450);
    return () => window.clearTimeout(timer);
  }, [nodes]);

  async function saveGraph(nextNodes: PipelineNode[]) {
    const response = await fetch(`${FLOW_API_URL}/flows/${encodeURIComponent(FLOW_ID)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ flow_id: FLOW_ID, version: versionRef.current, nodes: nextNodes }),
    });
    const payload = (await response.json()) as { version?: number; detail?: string };
    if (!response.ok) {
      setSyncStatus("error");
      setSyncMessage(payload.detail || "Backend không lưu được flow");
      throw new Error(payload.detail || "Backend không lưu được flow");
    }
    versionRef.current = payload.version || versionRef.current + 1;
    setSyncStatus("saved");
    setSyncMessage("Đã đồng bộ backend");
  }

  function startNodeDrag(event: PointerEvent<HTMLElement>, node: PipelineNode) {
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      id: node.id,
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      nodeX: node.x,
      nodeY: node.y,
    };
    setSelectedId(node.id);
  }

  function moveNode(event: PointerEvent<HTMLElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const canvas = canvasRef.current;
    const canvasWidth = canvas?.clientWidth || 1080;
    const canvasHeight = canvas?.clientHeight || 690;
    const x = Math.max(12, Math.min(canvasWidth - 236, drag.nodeX + event.clientX - drag.startX));
    const y = Math.max(18, Math.min(canvasHeight - 148, drag.nodeY + event.clientY - drag.startY));
    setNodes((current) => current.map((node) => (node.id === drag.id ? { ...node, x, y } : node)));
  }

  function stopNodeDrag(event: PointerEvent<HTMLElement>) {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    dragRef.current = null;
  }

  function createNode(entry: CatalogEntry, position?: { x: number; y: number }): PipelineNode {
    const index = nodes.length;
    return {
      id: `${entry.kind}-${crypto.randomUUID()}`,
      kind: entry.kind,
      title: entry.title,
      eyebrow: entry.kind === "veo" ? "GENERATE" : "STEP",
      description: entry.description,
      x: position?.x ?? 48 + (index % 3) * 270,
      y: position?.y ?? 72 + Math.floor(index / 3) * 190,
      status: "idle",
      settings: { ...(entry.default_settings || {}) },
    };
  }

  function addNode(kind: NodeKind, position?: { x: number; y: number }) {
    const entry = catalog.find((item) => item.kind === kind);
    if (!entry) return;
    const node = createNode(entry, position);
    setNodes((current) => [...current, node]);
    setSelectedId(node.id);
  }

  function onCanvasDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    const kind = event.dataTransfer.getData("application/genvideo-node") as NodeKind;
    if (!kind || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    addNode(kind, { x: Math.max(12, event.clientX - rect.left - 112), y: Math.max(18, event.clientY - rect.top - 40) });
  }

  function updateSelectedSetting(key: string, value: string) {
    setNodes((current) => current.map((node) => (node.id === selectedId ? { ...node, settings: { ...node.settings, [key]: value } } : node)));
  }

  function deleteSelected() {
    if (!selectedId || nodes.length <= 1) return;
    setNodes((current) => current.filter((node) => node.id !== selectedId));
    setSelectedId(nodes.find((node) => node.id !== selectedId)?.id || "");
  }

  async function runPipeline() {
    if (isRunning || nodes.length === 0) return;
    setIsRunning(true);
    setGeneratedVideoUrl("");
    setRunProgress(5);
    setRunMessage("Đang đồng bộ flow trước khi chạy...");
    try {
      await saveGraph(nodes);
      const response = await fetch(`${FLOW_API_URL}/flows/${encodeURIComponent(FLOW_ID)}/run`, { method: "POST" });
      const payload = (await response.json()) as { operation_id?: string; detail?: string };
      if (!response.ok || !payload.operation_id) throw new Error(payload.detail || "Backend không khởi tạo được pipeline.");
      setNodes((current) => current.map((node) => ({ ...node, status: node.kind === "veo" ? "running" : node.status })));
      setRunMessage("Backend đã nhận flow, Veo đang dựng video...");
      for (let attempt = 0; attempt < 90; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 10000));
        const pollResponse = await fetch(`${FLOW_API_URL}/${payload.operation_id}`);
        const pollPayload = (await pollResponse.json()) as { done?: boolean; progress?: number; video_url?: string; detail?: string };
        if (!pollResponse.ok) throw new Error(pollPayload.detail || "Không đọc được tiến độ Veo.");
        setRunProgress(Math.max(10, Math.min(98, pollPayload.progress || 50)));
        if (pollPayload.done && pollPayload.video_url) {
          setGeneratedVideoUrl(`${BACKEND_URL}${pollPayload.video_url}`);
          setRunProgress(100);
          setRunMessage("Backend đã hoàn tất video Veo 3.1");
          setNodes((current) => current.map((node) => ({ ...node, status: "done" })));
          return;
        }
      }
      throw new Error("Veo chưa hoàn tất trong thời gian chờ.");
    } catch (error) {
      setRunMessage(error instanceof Error ? error.message : "Pipeline thất bại");
      setNodes((current) => current.map((node) => ({ ...node, status: node.status === "running" ? "error" : node.status })));
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <section className="min-h-[calc(100vh-64px)] bg-[#f4f7fb] text-slate-950">
      <div className="mx-auto max-w-[1680px] px-5 py-6 lg:px-8">
        <header className="flex flex-wrap items-end justify-between gap-5">
          <div>
            <div className="flex items-center gap-3 text-xs font-bold uppercase tracking-[0.22em] text-blue-700">
              <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-600 text-white shadow-lg shadow-blue-200"><Workflow size={18} /></span>
              Gen Video
            </div>
            <h2 className="mt-4 text-3xl font-semibold tracking-tight sm:text-4xl">Biến một ý tưởng thành video hoàn chỉnh</h2>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">Graph và cấu hình được tải, lưu và chạy tại backend. Kéo node trực tiếp để sắp xếp flow.</p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 text-xs font-semibold ${provider?.configured ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-amber-200 bg-amber-50 text-amber-700"}`}>
              <span className={`h-2 w-2 rounded-full ${provider?.configured ? "bg-emerald-500" : "bg-amber-500"}`} />
              {provider?.configured ? "Google API đã cấu hình" : "Thiếu Gemini API key"}
            </span>
            <button type="button" onClick={runPipeline} disabled={isRunning || syncStatus === "loading"} className="inline-flex items-center gap-2 rounded-xl bg-slate-950 px-4 py-3 text-sm font-semibold text-white shadow-lg shadow-slate-300 hover:bg-blue-700 disabled:cursor-wait disabled:opacity-60">
              {isRunning ? <Loader2 className="animate-spin" size={16} /> : <Play size={16} fill="currentColor" />}
              {isRunning ? "Đang tạo..." : "Tạo video với Veo 3.1"}
            </button>
          </div>
        </header>

        <section className="mt-7 grid gap-5 xl:grid-cols-[260px_minmax(0,1fr)_300px]">
          <aside className="rounded-2xl border border-slate-200 bg-white p-4 shadow-soft">
            <div className="flex items-center justify-between"><div><p className="text-[11px] font-bold uppercase tracking-[0.18em] text-slate-400">Backend catalog</p><h3 className="mt-1 text-base font-semibold">Node pipeline</h3></div><Layers3 className="text-blue-600" size={19} /></div>
            <label className="mt-4 flex h-10 items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3"><Sparkles size={14} className="text-slate-400" /><input value={libraryFilter} onChange={(event) => setLibraryFilter(event.target.value)} className="min-w-0 flex-1 bg-transparent text-sm outline-none" placeholder="Tìm node..." /></label>
            <div className="mt-4 grid gap-2">
              {visibleCatalog.map((entry) => {
                const Icon = iconByKind[entry.kind];
                const colors = colorClasses[entry.color] || colorClasses.slate;
                return (
                  <button key={entry.kind} type="button" draggable onDragStart={(event) => event.dataTransfer.setData("application/genvideo-node", entry.kind)} onClick={() => addNode(entry.kind)} className={`group flex items-center gap-3 rounded-xl border bg-white p-3 text-left transition hover:-translate-y-0.5 hover:shadow-md ${colors.border}`}>
                    <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${colors.icon}`}><Icon size={17} /></span>
                    <span className="min-w-0"><span className="block truncate text-sm font-semibold text-slate-800">{entry.title}</span><span className="mt-0.5 block truncate text-[11px] text-slate-400">{entry.description}</span></span>
                    <Plus className="ml-auto shrink-0 text-slate-300 group-hover:text-blue-600" size={15} />
                  </button>
                );
              })}
            </div>
            <div className="mt-5 rounded-xl bg-slate-950 p-4 text-white"><p className="text-[10px] font-bold uppercase tracking-[0.18em] text-cyan-300">Đồng bộ</p><p className="mt-2 text-xs leading-5 text-slate-300">Vị trí, thứ tự và settings được tự động lưu về backend sau mỗi thay đổi.</p></div>
          </aside>

          <section className="min-w-0 overflow-auto rounded-2xl border border-slate-200 bg-white shadow-soft">
            <header className="sticky left-0 flex min-w-[920px] flex-wrap items-center justify-between gap-3 border-b border-slate-100 px-5 py-4"><div className="flex items-center gap-3"><span className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-50 text-blue-700"><Move size={16} /></span><div><h3 className="text-sm font-semibold">Canvas flow</h3><p className="text-xs text-slate-400">{nodes.length} node · {edges.length} kết nối</p></div></div><SyncBadge status={syncStatus} message={syncMessage} /></header>
            <div ref={canvasRef} data-testid="genvideo-canvas" onDragOver={(event) => event.preventDefault()} onDrop={onCanvasDrop} className="relative min-h-[690px] min-w-[920px] overflow-hidden bg-[radial-gradient(#dbe5f0_1px,transparent_1px)] [background-size:18px_18px]">
              <svg className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden="true"><defs><marker id="genvideo-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#94a3b8" /></marker></defs>{edges.map(({ from, to }) => <line key={`${from.id}-${to.id}`} x1={from.x + 112} y1={from.y + 68} x2={to.x + 8} y2={to.y + 68} stroke="#cbd5e1" strokeWidth="2" strokeDasharray="5 5" markerEnd="url(#genvideo-arrow)" />)}</svg>
              {syncStatus === "loading" && <div className="absolute inset-0 z-20 flex items-center justify-center bg-white/80"><Loader2 className="animate-spin text-blue-600" size={28} /><span className="ml-3 text-sm font-semibold text-slate-600">Đang tải flow từ backend...</span></div>}
              {nodes.map((node, index) => (
                <PipelineNodeCard key={node.id} node={node} index={index} selected={node.id === selectedId} entry={catalogByKind.get(node.kind)} onSelect={() => setSelectedId(node.id)} onPointerDown={(event) => startNodeDrag(event, node)} onPointerMove={moveNode} onPointerUp={stopNodeDrag} />
              ))}
              {syncStatus !== "loading" && nodes.length === 0 && <div className="absolute inset-0 flex items-center justify-center text-center"><div><UploadCloud className="mx-auto text-blue-500" size={34} /><p className="mt-3 font-semibold">Thả node đầu tiên vào đây</p></div></div>}
            </div>
            <footer className="sticky left-0 flex min-w-[920px] flex-wrap items-center justify-between gap-3 border-t border-slate-100 px-5 py-4"><div className="flex items-center gap-2 text-sm"><span className={`h-2.5 w-2.5 rounded-full ${isRunning ? "animate-pulse bg-blue-500" : runProgress === 100 ? "bg-emerald-500" : "bg-slate-300"}`} /><span className="font-semibold text-slate-700">{runMessage}</span><span className="text-slate-400">{runProgress}%</span></div><div className="h-2 min-w-40 flex-1 rounded-full bg-slate-100 sm:max-w-xs"><span className="block h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all" style={{ width: `${runProgress}%` }} /></div></footer>
          </section>

          <aside className="rounded-2xl border border-slate-200 bg-white p-5 shadow-soft">
            {selectedNode ? <NodeInspector node={selectedNode} provider={provider} entry={catalogByKind.get(selectedNode.kind)} onSettingChange={updateSelectedSetting} onDelete={deleteSelected} /> : <div className="flex h-full min-h-64 flex-col items-center justify-center text-center text-slate-400"><Settings2 size={28} /><p className="mt-3 text-sm font-semibold">Chọn một node để chỉnh</p></div>}
          </aside>

          {generatedVideoUrl && <section className="col-span-full rounded-2xl border border-emerald-200 bg-emerald-50 p-4 shadow-soft"><div className="flex flex-wrap items-center justify-between gap-3"><div><p className="text-[11px] font-bold uppercase tracking-[0.18em] text-emerald-700">Video output</p><h3 className="mt-1 text-base font-semibold">Backend đã tạo xong MP4</h3></div><a href={generatedVideoUrl} download className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-3 py-2 text-xs font-semibold text-white hover:bg-emerald-700"><Download size={14} /> Tải MP4</a></div><video className="mt-4 max-h-[520px] w-full rounded-xl bg-slate-950 object-contain" src={generatedVideoUrl} controls /></section>}
        </section>
      </div>
    </section>
  );
}

function PipelineNodeCard({ node, index, selected, entry, onSelect, onPointerDown, onPointerMove, onPointerUp }: { node: PipelineNode; index: number; selected: boolean; entry?: CatalogEntry; onSelect: () => void; onPointerDown: (event: PointerEvent<HTMLElement>) => void; onPointerMove: (event: PointerEvent<HTMLElement>) => void; onPointerUp: (event: PointerEvent<HTMLElement>) => void }) {
  const Icon = iconByKind[node.kind];
  const colors = colorClasses[entry?.color || "slate"] || colorClasses.slate;
  return (
    <article data-testid={`pipeline-node-${node.id}`} onClick={onSelect} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp} className={`absolute w-56 select-none rounded-xl border bg-white shadow-sm transition-shadow touch-none ${selected ? `z-10 cursor-grabbing ring-2 ring-blue-400 ${colors.border}` : "cursor-grab border-slate-200 hover:shadow-md"}`} style={{ left: node.x, top: node.y }}>
      <header className="flex items-center gap-2 border-b border-slate-100 px-3 py-2"><GripVertical className="text-slate-300" size={15} /><span className={`flex h-7 w-7 items-center justify-center rounded-md ${colors.icon}`}><Icon size={14} /></span><span className="min-w-0 flex-1 truncate text-xs font-bold text-slate-800">{node.title}</span><span className={`h-2 w-2 rounded-full ${node.status === "running" ? "animate-pulse bg-blue-500" : node.status === "done" ? "bg-emerald-500" : node.status === "error" ? "bg-rose-500" : "bg-slate-300"}`} /></header>
      <div className="px-3 py-3"><div className={`mb-2 inline-flex rounded px-1.5 py-0.5 text-[9px] font-bold tracking-[0.15em] ${colors.badge}`}>{node.eyebrow}</div><p className="min-h-9 text-[11px] leading-4 text-slate-500">{node.description}</p><div className="mt-3 flex items-center justify-between text-[10px] text-slate-400"><span>{index + 1} / pipeline</span><Link2 size={13} /></div></div>
    </article>
  );
}

function NodeInspector({ node, provider, entry, onSettingChange, onDelete }: { node: PipelineNode; provider: ProviderConfig | null; entry?: CatalogEntry; onSettingChange: (key: string, value: string) => void; onDelete: () => void }) {
  const Icon = iconByKind[node.kind];
  const colors = colorClasses[entry?.color || "slate"] || colorClasses.slate;
  return (
    <div>
      <header className="flex items-start justify-between gap-3"><div className="flex items-center gap-3"><span className={`flex h-10 w-10 items-center justify-center rounded-xl ${colors.icon}`}><Icon size={19} /></span><div><p className="text-[10px] font-bold uppercase tracking-[0.18em] text-slate-400">{node.eyebrow}</p><h3 className="mt-1 text-base font-semibold">{node.title}</h3></div></div><button type="button" onClick={onDelete} className="rounded-lg p-2 text-slate-400 hover:bg-rose-50 hover:text-rose-600" aria-label="Xóa node"><Trash2 size={15} /></button></header>
      <p className="mt-4 text-xs leading-5 text-slate-500">{node.description}</p>
      <div className="mt-5 grid gap-4">
        {Object.entries(node.settings).map(([key, value]) => (
          <SettingField key={key} nodeKind={node.kind} settingKey={key} value={value} provider={provider} onChange={(nextValue) => onSettingChange(key, nextValue)} />
        ))}
        {Object.keys(node.settings).length === 0 && <p className="rounded-lg bg-slate-50 p-3 text-xs text-slate-500">Node mới chưa có settings mặc định. Backend sẽ dùng schema của loại node này khi chạy.</p>}
      </div>
      <div className={`mt-6 rounded-xl border p-3 ${provider?.configured ? "border-cyan-100 bg-cyan-50" : "border-amber-200 bg-amber-50"}`}><div className={`flex items-center gap-2 text-xs font-semibold ${provider?.configured ? "text-cyan-800" : "text-amber-800"}`}><Bot size={15} /> {provider?.name || "Google provider"}</div><p className={`mt-1 text-[11px] leading-4 ${provider?.configured ? "text-cyan-700" : "text-amber-700"}`}>{provider?.configured ? `API key đang lấy từ ${provider.key_source}.` : `Cấu hình ${provider?.key_source || "GEMINI_API_KEY"} ở backend rồi reload.`}</p></div>
    </div>
  );
}

function SettingField({ nodeKind, settingKey, value, provider, onChange }: { nodeKind: NodeKind; settingKey: string; value: string; provider: ProviderConfig | null; onChange: (value: string) => void }) {
  const options = nodeKind === "veo" && settingKey === "model" ? provider?.models : nodeKind === "image" && settingKey === "model" ? provider?.image_models : settingKey === "aspect_ratio" ? provider?.aspect_ratios : settingKey === "resolution" ? provider?.resolutions : settingKey === "copyright_confirmed" ? ["false", "true"] : undefined;
  return (
    <label className="grid gap-1.5"><span className="text-[10px] font-bold uppercase tracking-[0.12em] text-slate-400">{settingKey.replace(/_/g, " ")}</span>
      {options?.length ? <select value={value} onChange={(event) => onChange(event.target.value)} className="h-10 rounded-lg border border-slate-200 bg-slate-50 px-3 text-sm outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100">{options.map((option) => <option key={option} value={option}>{option}</option>)}</select> : settingKey === "prompt" ? <textarea value={value} onChange={(event) => onChange(event.target.value)} className="min-h-28 resize-y rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs leading-5 outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100" /> : <input value={value} onChange={(event) => onChange(event.target.value)} className="h-10 rounded-lg border border-slate-200 bg-slate-50 px-3 text-sm outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-100" />}
    </label>
  );
}

function SyncBadge({ status, message }: { status: SyncStatus; message: string }) {
  const classes = status === "error" ? "bg-rose-50 text-rose-700" : status === "saved" ? "bg-emerald-50 text-emerald-700" : "bg-blue-50 text-blue-700";
  return <span title={message} className={`inline-flex max-w-sm items-center gap-2 truncate rounded-full px-3 py-2 text-xs font-semibold ${classes}`}>{status === "saving" || status === "loading" ? <Loader2 className="animate-spin" size={13} /> : <span className="h-2 w-2 rounded-full bg-current" />}{message}</span>;
}
