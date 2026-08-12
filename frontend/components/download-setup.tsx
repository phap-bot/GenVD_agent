"use client";

import { CheckCircle2, FolderOpen, Info, RotateCcw, Save, ShieldAlert } from "lucide-react";
import { useEffect, useState } from "react";
import {
  clearDownloadFolder,
  directoryPickerSupported,
  pickDownloadFolder,
  readDownloadFolder,
  saveDownloadFolder,
  type DownloadDirectoryHandle,
  type DownloadFolderKind,
} from "@/lib/download-destination";

type DownloadSetupProps = {
  onBack: () => void;
};

const folderLabels: Record<DownloadFolderKind, { title: string; description: string }> = {
  video: { title: "Thư mục video", description: "MP4 sau khi render hoặc tạo video." },
  subtitle: { title: "Thư mục phụ đề", description: "SRT dùng cho CapCut và hậu kỳ." },
};

export default function DownloadSetup({ onBack }: DownloadSetupProps) {
  const [folders, setFolders] = useState<Record<DownloadFolderKind, DownloadDirectoryHandle | null>>({ video: null, subtitle: null });
  const [busy, setBusy] = useState<DownloadFolderKind | null>(null);
  const [message, setMessage] = useState("");
  const supported = directoryPickerSupported();

  useEffect(() => {
    let cancelled = false;
    void Promise.all([readDownloadFolder("video"), readDownloadFolder("subtitle")]).then(([video, subtitle]) => {
      if (!cancelled) setFolders({ video, subtitle });
    });
    return () => {
      cancelled = true;
    };
  }, []);

  async function chooseFolder(kind: DownloadFolderKind) {
    setBusy(kind);
    setMessage("");
    try {
      const handle = await pickDownloadFolder();
      if (!handle) {
        setMessage("Trình duyệt hiện tại chưa hỗ trợ chọn thư mục trực tiếp.");
        return;
      }
      await saveDownloadFolder(kind, handle);
      setFolders((current) => ({ ...current, [kind]: handle }));
      setMessage(`Đã lưu ${folderLabels[kind].title.toLowerCase()}: ${handle.name}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không lưu được thư mục.");
    } finally {
      setBusy(null);
    }
  }

  async function removeFolder(kind: DownloadFolderKind) {
    try {
      await clearDownloadFolder(kind);
      setFolders((current) => ({ ...current, [kind]: null }));
      setMessage(`Đã xoá cấu hình ${folderLabels[kind].title.toLowerCase()}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Không xoá được cấu hình thư mục.");
    }
  }

  return (
    <main className="min-h-[calc(100vh-64px)] bg-slate-50 p-5 sm:p-8">
      <section className="mx-auto max-w-4xl rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
        <header className="flex flex-col gap-3 border-b border-slate-100 pb-6 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-blue-700">Thiết lập hệ thống</p>
            <h2 className="mt-2 text-2xl font-semibold text-slate-950">Thư mục lưu file tải xuống</h2>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
              Chọn thư mục một lần. Các lần tải MP4 và SRT sau sẽ ghi trực tiếp vào đây, không mở trang media riêng.
            </p>
          </div>
          <button type="button" onClick={onBack} className="inline-flex h-10 items-center justify-center rounded-md border border-slate-200 px-4 text-sm font-semibold text-slate-700 hover:bg-slate-50">
            Quay lại Studio
          </button>
        </header>

        {!supported && (
          <div className="mt-6 flex gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
            <ShieldAlert className="mt-0.5 shrink-0" size={18} />
            <p>Trình duyệt này chưa hỗ trợ File System Access API. Nút tải vẫn hoạt động theo kiểu tải xuống mặc định của trình duyệt.</p>
          </div>
        )}

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {(Object.keys(folderLabels) as DownloadFolderKind[]).map((kind) => {
            const handle = folders[kind];
            return (
              <article key={kind} className="rounded-xl border border-slate-200 p-5">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h3 className="font-semibold text-slate-950">{folderLabels[kind].title}</h3>
                    <p className="mt-1 text-sm text-slate-500">{folderLabels[kind].description}</p>
                  </div>
                  <FolderOpen className="text-blue-600" size={20} />
                </div>
                <div className="mt-5 rounded-lg bg-slate-50 px-3 py-3 text-sm">
                  {handle ? (
                    <span className="flex items-center gap-2 font-semibold text-emerald-700"><CheckCircle2 size={16} /> {handle.name}</span>
                  ) : (
                    <span className="text-slate-500">Chưa chọn thư mục</span>
                  )}
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  <button type="button" disabled={!supported || busy !== null} onClick={() => void chooseFolder(kind)} className="inline-flex h-9 items-center gap-2 rounded-md bg-blue-600 px-3 text-sm font-semibold text-white disabled:opacity-50">
                    <FolderOpen size={15} /> {busy === kind ? "Đang lưu..." : handle ? "Đổi thư mục" : "Chọn thư mục"}
                  </button>
                  {handle && <button type="button" disabled={busy !== null} onClick={() => void removeFolder(kind)} className="inline-flex h-9 items-center gap-2 rounded-md border border-slate-200 px-3 text-sm font-semibold text-slate-700 disabled:opacity-50"><RotateCcw size={15} /> Xoá</button>}
                </div>
              </article>
            );
          })}
        </div>

        <div className="mt-6 flex gap-3 rounded-lg border border-blue-100 bg-blue-50 p-4 text-sm text-blue-900">
          <Info className="mt-0.5 shrink-0" size={18} />
          <p>Quyền thư mục được lưu trên trình duyệt của máy này. Ứng dụng chỉ ghi file khi bạn nhấn tải và không gửi đường dẫn riêng tư lên backend.</p>
        </div>
        {message && <p className="mt-4 flex items-center gap-2 text-sm font-semibold text-slate-700" role="status"><Save size={15} /> {message}</p>}
      </section>
    </main>
  );
}
