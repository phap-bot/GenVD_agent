"use client";

export type DownloadFolderKind = "video" | "subtitle";

export type DownloadDirectoryHandle = FileSystemDirectoryHandle & {
  queryPermission: (options?: { mode?: "read" | "readwrite" }) => Promise<PermissionState>;
  requestPermission: (options?: { mode?: "read" | "readwrite" }) => Promise<PermissionState>;
};

type DirectoryPickerWindow = Window & {
  showDirectoryPicker?: (options?: { mode?: "read" | "readwrite" }) => Promise<DownloadDirectoryHandle>;
};

const DATABASE_NAME = "video-clone-download-destinations";
const DATABASE_VERSION = 1;
const STORE_NAME = "folders";

function isSupported(): boolean {
  return typeof window !== "undefined" && typeof (window as DirectoryPickerWindow).showDirectoryPicker === "function";
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) request.result.createObjectStore(STORE_NAME);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Không mở được kho thư mục tải xuống."));
  });
}

async function runRequest<T>(mode: IDBTransactionMode, action: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const database = await openDatabase();
  return new Promise<T>((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, mode);
    const request = action(transaction.objectStore(STORE_NAME));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Không truy cập được thư mục tải xuống."));
    transaction.oncomplete = () => database.close();
    transaction.onerror = () => reject(transaction.error ?? new Error("Không lưu được cấu hình thư mục."));
  });
}

async function requestPermission(handle: DownloadDirectoryHandle): Promise<boolean> {
  try {
    const query = await handle.queryPermission({ mode: "readwrite" });
    if (query === "granted") return true;
    return (await handle.requestPermission({ mode: "readwrite" })) === "granted";
  } catch {
    return false;
  }
}

export function directoryPickerSupported(): boolean {
  return isSupported();
}

export async function pickDownloadFolder(): Promise<DownloadDirectoryHandle | null> {
  if (!isSupported()) return null;
  const picker = (window as DirectoryPickerWindow).showDirectoryPicker;
  if (!picker) return null;
  const handle = await picker({ mode: "readwrite" });
  if (!(await requestPermission(handle))) throw new Error("Chưa được cấp quyền ghi vào thư mục đã chọn.");
  return handle;
}

export async function saveDownloadFolder(kind: DownloadFolderKind, handle: DownloadDirectoryHandle): Promise<void> {
  await runRequest<IDBValidKey>("readwrite", (store) => store.put(handle, kind));
}

export async function readDownloadFolder(kind: DownloadFolderKind): Promise<DownloadDirectoryHandle | null> {
  try {
    const handle = await runRequest<DownloadDirectoryHandle | undefined>("readonly", (store) => store.get(kind));
    return handle || null;
  } catch {
    return null;
  }
}

export async function clearDownloadFolder(kind: DownloadFolderKind): Promise<void> {
  await runRequest<undefined>("readwrite", (store) => store.delete(kind));
}

export async function saveBlobToDownloadFolder(
  kind: DownloadFolderKind,
  filename: string,
  blob: Blob,
): Promise<boolean> {
  const handle = await readDownloadFolder(kind);
  if (!handle || !(await requestPermission(handle))) return false;
  try {
    const fileHandle = await handle.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(blob);
    await writable.close();
    return true;
  } catch {
    return false;
  }
}

export function triggerBrowserDownload(blob: Blob, filename: string): void {
  const blobUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = blobUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(blobUrl), 1000);
}

export async function downloadUrlToDestination(
  url: string,
  kind: DownloadFolderKind,
  fallbackName: string,
): Promise<{ filename: string; savedInFolder: boolean }> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Không tải được file ${kind === "video" ? "video" : "SRT"}.`);
  const blob = await response.blob();
  const pathname = new URL(url, window.location.origin).pathname;
  const filename = decodeURIComponent(pathname.split("/").pop() || fallbackName);
  const savedInFolder = await saveBlobToDownloadFolder(kind, filename, blob);
  if (!savedInFolder) triggerBrowserDownload(blob, filename);
  return { filename, savedInFolder };
}
