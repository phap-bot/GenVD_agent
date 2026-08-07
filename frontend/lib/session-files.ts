const DATABASE_NAME = "video-clone-session-files";
const DATABASE_VERSION = 1;
const STORE_NAME = "files";

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);

    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Không mở được kho file của session."));
  });
}

function runRequest<T>(mode: IDBTransactionMode, action: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDatabase().then(
    (database) =>
      new Promise<T>((resolve, reject) => {
        const transaction = database.transaction(STORE_NAME, mode);
        const request = action(transaction.objectStore(STORE_NAME));

        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error ?? new Error("Không truy cập được file của session."));
        transaction.oncomplete = () => database.close();
        transaction.onerror = () => {
          database.close();
          reject(transaction.error ?? new Error("Không lưu được file của session."));
        };
      }),
  );
}

export async function readSessionFile(key: string): Promise<File | null> {
  const value = await runRequest<File | undefined>("readonly", (store) => store.get(key));
  return value instanceof File ? value : null;
}

export async function writeSessionFile(key: string, file: File | null): Promise<void> {
  if (file) {
    await runRequest<IDBValidKey>("readwrite", (store) => store.put(file, key));
    return;
  }
  await runRequest<undefined>("readwrite", (store) => store.delete(key));
}

export async function deleteSessionFiles(keys: string[]): Promise<void> {
  await Promise.all(keys.map((key) => writeSessionFile(key, null)));
}
