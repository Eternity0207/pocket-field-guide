// Tiny IndexedDB wrapper for the field journal. Everything stays on this device.

const DB_NAME = 'pocket-field-guide';
const STORE = 'finds';

let dbPromise;
function open() {
  dbPromise ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const s = req.result.createObjectStore(STORE, { keyPath: 'id' });
      s.createIndex('createdAt', 'createdAt');
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

async function tx(mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const result = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(result?.result ?? result);
    t.onerror = () => reject(t.error);
  });
}

export const saveFind = (find) => tx('readwrite', (s) => s.put(find));
export const deleteFind = (id) => tx('readwrite', (s) => s.delete(id));

export async function listFinds() {
  const all = await tx('readonly', (s) => s.getAll());
  return all.sort((a, b) => b.createdAt - a.createdAt);
}
