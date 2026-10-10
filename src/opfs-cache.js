// A Cache-API-shaped store backed by the Origin Private File System.
// The browser Cache API can fail on very large single entries (Gemma's weights
// file is ~800 MB). OPFS writes straight to disk and reads back as a
// disk-backed File, so big models survive restarts and work offline.

const DIR = 'models';

function keyToName(key) {
  const url = typeof key === 'string' ? key : key.url;
  // Readable tail + short hash so names stay unique and under length limits.
  let h = 2166136261;
  for (let i = 0; i < url.length; i++) { h ^= url.charCodeAt(i); h = Math.imul(h, 16777619); }
  const tail = url.split('/').slice(-3).join('_').replace(/[^\w.-]/g, '_').slice(-80);
  return `${(h >>> 0).toString(36)}_${tail}`;
}

let dirPromise;
const dir = () => (dirPromise ??= navigator.storage.getDirectory().then((root) => root.getDirectoryHandle(DIR, { create: true })));

export const failures = [];

export const opfsCache = {
  async match(key) {
    try {
      const file = await withTimeout((async () => (await (await dir()).getFileHandle(keyToName(key))).getFile())(), 5000);
      if (file.size === 0) return undefined;
      return new Response(file, { headers: { 'content-length': String(file.size) } });
    } catch {
      return undefined;
    }
  },

  async put(key, response) {
    const name = keyToName(key);
    let writable;
    try {
      const handle = await withTimeout((await dir()).getFileHandle(name, { create: true }), 5000);
      writable = await withTimeout(handle.createWritable(), 5000);
      // createWritable writes to a swap file and only replaces the real file on
      // close(), so an interrupted download never leaves a half-written model.
      // The data is already in memory here, so even ~1 GB finishes well inside
      // the timeout unless storage is stuck.
      await withTimeout(response.body ? response.body.pipeTo(writable) : (async () => {
        await writable.write(await response.arrayBuffer());
        await writable.close();
      })(), 180000);
    } catch (err) {
      failures.push({ key: typeof key === 'string' ? key : key.url, error: String(err?.message || err) });
      try { await writable?.abort(); } catch { /* already closed */ }
      try { await (await dir()).removeEntry(name); } catch { /* nothing to clean */ }
    }
  },
};

// Some browsers hang (rather than throw) on OPFS writes when storage is
// exhausted, so every probe is raced against a timeout.
const withTimeout = (p, ms) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), ms))]);

export async function opfsAvailable() {
  return withTimeout(probe(), 3000).catch(() => false);
}

async function probe() {
  try {
    const d = await dir();
    const h = await d.getFileHandle('.probe', { create: true });
    const w = await h.createWritable();
    await w.write('ok');
    await w.close();
    await d.removeEntry('.probe');
    return true;
  } catch {
    return false;
  }
}

export async function clearModels() {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(DIR, { recursive: true }).catch(() => {});
  dirPromise = null;
}
