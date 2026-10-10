// Copy the ONNX Runtime WASM build into public/ so it's served from our own
// origin (works offline, no CDN dependency).
import { cpSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const dist = join(dirname(require.resolve('@huggingface/transformers')), '..', '..', '..', 'onnxruntime-web', 'dist');
const out = new URL('../public/ort/', import.meta.url);
mkdirSync(out, { recursive: true });
for (const f of ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm']) {
  cpSync(join(dist, f), new URL(f, out));
}
console.log('Copied ONNX Runtime WASM to public/ort');
