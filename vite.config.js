import { defineConfig } from 'vite';

// Cross-origin isolation unlocks SharedArrayBuffer, which ONNX Runtime needs to
// run WASM on multiple CPU threads. "credentialless" keeps CORS fetches of the
// model files from Hugging Face working.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
};

export default defineConfig({
  base: './',
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['@huggingface/transformers'] },
  build: { target: 'es2022' },
  server: { headers: isolation },
  preview: { headers: isolation },
});
