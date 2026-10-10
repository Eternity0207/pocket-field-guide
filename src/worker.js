// All model work happens here so the UI never freezes.
// Two open-weight models, both running locally in the browser:
//   - SigLIP base (Google, Apache-2.0)   -> zero-shot "what is this?" from a photo
//   - Gemma 3 (Google, open weights)     -> field notes, ID checks, walk quests
// After the first download, both are served from the browser cache, so the
// app keeps working with no network at all.

import {
  env,
  AutoTokenizer,
  AutoProcessor,
  AutoModelForCausalLM,
  SiglipVisionModel,
  RawImage,
  TextStreamer,
  InterruptableStoppingCriteria,
} from '@huggingface/transformers';
import { LABELS } from './labels.js';
import { opfsCache, opfsAvailable, failures, clearModels } from './opfs-cache.js';

// Serve the ONNX Runtime WASM from our own origin instead of a CDN, so the app
// works offline and doesn't depend on a third party staying up.
const base = new URL(import.meta.env.BASE_URL + 'ort/', self.location.origin).href;
env.backends.onnx.wasm.wasmPaths = {
  mjs: base + 'ort-wasm-simd-threaded.asyncify.mjs',
  wasm: base + 'ort-wasm-simd-threaded.asyncify.wasm',
};
env.allowLocalModels = false;

// Store model files in OPFS when we can (handles ~1 GB files reliably);
// otherwise fall back to transformers.js' default Cache API storage.
// Decided lazily, before the first model load, so a slow storage check can
// never block the worker from answering messages.
let storageReady = null;
const setupStorage = () => (storageReady ??= opfsAvailable().then((ok) => {
  if (ok) {
    env.useBrowserCache = false;
    env.useCustomCache = true;
    env.customCache = opfsCache;
  }
}));

const VISION_ID = 'Xenova/siglip-base-patch16-224';
const GEMMA = {
  '1b': 'onnx-community/gemma-3-1b-it-ONNX',
  '270m': 'onnx-community/gemma-3-270m-it-ONNX',
};

const post = (msg) => self.postMessage(msg);

// Files are discovered one at a time, so the running total would jump around.
// Use the known download size as a floor so the bar moves smoothly.
const EXPECTED_MB = {
  vision: 100,
  '1b:q4': 880, '1b:q4f16': 785,
  '270m:q4': 345, '270m:q4f16': 295, '270m:fp16': 590,
};

function progressReporter(which, expectedMB = 0) {
  const files = new Map();
  return (p) => {
    if (p.status === 'progress' || p.status === 'done') {
      const prev = files.get(p.file) || { loaded: 0, total: 0 };
      files.set(p.file, {
        loaded: p.status === 'done' ? prev.total || p.loaded || 0 : p.loaded,
        total: p.total ?? prev.total,
      });
      let loaded = 0, total = 0;
      for (const f of files.values()) { loaded += f.loaded || 0; total += f.total || 0; }
      post({ type: 'progress', which, loaded, total: Math.max(total, expectedMB * 1e6) });
    }
  };
}

async function webgpuInfo() {
  try {
    if (!self.navigator.gpu) return null;
    const adapter = await self.navigator.gpu.requestAdapter();
    if (!adapter) return null;
    return { fp16: adapter.features.has('shader-f16') };
  } catch {
    return null;
  }
}

// ---------- Vision (SigLIP) ----------

let clip = null;

async function loadVision() {
  if (clip) return clip;
  await setupStorage();
  const progress_callback = progressReporter('vision', EXPECTED_MB.vision);
  const [processor, visionModel, labelEmbeds] = await Promise.all([
    AutoProcessor.from_pretrained(VISION_ID, { progress_callback }),
    SiglipVisionModel.from_pretrained(VISION_ID, { dtype: 'q8', device: 'wasm', progress_callback }),
    loadLabelEmbeds(),
  ]);
  clip = { processor, visionModel, labelEmbeds };
  return clip;
}

// The field guide's text embeddings are computed once at build time
// (scripts/embed-labels.mjs), so the browser never needs the text encoder.
async function loadLabelEmbeds() {
  const root = new URL(import.meta.env.BASE_URL, self.location.origin).href;
  const [meta, buf] = await Promise.all([
    fetch(root + 'label-embeds.json').then((r) => r.json()),
    fetch(root + 'label-embeds.bin').then((r) => r.arrayBuffer()),
  ]);
  const expected = LABELS.map(([n, g]) => `${g}:${n}`);
  if (meta.model !== VISION_ID || meta.labels.join('|') !== expected.join('|')) {
    throw new Error('Field guide embeddings are out of date. Run `npm run embed`.');
  }
  const all = new Float32Array(buf);
  return LABELS.map((_, i) => all.subarray(i * meta.dim, (i + 1) * meta.dim));
}

function normalize(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  s = Math.sqrt(s) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / s;
  return out;
}

async function classify(blob) {
  const { processor, visionModel, labelEmbeds } = await loadVision();
  const image = await RawImage.fromBlob(blob);
  const inputs = await processor(image);
  const { pooler_output } = await visionModel(inputs);
  const img = normalize(pooler_output.data);

  // SigLIP's learned scale (~110) makes softmax over 160 labels overconfident;
  // a softer scale keeps the confidence bar honest without changing the ranking.
  const logits = labelEmbeds.map((t) => {
    let dot = 0;
    for (let i = 0; i < t.length; i++) dot += t[i] * img[i];
    return dot * 60;
  });
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sum);

  const groupScore = {};
  probs.forEach((p, i) => { const g = LABELS[i][1]; groupScore[g] = (groupScore[g] || 0) + p; });

  const top = probs
    .map((p, i) => ({ name: LABELS[i][0], group: LABELS[i][1], traits: LABELS[i][2], p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 5);
  return { top, groupScore };
}

// ---------- Language (Gemma) ----------

let gemma = null;
const stopper = new InterruptableStoppingCriteria();
let genChain = Promise.resolve();

// GPU: the chosen size in 4-bit. CPU (WASM): ONNX Runtime Web has no CPU
// kernel for the 4/8-bit embedding op these exports use, so we run the 270M
// model in fp16 there. A failed session load leaves transformers.js unable to
// create further sessions in this worker, so we never "try and see" here;
// the UI restarts the worker to fall back instead.
function gemmaPlan(size, device, gpu) {
  if (device === 'webgpu') return { id: GEMMA[size] || GEMMA['1b'], size, dtype: gpu?.fp16 ? 'q4f16' : 'q4' };
  return { id: GEMMA['270m'], size: '270m', dtype: 'fp16' };
}

async function loadGemma(size = '1b', device = 'webgpu') {
  if (gemma && gemma.requested === `${size}:${device}`) return gemma;
  await setupStorage();
  if (gemma) { await gemma.model.dispose?.(); gemma = null; }
  const gpu = device === 'webgpu' ? await webgpuInfo() : null;
  if (device === 'webgpu' && !gpu) throw new Error('WebGPU is not available');
  const plan = gemmaPlan(size, device, gpu);

  const tokenizer = await AutoTokenizer.from_pretrained(plan.id);
  const progress_callback = progressReporter('llm', EXPECTED_MB[`${plan.size}:${plan.dtype}`]);
  post({ type: 'status', which: 'llm', text: `Loading Gemma 3 ${plan.size.toUpperCase()} on ${device === 'webgpu' ? 'GPU' : 'CPU'}…` });
  const model = await AutoModelForCausalLM.from_pretrained(plan.id, { device, dtype: plan.dtype, progress_callback });
  // Warm up once so the first real answer starts quickly, and so GPU driver
  // problems surface now rather than mid-sentence.
  post({ type: 'status', which: 'llm', text: 'Warming up…' });
  const tps = await timeTokens(tokenizer, model, 8);
  gemma = { requested: `${size}:${device}`, tokenizer, model, device, dtype: plan.dtype, size: plan.size, tps };
  return gemma;
}

async function timeTokens(tokenizer, model, n) {
  const inputs = tokenizer.apply_chat_template([{ role: 'user', content: 'Name a tree.' }], {
    add_generation_prompt: true, return_dict: true,
  });
  let first = null;
  let count = 0;
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    callback_function: () => {},
    token_callback_function: () => { first ??= performance.now(); count++; },
  });
  await model.generate({ ...inputs, max_new_tokens: n, do_sample: false, streamer });
  const secs = first ? (performance.now() - first) / 1000 : 0;
  return count > 1 && secs > 0 ? (count - 1) / secs : 0;
}

async function generate(id, messages, size, maxTokens = 320, prefill = '', device = 'webgpu') {
  const { tokenizer, model } = await loadGemma(size, device);
  // Prefilling the start of the answer pins the format and stops the small
  // model from opening with chatter ("Okay, here's...").
  const prompt = tokenizer.apply_chat_template(messages, { add_generation_prompt: true, tokenize: false }) + prefill;
  const inputs = tokenizer(prompt, { add_special_tokens: false });
  let text = '';
  let tokens = 0;
  let start = null;
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (chunk) => {
      text += chunk;
      post({ type: 'token', id, chunk });
    },
    token_callback_function: () => {
      start ??= performance.now();
      tokens++;
    },
  });
  stopper.reset();
  await model.generate({
    ...inputs,
    max_new_tokens: maxTokens,
    do_sample: true,
    temperature: 0.4,
    top_p: 0.9,
    repetition_penalty: 1.1,
    streamer,
    stopping_criteria: stopper,
  });
  const secs = start ? (performance.now() - start) / 1000 : 0;
  return { text: text.trim(), tps: secs > 0 ? tokens / secs : 0 };
}

// ---------- Message loop ----------

self.addEventListener('message', async ({ data }) => {
  const { type, id } = data;
  try {
    switch (type) {
      case 'load-vision':
        await loadVision();
        post({ type: 'ready', which: 'vision', offline: failures.length === 0 });
        break;
      case 'load-llm': {
        const g = await loadGemma(data.size, data.device);
        post({ type: 'ready', which: 'llm', device: g.device, dtype: g.dtype, size: g.size, tps: g.tps, offline: failures.length === 0 });
        break;
      }
      case 'classify':
        post({ type: 'result', id, result: await classify(data.blob) });
        break;
      case 'generate': {
        // One generation at a time: a new request waits for the previous one
        // (which the UI has usually just interrupted) before resetting the stopper.
        const run = genChain.then(() => generate(id, data.messages, data.size, data.maxTokens, data.prefill, data.device));
        genChain = run.catch(() => {});
        post({ type: 'result', id, result: await run });
        break;
      }
      case 'clear-models':
        await clearModels();
        await caches.delete('transformers-cache').catch(() => {});
        post({ type: 'result', id, result: true });
        break;
      case 'stop':
        stopper.interrupt();
        break;
    }
  } catch (err) {
    console.error(err);
    post({ type: 'error', id, which: data.which, message: String(err?.message || err) });
  }
});
