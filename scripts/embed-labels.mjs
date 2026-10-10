// Pre-compute the field guide's text embeddings at build time.
// The browser then only needs SigLIP's vision half (~95 MB) and identifying a
// photo is one image pass plus a dot product. Re-run after editing labels.js:
//   npm run embed
import { AutoTokenizer, SiglipTextModel } from '@huggingface/transformers';
import { writeFileSync } from 'node:fs';
import { LABELS, labelPrompts } from '../src/labels.js';

const MODEL = 'Xenova/siglip-base-patch16-224';
const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
const model = await SiglipTextModel.from_pretrained(MODEL, { dtype: 'fp32' });

const normalize = (v) => {
  let s = 0;
  for (const x of v) s += x * x;
  s = Math.sqrt(s) || 1;
  return Float32Array.from(v, (x) => x / s);
};

let dim = 0;
const rows = [];
for (const label of LABELS) {
  // SigLIP was trained on fixed-length (64 token) text, so pad to max_length.
  const inputs = tokenizer(labelPrompts(label), { padding: 'max_length', truncation: true });
  const { pooler_output } = await model(inputs);
  const [n, d] = pooler_output.dims;
  dim = d;
  const sum = new Float32Array(d);
  for (let r = 0; r < n; r++) {
    const v = normalize(pooler_output.data.slice(r * d, (r + 1) * d));
    for (let j = 0; j < d; j++) sum[j] += v[j];
  }
  rows.push(normalize(sum));
}

const out = new Float32Array(rows.length * dim);
rows.forEach((r, i) => out.set(r, i * dim));
writeFileSync(new URL('../public/label-embeds.bin', import.meta.url), Buffer.from(out.buffer));
writeFileSync(new URL('../public/label-embeds.json', import.meta.url), JSON.stringify({
  model: MODEL, dim, count: rows.length, labels: LABELS.map(([n, g]) => `${g}:${n}`),
}));
console.log(`Embedded ${rows.length} labels (${dim}-d) with ${MODEL}`);
