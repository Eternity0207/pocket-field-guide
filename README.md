# Pocket Field Guide 🌿

An AI field guide that runs **entirely in your browser** and is built to get you off the screen.

- **Go outside:** pick 15, 30 or 60 minutes and where you are. Gemma writes a short list of things to look for, then the app switches to a quiet walking screen with a timer and a checklist.
- **Identify:** take or drop in a photo of a plant, bird, bug, mushroom or leaf. SigLIP matches it against a built-in field guide of about 150 species, and Gemma writes a short field note: what to check in person to confirm it, plus something related to find nearby.
- **Journal:** finds are saved on your device with an optional location, and you can export them as Markdown.

No account, no API key, no server. After the first download it works with **no internet at all**.

## Open-source AI inside

| Part | What | License |
|---|---|---|
| Language model | [Gemma 3 1B instruct](https://huggingface.co/onnx-community/gemma-3-1b-it-ONNX), 4-bit, on WebGPU ([270M](https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX) on CPU) | Gemma Terms of Use |
| Vision model | [SigLIP base patch16-224](https://huggingface.co/Xenova/siglip-base-patch16-224), 8-bit vision tower | Apache-2.0 |
| Runtime | [Transformers.js](https://github.com/huggingface/transformers.js) + ONNX Runtime Web | Apache-2.0 / MIT |

### How it fits together

```
build time:  labels.js ──► SigLIP text encoder ──► public/label-embeds.bin   (161 × 768 floats)

in browser:  photo ──► SigLIP vision encoder ──► dot product with label embeddings ──► top 5
                                                                                      │
                     trusted ID traits for that species (labels.js) ◄─────────────────┘
                                       │
                                       ▼
                 Gemma 3 (few-shot, prefilled "**Check it:**") ──► checked ──► field note
```

- **Zero-shot, no training.** Each species is described by three phrasings ("a photo of a hoopoe, a type of bird.", ...). Their text embeddings are averaged and computed **once at build time**, so the browser only downloads SigLIP's vision half (about 95 MB). Identifying a photo is one image pass plus a dot product.
- **Grounded generation.** A 1B model will happily make up field marks. Gemma never recalls facts on its own: it rewrites the trusted traits from `src/labels.js` into checks you can do in person. A worked example plus a prefilled answer keeps the format fixed. Each note is checked (sections present, uses the traits, didn't echo the prompt), and a note that fails falls back to a template built from the traits. Safety tips come from a fixed list, not the model.
- **Offline storage.** Model weights go into the Origin Private File System, which handles files around 1 GB better than the Cache API. A service worker caches the app shell and the ONNX Runtime WASM, which is served from the app's own origin, not a CDN.
- **Stable fallbacks.** Gemma runs on WebGPU when it's available. If the GPU fails, the worker restarts on the CPU with Gemma 270M (fp16, since the 4-bit builds use an operator the WASM backend lacks). If storage is full, the app still works online and says so. If Gemma isn't available at all, walks use a built-in list.

### Measured on a laptop (Intel Iris Xe, Chromium)

| | |
|---|---|
| Identify a photo (SigLIP, WASM, 12 threads) | ~1–2 s |
| Gemma 3 1B on WebGPU | 7–12 tokens/s, so a field note takes ~10 s |
| Gemma 3 270M on CPU | ~7 tokens/s |
| Top-1 accuracy on 16 Wikipedia lead photos of species in the guide | 15 / 16 (CLIP ViT-B/32 got 4 / 8 on the same first 8) |
| First download | ~100 MB vision + ~785 MB Gemma 1B |

## Run it

```bash
npm install
npm run dev
```

Open the printed URL, go to **Offline setup** and click **Download models**. Chrome or Edge on desktop gives you WebGPU.

```bash
npm run build
```

`dist/` is a static site. `render.yaml` deploys it to Render as a free static site, with the cross-origin isolation headers that let ONNX Runtime use every CPU thread.

## Make it yours

Add the species in your area to `src/labels.js`: a name, a group and a line of ID traits. Then run:

```bash
npm run embed
```

SigLIP recognises the new species right away and Gemma uses your traits in its notes. You don't need to retrain or fine-tune anything.

## Privacy

Photos, notes and locations never leave the browser. The only network requests are downloading the app and the model files from Hugging Face the first time.

## License

MIT for this code. Model weights are under their own licenses (see above).
