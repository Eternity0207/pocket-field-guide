import './style.css';
import { GROUPS, LABELS } from './labels.js';
import { saveFind, listFinds, deleteFind } from './db.js';
import { renderMarkdown, escapeHtml } from './markdown.js';

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};

// ---------- Worker client ----------

let worker;
let nextId = 1;
const pending = new Map();
const state = { vision: 'idle', llm: 'idle', llmDevice: null, gpu: null };
const loaders = {};
let gpuReady = Promise.resolve();

function startWorker() {
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.addEventListener('message', ({ data }) => {
    switch (data.type) {
      case 'progress': onProgress(data); break;
      case 'status': setModelStatus(data.which, data.text); break;
      case 'ready': onReady(data); break;
      case 'token': pending.get(data.id)?.onToken?.(data.chunk); break;
      case 'result': pending.get(data.id)?.resolve(data.result); pending.delete(data.id); break;
      case 'error':
        if (data.id && pending.has(data.id)) { pending.get(data.id).reject(new Error(data.message)); pending.delete(data.id); }
        else if (data.which) onLoadError(data.which, data.message);
        break;
    }
  });
  worker.addEventListener('error', (e) => {
    console.error('Worker error', e);
    for (const which of ['vision', 'llm']) if (state[which] === 'loading') onLoadError(which, e.message || 'the AI worker crashed');
  });
}

// A model that fails to load can leave the runtime unable to create new
// sessions, so recovery means a fresh worker. Models reload from local storage.
function restartWorker() {
  worker?.terminate();
  for (const [, p] of pending) p.reject(new Error('restarting'));
  pending.clear();
  for (const which of ['vision', 'llm']) {
    if (state[which] === 'loading') loaders[which + 'Reject']?.(new Error('restarting'));
    state[which] = 'idle';
    delete loaders[which];
  }
  startWorker();
}

function call(type, payload = {}, onToken) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onToken });
    worker.postMessage({ type, id, ...payload });
  });
}

function ensure(which) {
  if (state[which] === 'ready') return Promise.resolve();
  loaders[which] ??= new Promise((resolve, reject) => {
    loaders[which + 'Resolve'] = resolve;
    loaders[which + 'Reject'] = reject;
    state[which] = 'loading';
    renderStatus();
    if (which === 'vision') worker.postMessage({ type: 'load-vision', which });
    else gpuReady.then(() => worker.postMessage({ type: 'load-llm', which, size: modelSize(), device: llmDevice() }));
  });
  return loaders[which];
}

function onReady({ which, device, size, tps, offline }) {
  state[which] = 'ready';
  state.offline = offline;
  if (which === 'llm') {
    state.llmDevice = `${device === 'webgpu' ? 'GPU' : 'CPU'}, Gemma 3 ${String(size).toUpperCase()}`;
  }
  setBar(which, 1);
  const where = offline ? 'saved for offline use' : "couldn't save for offline use (browser storage is full or blocked)";
  setModelStatus(which, which === 'llm' ? `Ready on ${state.llmDevice}; ${where}` : `Ready, ${where}`);
  const downloaded = store.get('downloaded', {});
  downloaded[which] = which === 'llm' ? modelSize() : true;
  store.set('downloaded', downloaded);
  loaders[which + 'Resolve']?.();
  renderStatus();
}

function onLoadError(which, message) {
  if (which === 'llm' && llmDevice() === 'webgpu' && devicePrefValue() === 'auto') {
    // GPU path failed: remember that, and come back on the CPU.
    console.warn('Gemma failed on GPU, falling back to CPU:', message);
    store.set('gpuFailed', true);
    const visionWasReady = state.vision === 'ready';
    const reject = loaders.llmReject;
    const resolve = loaders.llmResolve;
    state.llm = 'idle';
    restartWorker();
    toast('GPU unavailable for Gemma, switching to CPU.');
    if (visionWasReady) ensure('vision').catch(() => {});
    ensure('llm').then(resolve, reject);
    return;
  }
  state[which] = 'error';
  setModelStatus(which, `Couldn't load: ${message}`);
  loaders[which + 'Reject']?.(new Error(message));
  delete loaders[which];
  renderStatus();
  toast(navigator.onLine ? 'A model failed to load. Try again from Offline setup.' : 'You are offline and this model is not downloaded yet.');
}

function onProgress({ which, loaded, total }) {
  if (!total) return;
  setBar(which, loaded / total);
  setModelStatus(which, `Downloading… ${fmtMB(loaded)} of ${fmtMB(total)}`);
}

const fmtMB = (b) => `${Math.round(b / 1e6)} MB`;
const setBar = (which, f) => { $(which === 'vision' ? 'visionBar' : 'llmBar').style.width = `${Math.min(100, f * 100)}%`; };
function setModelStatus(which, text) {
  $(which === 'vision' ? 'visionStatus' : 'llmStatus').textContent = text;
  renderStatus();
}
const modelSize = () => document.querySelector('input[name=size]:checked')?.value || '1b';
const devicePrefValue = () => document.querySelector('input[name=device]:checked')?.value || 'auto';
// Auto: GPU when WebGPU exists and hasn't failed before on this device, else CPU.
function llmDevice() {
  const v = devicePrefValue();
  if (v !== 'auto') return v;
  return state.gpu && !store.get('gpuFailed', false) ? 'webgpu' : 'wasm';
}

function renderStatus() {
  const labels = { idle: 'Not loaded', loading: 'Loading', ready: 'Ready', error: 'Error' };
  $('visionBadge').textContent = labels[state.vision];
  $('llmBadge').textContent = labels[state.llm];
  $('visionBadge').dataset.state = state.vision;
  $('llmBadge').dataset.state = state.llm;

  const both = state.vision === 'ready' && state.llm === 'ready';
  const any = state.vision === 'loading' || state.llm === 'loading';
  $('statusDot').dataset.state = both ? 'ready' : any ? 'loading' : state.vision === 'error' || state.llm === 'error' ? 'error' : 'idle';
  $('statusText').textContent = both
    ? (!navigator.onLine ? 'Offline and ready' : state.offline === false ? 'Ready (online only)' : 'Ready, works offline')
    : any ? 'Loading models…' : 'Models not downloaded';
  $('downloadBtn').disabled = any || both;
  $('downloadBtn').textContent = both ? 'Downloaded ✓' : any ? 'Downloading…' : 'Download models';
}

// ---------- Tabs ----------

function showTab(name) {
  document.querySelectorAll('.tab').forEach((t) => t.setAttribute('aria-selected', String(t.dataset.tab === name)));
  document.querySelectorAll('.panel').forEach((p) => { p.hidden = p.id !== `panel-${name}`; });
  store.set('tab', name);
  if (name === 'journal') renderJournal();
}
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => showTab(t.dataset.tab)));
$('statusPill').addEventListener('click', () => showTab('setup'));

// ---------- Setup ----------

$('downloadBtn').addEventListener('click', () => {
  // Ask the browser not to evict ~1 GB of models when space gets tight.
  navigator.storage?.persist?.().catch(() => {});
  ensure('vision').catch(() => {});
  ensure('llm').catch(() => {});
});
$('clearBtn').addEventListener('click', async () => {
  if (!confirm('Remove the downloaded models from this device? Your journal is kept.')) return;
  await call('clear-models');
  store.set('downloaded', {});
  location.reload();
});

document.querySelectorAll('input[name=size], input[name=device]').forEach((r) => r.addEventListener('change', () => {
  store.set(r.name, r.value);
  if (r.name === 'device' && r.value === 'auto') store.set('gpuFailed', false);
  if (state.llm === 'ready' || state.llm === 'error') {
    state.llm = 'idle';
    delete loaders.llm;
    setBar('llm', 0);
    ensure('llm').catch(() => {});
  }
}));

// ---------- Identify ----------

let current = null; // { blob, thumb, top, chosen, note }
let noteGen = 0;

$('pickBtn').addEventListener('click', () => $('fileInput').click());
$('fileInput').addEventListener('change', (e) => { const f = e.target.files?.[0]; if (f) identify(f); e.target.value = ''; });
const drop = $('drop');
drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('fileInput').click(); } });
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => { const f = [...(e.dataTransfer?.files || [])].find((x) => x.type.startsWith('image/')); if (f) identify(f); });
window.addEventListener('paste', (e) => {
  const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith('image/'));
  if (f) { showTab('identify'); identify(f); }
});

let stream = null;
$('camBtn').addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    $('video').srcObject = stream;
    await $('video').play();
    $('camera').hidden = false;
    drop.hidden = true;
  } catch {
    toast('Camera not available. Choose a photo instead.');
  }
});
function closeCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  $('camera').hidden = true;
  drop.hidden = false;
}
$('camClose').addEventListener('click', closeCamera);
$('snapBtn').addEventListener('click', () => {
  const v = $('video');
  const c = document.createElement('canvas');
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  c.toBlob((b) => { closeCamera(); if (b) identify(b); }, 'image/jpeg', 0.9);
});

async function makeThumb(blob, size = 480) {
  const bmp = await createImageBitmap(blob);
  const scale = Math.min(1, size / Math.max(bmp.width, bmp.height));
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * scale); c.height = Math.round(bmp.height * scale);
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close?.();
  return c.toDataURL('image/jpeg', 0.8);
}

async function identify(blob) {
  const thumb = await makeThumb(blob).catch(() => null);
  if (!thumb) { toast("Couldn't read that image."); return; }
  current = { blob, thumb, top: [], chosen: null, note: '' };
  $('resultCard').hidden = false;
  $('resultImg').src = thumb;
  $('resultName').textContent = 'Looking…';
  $('resultGroup').textContent = '';
  $('confBar').style.width = '0%';
  $('confText').textContent = state.vision === 'ready' ? '' : 'Loading the vision model (first time only)…';
  $('alts').innerHTML = '';
  $('altLabel').hidden = true;
  $('noteStream').innerHTML = '';
  $('userNote').value = '';
  $('saveFind').disabled = true;
  $('resultCard').scrollIntoView({ behavior: 'smooth', block: 'start' });

  try {
    await ensure('vision');
    const { top } = await call('classify', { blob });
    current.top = top;
    choose(top[0], true);
  } catch (err) {
    $('resultName').textContent = "Couldn't identify";
    $('confText').textContent = err.message;
  }
}

function confidenceWord(p) {
  if (p >= 0.6) return 'Confident';
  if (p >= 0.3) return 'Likely';
  return 'Unsure';
}

function choose(pick, auto = false) {
  current.chosen = { ...pick, confirmed: !auto };
  const g = GROUPS[pick.group];
  $('resultGroup').textContent = `${g.emoji} ${g.label}`;
  $('resultName').textContent = cap(pick.name);
  $('confBar').style.width = `${Math.round(pick.p * 100)}%`;
  $('confBar').dataset.level = confidenceWord(pick.p).toLowerCase();
  $('confText').textContent = auto
    ? `${confidenceWord(pick.p)}: ${Math.round(pick.p * 100)}% match against ${LABELS.length} field-guide entries`
    : 'You picked this one';
  const alts = current.top.filter((t) => !(t.name === pick.name && t.group === pick.group)).slice(0, 4);
  $('altLabel').hidden = alts.length === 0;
  $('alts').innerHTML = '';
  for (const a of alts) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip';
    b.textContent = `${GROUPS[a.group].emoji} ${cap(a.name)} · ${Math.round(a.p * 100)}%`;
    b.addEventListener('click', () => choose(a));
    $('alts').appendChild(b);
  }
  $('saveFind').disabled = false;
  writeNote();
}

const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

const NOTE_PREFILL = '**Check it:** ';

const SAFETY = {
  bird: 'Keep your distance, especially in spring when there may be a nest nearby.',
  tree: 'Leave bark and branches as you found them.',
  leaf: 'Wash your hands afterwards; some leaves irritate skin.',
  flower: 'Leave the flowers for the pollinators; take a photo instead of picking.',
  fungus: 'Never eat a wild mushroom identified by an app.',
  insect: 'Watch without touching; some bugs sting or bite.',
  animal: 'Give wild animals space and never feed them.',
  plant: 'Wash your hands after touching unknown plants.',
};
const NEARBY = {
  bird: 'Stand still for one minute and count how many different bird calls you can hear.',
  tree: 'Find a fallen leaf or seed from this tree on the ground below it.',
  leaf: 'Find the tree or plant this leaf came from by matching the shape.',
  flower: 'Watch the flowers for two minutes and see which insects visit.',
  fungus: 'Look for the dead wood or roots it may be growing from.',
  insect: 'Find the plant it was on and look for chewed leaves or eggs.',
  animal: 'Look for its tracks or droppings on soft ground nearby.',
  plant: 'Find another plant growing in the same spot and compare the leaves.',
};

// A worked example teaches the small model the format far better than rules.
const NOTE_EXAMPLE = `Trusted traits for monarch butterfly: orange wings with thick black veins; black borders with white dots; glides with wings held in a V.
Note:
**Check it:** Look for thick black veins across orange wings, and white dots along the black edges.
**Look nearby:** Find a sunny patch of flowers and watch for other butterflies feeding.`;

function notePrompt({ name, group, traits }) {
  if (group === 'other' || !traits) {
    return {
      prefill: '**Look nearby:** ',
      content: `I'm outside and photographed ${name}. In one or two sentences, suggest one common living thing (a plant, bird or insect) I could look for within a few steps, and the one feature that helps me spot it.`,
    };
  }
  return {
    prefill: NOTE_PREFILL,
    content: `Write a two-line field note for someone outdoors, using only the trusted traits. Follow the example exactly.

${NOTE_EXAMPLE}

Trusted traits for ${name}: ${traits}.
Note:`,
  };
}

// Small models sometimes echo the instructions or drop a section. If so, build
// the note straight from the trusted traits instead of showing nonsense.
function validNote(text, pick) {
  if (pick.group === 'other' || !pick.traits) return text.length > 20 && !/\b(sentence|suggest one|one feature)\b/i.test(text);
  const labels = ['**Check it:**', '**Look nearby:**'].every((l) => text.includes(l));
  const echoed = /traits above|phrased as|trusted traits|follow the example|one simple, related thing/i.test(text)
    || (/butterfl/i.test(text) && !/butterfl/i.test(pick.name)); // copied the worked example
  // The "Check it" line must actually use the trusted traits.
  const check = (text.match(/\*\*Check it:\*\*([^\n]*)/) || [])[1]?.toLowerCase() || '';
  const traitWords = new Set(pick.traits.toLowerCase().match(/[a-z]{4,}/g));
  const overlap = (check.match(/[a-z]{4,}/g) || []).filter((w) => traitWords.has(w)).length;
  return labels && !echoed && overlap >= 2;
}

function templateNote({ name, group, traits }) {
  if (group === 'other' || !traits) return `**Look nearby:** ${NEARBY.plant}`;
  const t = traits.split(';').map((x) => x.trim()).filter(Boolean);
  return [
    `**Check it:** ${cap(t[0])}${t[1] ? `; ${t[1]}` : ''}.`,
    `**Look nearby:** ${NEARBY[group]}`,
  ].join('\n\n');
}

async function writeNote() {
  const my = ++noteGen;
  const out = $('noteStream');
  out.innerHTML = '';
  out.classList.add('typing');
  $('stopNote').hidden = false;
  $('noteMeta').textContent = '';
  if (state.llm !== 'ready') out.innerHTML = '<p class="muted">Waking up Gemma (first time takes a moment)…</p>';
  try {
    await ensure('llm');
    if (my !== noteGen) return;
    const { content, prefill } = notePrompt(current.chosen);
    let text = prefill;
    out.innerHTML = renderMarkdown(text);
    const { text: final, tps } = await call('generate', {
      messages: [{ role: 'user', content }],
      prefill,
      size: modelSize(),
      device: llmDevice(),
      maxTokens: 110,
    }, (chunk) => {
      if (my !== noteGen) return;
      text += chunk;
      out.innerHTML = renderMarkdown(text);
    });
    if (my === noteGen) {
      const note = tidyNote(prefill + final);
      const body = validNote(note, current.chosen) ? note : templateNote(current.chosen);
      // Safety advice is too important to leave to a 1B model: it comes from a fixed list.
      const safety = SAFETY[current.chosen.group];
      current.note = safety ? `${body}\n\n**Careful:** ${safety}` : body;
      out.innerHTML = renderMarkdown(current.note);
      if (tps) $('noteMeta').textContent = `${state.llmDevice || 'On this device'} · ${tps.toFixed(1)} tokens/s · nothing left this device`;
    }
  } catch (err) {
    if (my === noteGen) out.innerHTML = `<p class="muted">Field note unavailable: ${escapeHtml(err.message)}</p>`;
  } finally {
    if (my === noteGen) { out.classList.remove('typing'); $('stopNote').hidden = true; }
  }
}

// Put each labelled part on its own line; keep only the two parts Gemma writes.
function tidyNote(text) {
  const t = text.replace(/\s*(\*\*(?:Check it|Look nearby|Careful):\*\*)\s*/g, '\n\n$1 ').trim();
  const parts = t.split(/\n\n(?=\*\*)/);
  return parts.filter((x) => !x.startsWith('**Careful:**')).slice(0, 2).join('\n\n');
}

// If the user switches species mid-note, stop the old one so the new one starts promptly.
const stopGen = () => worker?.postMessage({ type: 'stop' });
$('stopNote').addEventListener('click', stopGen);
$('alts').addEventListener('click', stopGen, true);

function getLocation() {
  return new Promise((resolve) => {
    if (!navigator.geolocation) return resolve(null);
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: +pos.coords.latitude.toFixed(4), lon: +pos.coords.longitude.toFixed(4) }),
      () => resolve(null),
      { enableHighAccuracy: false, timeout: 8000, maximumAge: 300000 },
    );
  });
}

$('saveLoc').checked = store.get('saveLoc', false);
$('saveLoc').addEventListener('change', (e) => store.set('saveLoc', e.target.checked));

$('saveFind').addEventListener('click', async () => {
  if (!current?.chosen) return;
  $('saveFind').disabled = true;
  const loc = $('saveLoc').checked ? await getLocation() : null;
  await saveFind({
    id: crypto.randomUUID(),
    createdAt: Date.now(),
    name: current.chosen.name,
    group: current.chosen.group,
    confidence: current.chosen.p,
    confirmed: current.chosen.confirmed,
    note: current.note,
    userNote: $('userNote').value.trim(),
    thumb: current.thumb,
    loc,
    walkId: activeWalk?.id || null,
  });
  toast(`Saved ${cap(current.chosen.name)} to your journal`);
  if (activeWalk) { $('resultCard').hidden = true; $('walkMode').hidden = false; }
});

$('again').addEventListener('click', () => {
  stopGen();
  $('resultCard').hidden = true;
  if (activeWalk) $('walkMode').hidden = false;
  else window.scrollTo({ top: 0, behavior: 'smooth' });
});

// ---------- Walk ----------

function season(date = new Date(), lat = store.get('lastLat', 20)) {
  const m = date.getMonth();
  const north = ['winter', 'winter', 'spring', 'spring', 'spring', 'summer', 'summer', 'summer', 'autumn', 'autumn', 'autumn', 'winter'];
  const s = north[m];
  if (lat >= 0) return s;
  return { winter: 'summer', summer: 'winter', spring: 'autumn', autumn: 'spring' }[s];
}

let quests = [];
let activeWalk = null;
let timerHandle = null;

function parseQuests(text) {
  return text
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => /^(\d+[.)]|[-*•])\s+/.test(l))
    .map((l) => l.replace(/^(\d+[.)]|[-*•])\s+/, '').replace(/<[^>]*>|\*\*/g, '').trim())
    .filter((l) => l.length > 6 && !/crack in the pavement|rooftop or wire|two street trees/i.test(l)) // not the prompt's example
    .slice(0, 5);
}

function renderQuestList(el, interactive) {
  el.innerHTML = '';
  quests.forEach((q, i) => {
    const li = document.createElement('li');
    li.className = q.done ? 'done' : '';
    if (interactive) {
      const label = document.createElement('label');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = q.done;
      cb.addEventListener('change', () => { quests[i].done = cb.checked; li.className = cb.checked ? 'done' : ''; saveWalk(); });
      const span = document.createElement('span');
      span.textContent = q.text;
      label.append(cb, span);
      li.appendChild(label);
    } else {
      li.textContent = q.text;
    }
    el.appendChild(li);
  });
}

$('walkForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const minutes = +fd.get('time');
  const place = fd.get('place');
  const s = season();
  const recent = (await listFinds()).slice(0, 6).map((f) => f.name);
  const count = minutes <= 15 ? 3 : minutes <= 30 ? 4 : 5;

  $('questCard').hidden = false;
  $('questList').innerHTML = '';
  $('questStream').innerHTML = state.llm === 'ready' ? '' : '<p class="muted">Waking up Gemma (first time takes a moment)…</p>';
  $('questStream').classList.add('typing');
  $('questMeta').textContent = `${minutes} min · ${place} · ${s}`;
  $('startWalk').disabled = true;
  $('walkBtn').disabled = true;
  $('stopQuest').hidden = false;
  $('questCard').scrollIntoView({ behavior: 'smooth', block: 'start' });

  const prompt = `Make a nature scavenger hunt for a ${minutes}-minute walk in ${place} during ${s}.
Write exactly ${count} numbered items. Each item starts with an action verb (Find, Spot, Listen for, Touch, Smell, Count) and names one specific thing a beginner can find without equipment. Mix plants, animals, sounds, and small details like bark, seeds or tracks. Under 15 words each.${recent.length ? ` They already found: ${recent.join(', ')}, so pick different things.` : ''}

Example for a 15-minute walk on city streets in spring:
1. Find a weed flowering in a crack in the pavement.
2. Listen for a bird singing from a rooftop or wire.
3. Touch the bark of two street trees and compare them.

Only output the numbered list.`;
  const prefill = '1. ';

  try {
    await ensure('llm');
    let text = prefill;
    $('questStream').innerHTML = '';
    const { text: final } = await call('generate', {
      messages: [{ role: 'user', content: prompt }],
      prefill,
      size: modelSize(),
      device: llmDevice(),
      maxTokens: 200,
    }, (chunk) => { text += chunk; $('questStream').innerHTML = renderMarkdown(text); });
    const parsed = parseQuests(prefill + final);
    quests = (parsed.length ? parsed : fallbackQuests(place)).map((t) => ({ text: t, done: false }));
    $('questStream').innerHTML = '';
    renderQuestList($('questList'), false);
    $('startWalk').disabled = false;
    $('startWalk').dataset.minutes = minutes;
    $('startWalk').dataset.place = place;
  } catch (err) {
    quests = fallbackQuests(place).map((t) => ({ text: t, done: false }));
    $('questStream').innerHTML = `<p class="muted small">Gemma isn't available (${escapeHtml(err.message)}). Here's a classic list instead.</p>`;
    renderQuestList($('questList'), false);
    $('startWalk').disabled = false;
    $('startWalk').dataset.minutes = minutes;
  } finally {
    $('questStream').classList.remove('typing');
    $('walkBtn').disabled = false;
    $('stopQuest').hidden = true;
  }
});
$('stopQuest').addEventListener('click', stopGen);

function fallbackQuests() {
  return [
    'A leaf with jagged (toothed) edges',
    'A bird you can hear before you see it',
    'Something growing out of a crack in stone or pavement',
    'A tree with bark rough enough to fit a fingertip into the grooves',
    'A seed, nut or pod on the ground',
  ];
}

function saveWalk() { if (activeWalk) store.set('activeWalk', { ...activeWalk, quests }); }

function startWalk(walk) {
  activeWalk = walk;
  quests = walk.quests;
  saveWalk();
  renderQuestList($('walkList'), true);
  $('walkMode').hidden = false;
  document.body.classList.add('walking');
  tick();
  clearInterval(timerHandle);
  timerHandle = setInterval(tick, 1000);
  // Keep the screen dim-friendly and off: we don't request a wake lock on purpose.
}

function tick() {
  if (!activeWalk) return;
  const left = activeWalk.endsAt - Date.now();
  const abs = Math.abs(left);
  const mm = String(Math.floor(abs / 60000)).padStart(2, '0');
  const ss = String(Math.floor((abs % 60000) / 1000)).padStart(2, '0');
  $('timer').textContent = `${left < 0 ? '+' : ''}${mm}:${ss}`;
  $('timer').classList.toggle('over', left < 0);
}

$('startWalk').addEventListener('click', () => {
  const minutes = +$('startWalk').dataset.minutes || 30;
  startWalk({
    id: crypto.randomUUID(),
    startedAt: Date.now(),
    endsAt: Date.now() + minutes * 60000,
    minutes,
    place: $('startWalk').dataset.place || '',
    quests,
  });
  if ($('saveLoc').checked) getLocation().then((l) => { if (l) store.set('lastLat', l.lat); });
});

$('walkIdentify').addEventListener('click', () => {
  $('walkMode').hidden = true;
  showTab('identify');
  $('fileInput').click();
});

$('endWalk').addEventListener('click', () => {
  if (!activeWalk) return;
  const mins = Math.max(1, Math.round((Date.now() - activeWalk.startedAt) / 60000));
  const walks = store.get('walks', []);
  walks.push({ id: activeWalk.id, startedAt: activeWalk.startedAt, minutes: mins, found: quests.filter((q) => q.done).length, total: quests.length, place: activeWalk.place });
  store.set('walks', walks);
  store.set('activeWalk', null);
  activeWalk = null;
  clearInterval(timerHandle);
  $('walkMode').hidden = true;
  document.body.classList.remove('walking');
  toast(`Nice. ${mins} min outside, ${walks.at(-1).found}/${walks.at(-1).total} found.`);
  showTab('journal');
});

// ---------- Journal ----------

async function renderJournal() {
  const finds = await listFinds();
  const walks = store.get('walks', []);
  const minutes = walks.reduce((a, w) => a + w.minutes, 0);
  const species = new Set(finds.filter((f) => f.group !== 'other').map((f) => f.name)).size;
  $('stats').innerHTML = [
    ['Minutes outside', minutes],
    ['Walks', walks.length],
    ['Finds', finds.length],
    ['Different species', species],
  ].map(([k, v]) => `<div class="stat"><span class="stat-v">${v}</span><span class="stat-k">${k}</span></div>`).join('');

  $('journalEmpty').hidden = finds.length > 0;
  $('exportBtn').hidden = finds.length === 0;
  const el = $('journal');
  el.innerHTML = '';
  for (const f of finds) {
    const card = document.createElement('article');
    card.className = 'find';
    const date = new Date(f.createdAt).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
    card.innerHTML = `
      <img src="${f.thumb}" alt="${escapeHtml(f.name)}" loading="lazy" />
      <div class="find-body">
        <p class="eyebrow">${GROUPS[f.group]?.emoji || ''} ${escapeHtml(GROUPS[f.group]?.label || f.group)}${f.confirmed ? ' · you confirmed' : ''}</p>
        <h3>${escapeHtml(cap(f.name))}</h3>
        <p class="muted small">${date}${f.loc ? ` · ${f.loc.lat}, ${f.loc.lon}` : ''}</p>
        ${f.userNote ? `<p class="small">“${escapeHtml(f.userNote)}”</p>` : ''}
        <details><summary class="small">Field note</summary><div class="small">${renderMarkdown(f.note || '')}</div></details>
        <button class="btn ghost small del" type="button">Delete</button>
      </div>`;
    card.querySelector('.del').addEventListener('click', async () => {
      if (!confirm(`Delete ${cap(f.name)} from your journal?`)) return;
      await deleteFind(f.id);
      renderJournal();
    });
    el.appendChild(card);
  }
}

$('exportBtn').addEventListener('click', async () => {
  const finds = await listFinds();
  const md = ['# My field journal', ''];
  for (const f of finds) {
    md.push(`## ${cap(f.name)} (${f.group})`);
    md.push(`*${new Date(f.createdAt).toLocaleString()}*${f.loc ? ` · ${f.loc.lat}, ${f.loc.lon}` : ''} · ${Math.round(f.confidence * 100)}% match${f.confirmed ? ', confirmed by me' : ''}`);
    if (f.userNote) md.push('', `> ${f.userNote}`);
    if (f.note) md.push('', f.note);
    md.push('');
  }
  const url = URL.createObjectURL(new Blob([md.join('\n')], { type: 'text/markdown' }));
  const a = Object.assign(document.createElement('a'), { href: url, download: 'field-journal.md' });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

// ---------- Misc ----------

let toastTimer;
function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 3500);
}

async function deviceInfo() {
  try {
    const a = await navigator.gpu?.requestAdapter();
    state.gpu = a ? { vendor: a.info?.vendor || 'unknown', fp16: a.features.has('shader-f16') } : null;
  } catch { state.gpu = null; }
  const gpu = state.gpu ? `available (${state.gpu.vendor})` : 'not available';
  const threads = self.crossOriginIsolated ? `${navigator.hardwareConcurrency || '?'} CPU threads` : '1 CPU thread (page is not cross-origin isolated)';
  $('deviceInfo').textContent = `WebGPU: ${gpu}. ${threads}. On CPU, Gemma uses the 270M model.`;
}

window.addEventListener('online', renderStatus);
window.addEventListener('offline', renderStatus);

// ---------- Boot ----------

for (const [name, d] of [['size', '1b'], ['device', 'auto']]) {
  const input = document.querySelector(`input[name=${name}][value="${store.get(name, d)}"]`);
  if (input) input.checked = true;
}

startWorker();
showTab(store.get('tab', 'walk'));
renderStatus();

// If models were downloaded before, warm them from the local cache in the background.
gpuReady = deviceInfo();
gpuReady.then(() => {
  const downloaded = store.get('downloaded', {});
  if (downloaded.vision) ensure('vision').catch(() => {});
  if (downloaded.llm) ensure('llm').catch(() => {});
});

// Resume a walk that was in progress when the page was closed.
const resumable = store.get('activeWalk', null);
if (resumable?.endsAt) startWalk(resumable);

if ('serviceWorker' in navigator && import.meta.env.PROD) {
  navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`).catch(() => {});
}
