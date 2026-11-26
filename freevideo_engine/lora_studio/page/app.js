// LoRA Studio page. Plain module, no build step. It talks to:
//   /freevideo/lora-studio/api/*   datasets, imports, exports, training (this fork's routes)
//   /freevideo/image-gen/*         the engine's picture models (H3 Image, Qwen-Image, MOSS) for the coverage set
//   /freevideo/prompt-vlm/*        the engine's prompt model (Qwen3-VL) for captions and likeness checks
//   /freevideo/agent/cast*         cast cards, so a finished LoRA loads when @Name is written
//   /prompt, /history              ComfyUI's queue, for the strength test grid
const $ = (selector, root = document) => root.querySelector(selector);
function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
        if (value === undefined || value === null || value === false) continue;
        if (key === 'class') node.className = value;
        else if (key === 'text') node.textContent = value;
        else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
        else if (key in node && typeof value !== 'string') node[key] = value;
        else node.setAttribute(key, value === true ? '' : value);
    }
    for (const child of children.flat()) if (child !== null && child !== undefined && child !== false) node.append(child);
    return node;
}
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const view = file => { const at = file.lastIndexOf('/'); return '/view?' + new URLSearchParams({filename: file.slice(at + 1), subfolder: at < 0 ? '' : file.slice(0, at), type: 'input'}) + '&v=' + (state.bust || 0); };
const randomSeed = () => crypto.getRandomValues(new Uint32Array(1))[0] >>> 1;
let toastTimer = null;
function toast(text, bad = false) {
    const box = $('#toast'); box.textContent = text; box.className = 'toast' + (bad ? ' bad' : ''); box.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { box.hidden = true; }, bad ? 9000 : 4000);
}
const ERRORS = {
    reload: 'FreeVideo restarted. Reload this page.', unknown_dataset: 'That LoRA set no longer exists.',
    invalid_media: 'That file could not be read as a picture.', too_large: 'That picture is too large (40 MB at most).',
    trainer_missing: 'The trainer (ai-toolkit) was not found. Set its folder in Train → Trainer setup, or run scripts\\install-lora-trainer.ps1.',
    gpu_busy: 'A video is being made right now. Train when the queue is empty.', training_busy: 'A training run is already going.',
    empty_dataset: 'There are no pictures to use yet (or all are excluded).', no_lora: 'Train or choose a LoRA first.',
    invalid_trigger: 'The trigger word can use letters, numbers, spaces, _ and - (40 at most).', too_many_items: 'A set holds 400 pictures at most.',
    segment_not_installed: 'Background removal needs the small selection model. Install it from the Pictures tab.',
    busy: 'The model is busy with another job; trying again shortly.', busy_video: 'A video is being made; pictures wait until it ends.',
    model_missing: 'The prompt model is not installed. Install it in FreeVideo (Create → Enhance prompt) first.',
    lora_not_found: 'That LoRA file is not in ComfyUI\'s LoRA list.', unknown_checkpoint: 'That checkpoint is gone.',
};
const message = error => ERRORS[error?.code] || error?.message || String(error);
class Failure extends Error { constructor(code, detail = {}) { super(code); this.code = code; Object.assign(this, detail); } }

// ── Clients ────────────────────────────────────────────────────────────────────────────────────────────────────
let studioToken = '';
async function studioInfo() {
    const response = await fetch('/freevideo/lora-studio/api/info', {cache: 'no-store'});
    const value = await response.json();
    if (!response.ok) throw new Failure(value.error || 'unavailable');
    studioToken = value.token; return value;
}
async function studio(action, body = {}) {
    const response = await fetch('/freevideo/lora-studio/api/' + action, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-FreeVideo-Lora': studioToken}, body: JSON.stringify(body)});
    const value = await response.json().catch(() => ({error: 'failed'}));
    if (!response.ok) throw new Failure(value.error || 'failed');
    return value;
}
async function studioUpload(dataset, file, {source = 'import', crop = true, dedupe = true} = {}) {
    const query = new URLSearchParams({dataset, source, crop: crop ? '1' : '0', dedupe: dedupe ? '1' : '0'});
    const response = await fetch('/freevideo/lora-studio/api/upload?' + query, {method: 'POST', headers: {'X-FreeVideo-Lora': studioToken, 'Content-Type': 'application/octet-stream'}, body: file});
    const value = await response.json().catch(() => ({error: 'failed'}));
    if (!response.ok) throw new Failure(value.error || 'failed');
    return value;
}

const imageGen = {
    token: '', info: null,
    async load() {
        const response = await fetch('/freevideo/image-gen', {cache: 'no-store'});
        const value = await response.json().catch(() => ({}));
        if (!response.ok) throw new Failure(value.error || 'setup_required');
        this.token = value.token; this.info = value; return value;
    },
    async post(action, body, retried = false) {
        if (!this.token) await this.load();
        const response = await fetch('/freevideo/image-gen/' + action, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-FreeVideo-Image': this.token}, body: JSON.stringify(body)});
        const value = await response.json().catch(() => ({error: 'engine_error'}));
        if (response.status === 403 && !retried) { this.token = ''; return this.post(action, body, true); }
        if (!response.ok || (value.error && !value.job && value.phase !== 'complete')) throw new Failure(value.error || 'generation_failed', value);
        return value;
    },
    ready(id) { return (this.info?.models || []).some(row => row.id === id && row.ready === true); },
    // H3 Image keeps a reference's identity best; Qwen-Image also edits from references; MOSS draws from words only.
    editModel() { return this.ready('minimax-h3-image') ? 'minimax-h3-image' : this.ready('qwen-image-2.1-turbo') ? 'qwen-image-2.1-turbo' : ''; },
    textModel() { return this.ready('moss-image-sft') ? 'moss-image-sft' : this.editModel(); },
    fit(model, width, height) {
        const limit = model === 'minimax-h3-image' ? 4096 : 2048, scale = Math.min(1, limit / Math.max(width, height));
        const step = model === 'moss-image-sft' ? 16 : 32;
        const snap = v => Math.max(256, Math.round(v * scale / step) * step);
        if (model === 'moss-image-sft') { const s = Math.sqrt(1024 * 1024 / (width * height)); return [Math.round(width * s / 16) * 16, Math.round(height * s / 16) * 16]; }
        return [snap(width), snap(height)];
    },
    async run({model, prompt, width, height, images = [], seed}, onProgress = () => {}, signal = null) {
        const [w, h] = this.fit(model, width, height);
        let started;
        for (let attempt = 0; ; attempt++) {
            try { started = await this.post('generate', {...(model ? {model} : {}), prompt, width: w, height: h, images, seed}); break; }
            catch (error) { if (!['busy', 'busy_video'].includes(error.code) || attempt > 120) throw error; onProgress({waiting: true}); await sleep(5000); }
        }
        const job = started.job;
        for (;;) {
            await sleep(700);
            if (signal?.aborted) { this.post('cancel', {job}).catch(() => {}); throw new Failure('cancelled'); }
            const status = await this.post('status', {job});
            if (status.phase === 'complete') { if (!status.file) throw new Failure('result_missing'); return status.file; }
            if (status.phase === 'failed' || status.phase === 'cancelled') throw new Failure(status.error || status.phase, status);
            onProgress({fraction: status.fraction || 0, stage: status.stage || '', waiting: status.phase === 'queued'});
        }
    },
};

const vlm = {
    token: '',
    async load() {
        const response = await fetch('/freevideo/prompt-vlm', {cache: 'no-store'});
        const value = await response.json().catch(() => ({}));
        if (!response.ok) throw new Failure(value.error || 'setup_required');
        this.token = value.token; return value;
    },
    async post(action, body, retried = false) {
        if (!this.token) await this.load();
        const response = await fetch('/freevideo/prompt-vlm/' + action, {method: 'POST', headers: {'Content-Type': 'application/json', 'X-FreeVideo-Prompt': this.token}, body: JSON.stringify(body)});
        const value = await response.json().catch(() => ({error: 'failed'}));
        if (response.status === 403 && !retried) { this.token = ''; return this.post(action, body, true); }
        if (!response.ok) throw new Failure(value.error || 'failed');
        return value;
    },
    // Up to 8 questions per job, each with its pictures. Returns one answer per question ({text, json?, error?}).
    async ask(questions, {system, format = 'text', max_tokens = 300} = {}, onWait = () => {}) {
        let started;
        for (let attempt = 0; ; attempt++) {
            try { started = await this.post('ask', {questions, format, max_tokens, ...(system ? {system} : {})}); break; }
            catch (error) { if (error.code !== 'busy' || attempt > 120) throw error; onWait(); await sleep(5000); }
        }
        for (;;) {
            await sleep(800);
            const status = await this.post('status', {job: started.job});
            if (status.phase === 'complete') return status.answers || [];
            if (status.phase === 'failed' || status.phase === 'cancelled') throw new Failure(status.error || 'failed');
        }
    },
};

const agent = {
    csrf: null,
    async headers() {
        if (!this.csrf) {
            const response = await fetch('/freevideo/agent/session', {cache: 'no-store', credentials: 'same-origin'});
            if (!response.ok) throw new Failure('agent_off', {message: 'Cast cards need FreeVideo\'s assistant features. Turn them on in FreeVideo settings, or link the LoRA by hand from the creation panel.'});
            this.csrf = (await response.json()).csrf;
        }
        return {'X-FreeVideo-CSRF': this.csrf};
    },
    async send(method, path, body) {
        for (let attempt = 0; attempt < 2; attempt++) {
            const response = await fetch(path, {method, credentials: 'same-origin', cache: 'no-store', headers: {...(body ? {'Content-Type': 'application/json'} : {}), ...await this.headers()}, ...(body ? {body: JSON.stringify(body)} : {})});
            if ((response.status === 401 || response.status === 403) && !attempt) { this.csrf = null; continue; }
            const value = await response.json().catch(() => null);
            if (!response.ok) throw new Failure('agent_error', {message: value?.error?.message || `Cast request failed (${response.status})`});
            return value;
        }
    },
    cards: kind => agent.send('GET', `/freevideo/agent/cast?kind=${kind}&limit=200`).then(value => value?.cards || []),
    create: body => agent.send('POST', '/freevideo/agent/cast', body),
};

const comfy = {
    clientId: crypto.randomUUID(),
    async queue(prompt) {
        const response = await fetch('/prompt', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({prompt, client_id: this.clientId})});
        const value = await response.json().catch(() => ({}));
        if (!response.ok) throw new Failure('queue_failed', {message: value?.error?.message || value?.error || 'ComfyUI refused the test prompt.'});
        return value.prompt_id;
    },
    async result(promptId) {
        const response = await fetch('/history/' + encodeURIComponent(promptId), {cache: 'no-store'});
        const value = await response.json().catch(() => ({}));
        const entry = value[promptId];
        if (!entry) return null;
        const found = [];
        const walk = node => { if (!node || typeof node !== 'object') return; if (typeof node.filename === 'string' && /\.(mp4|webm|mov)$/i.test(node.filename)) found.push(node); else Object.values(node).forEach(walk); };
        walk(entry.outputs);
        const failed = entry.status?.status_str === 'error';
        return {done: Boolean(entry.status?.completed) || failed || found.length > 0, failed, video: found[0] || null,
            message: failed ? (entry.status?.messages || []).map(m => m?.[1]?.exception_message).filter(Boolean).join(' ') : ''};
    },
};

// ── State ──────────────────────────────────────────────────────────────────────────────────────────────────────
const state = {info: null, datasets: [], ds: null, tab: 'pictures', jobs: new Map(), questions: null, bust: Date.now(), queue: [], running: false, abort: null, trainTimer: null};
const KIND_WORDS = {character: 'Character', object: 'Object', style: 'Style'};

// A one-at-a-time work queue: the engine runs one picture job and one prompt-model job at a time anyway.
function enqueue(label, work) {
    return new Promise((resolve, reject) => { state.queue.push({label, work, resolve, reject}); pump(); renderQueueNote(); });
}
async function pump() {
    if (state.running) return;
    const next = state.queue.shift();
    if (!next) { renderQueueNote(); return; }
    state.running = next; renderQueueNote();
    try { next.resolve(await next.work()); } catch (error) { next.reject(error); }
    state.running = false; pump();
}
function stopQueue() {
    for (const job of state.queue.splice(0)) job.reject(new Failure('cancelled'));
    state.abort?.abort(); state.abort = null; renderQueueNote();
}
function renderQueueNote() {
    const note = $('#gpu-note');
    const count = state.queue.length + (state.running ? 1 : 0);
    note.hidden = !count; note.className = 'pill warn';
    note.textContent = count ? `${state.running?.label || 'Working'}${count > 1 ? ` · ${count - 1} more waiting` : ''}` : '';
}

async function refreshList() {
    state.datasets = (await studio('list')).datasets;
    const list = $('#dataset-list'); list.replaceChildren();
    for (const row of state.datasets) {
        list.append(el('div', {class: 'ds-row' + (state.ds?.id === row.id ? ' on' : ''), onclick: () => openDataset(row.id)},
            row.cover ? el('img', {src: view(row.cover), alt: '', loading: 'lazy'}) : el('div', {class: 'ph'}),
            el('div', {class: 'grow', style: 'min-width:0'}, el('div', {class: 't', text: row.name}),
                el('div', {class: 's', text: `${KIND_WORDS[row.kind] || row.kind} · ${row.count} pictures${row.lora ? ' · trained' : ''}`}))));
    }
    if (!state.datasets.length) list.append(el('div', {class: 'muted', text: 'No LoRA sets yet.'}));
}
async function openDataset(id) {
    state.ds = await studio('get', {dataset: id}); state.bust = Date.now();
    state.questions = null;
    $('#welcome').hidden = true; $('#editor').hidden = false;
    try { localStorage.setItem('loraStudio.last', id); } catch {}
    renderHead(); renderTab(); refreshList();
}
let listTimer = null;
function setDataset(value) {
    if (value?.id && value.id === state.ds?.id) { state.ds = value; state.bust = Date.now(); renderHead(); }
    clearTimeout(listTimer); listTimer = setTimeout(() => refreshList().catch(() => {}), 1500);
}
function renderHead() {
    const ds = state.ds; const kept = ds.items.filter(i => i.source !== 'reg' && !i.excluded);
    $('#ds-title').textContent = ds.name;
    $('#ds-sub').textContent = `${KIND_WORDS[ds.kind]} · trigger "${ds.trigger}"${ds.class_word ? ` · class "${ds.class_word}"` : ''} · ${kept.length} pictures` + (ds.lora ? ` · LoRA: ${ds.lora.name}` : '');
}
function renderTab() {
    for (const button of document.querySelectorAll('#tabs button')) button.classList.toggle('on', button.dataset.tab === state.tab);
    for (const name of ['pictures', 'coverage', 'captions', 'train', 'use']) $('#tab-' + name).hidden = name !== state.tab;
    ({pictures: renderPictures, coverage: renderCoverage, captions: renderCaptions, train: renderTrain, use: renderUse})[state.tab]();
}
const items = () => state.ds.items.filter(i => i.source !== 'reg');
const reference = () => state.ds.items.find(i => i.id === state.ds.reference) || null;
async function questions() { return state.questions ||= await studio('questions', {dataset: state.ds.id}); }

function viewer(file) {
    const dialog = $('#dialog'); dialog.className = 'viewer';
    dialog.replaceChildren(el('img', {src: view(file), alt: '', onclick: () => dialog.close()}));
    dialog.showModal();
}

function tile(item, {actions = [], extra = []} = {}) {
    const ds = state.ds, badges = [];
    if (item.id === ds.reference) badges.push(el('span', {class: 'badge good', text: 'reference'}));
    badges.push(el('span', {class: 'badge', text: item.source === 'import' ? 'photo' : item.source === 'reg' ? 'regularisation' : (item.slot || 'generated')}));
    if (item.drift?.score != null) badges.push(el('span', {class: 'badge ' + (item.drift.score >= state.info.drift_limit ? 'good' : 'bad'), title: item.drift.differences || '', text: `likeness ${item.drift.score}/10`}));
    if (item.quality?.blurry) badges.push(el('span', {class: 'badge warn', text: 'blurry'}));
    if (item.quality?.small) badges.push(el('span', {class: 'badge warn', text: 'small'}));
    for (const note of item.notes || []) if (['several_people', 'no_face', 'duplicate'].includes(note)) badges.push(el('span', {class: 'badge warn', text: note.replace('_', ' ')}));
    if (item.excluded) badges.push(el('span', {class: 'badge bad', text: 'excluded'}));
    return el('div', {class: 'tile' + (item.id === ds.reference ? ' ref' : '') + (item.excluded ? ' ex' : '')},
        el('img', {src: view(item.input), alt: '', loading: 'lazy', onclick: () => viewer(item.input)}),
        el('div', {class: 'meta'}, badges, extra), el('div', {class: 'acts'}, actions));
}
const act = (text, fn, cls = '') => el('button', {class: 'btn small ' + cls, text, onclick: async event => {
    const button = event.currentTarget; button.disabled = true;
    try { await fn(); } catch (error) { if (error.code !== 'cancelled') toast(message(error), true); } finally { button.disabled = false; }
}});

// ── 1 · Pictures ───────────────────────────────────────────────────────────────────────────────────────────────
function renderPictures() {
    const ds = state.ds, host = $('#tab-pictures');
    const input = el('input', {type: 'file', accept: 'image/*', multiple: true, hidden: true});
    const crop = el('input', {type: 'checkbox', checked: true}), dedupe = el('input', {type: 'checkbox', checked: true});
    const drop = el('div', {class: 'drop', onclick: () => input.click()}, el('div', {text: 'Drop photos here or click to choose'}),
        el('div', {class: 'muted', style: 'font-size:12px', text: ds.kind === 'character' ? 'Clear, varied photos of one person: different angles, expressions, clothes, places. 15–40 is ideal.' : ds.kind === 'object' ? 'The object from different sides and in different places. Remove backgrounds below for a cleaner set.' : 'Pictures in the style you want, of many different subjects.'}));
    const status = el('div', {class: 'muted'});
    async function take(files) {
        let added = 0, repeated = 0, failed = 0, done = 0;
        for (const file of files) {
            status.textContent = `Importing ${++done} of ${files.length}…`;
            try { const result = await studioUpload(ds.id, file, {crop: crop.checked, dedupe: dedupe.checked}); if (result.duplicate_of) repeated++; else added++; setDataset(result.dataset); }
            catch (error) { failed++; toast(`${file.name}: ${message(error)}`, true); }
        }
        status.textContent = `Added ${added}${repeated ? `, skipped ${repeated} near-duplicate${repeated > 1 ? 's' : ''}` : ''}${failed ? `, ${failed} failed` : ''}.`;
        renderPictures();
    }
    input.addEventListener('change', () => take([...input.files]));
    drop.addEventListener('dragover', event => { event.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', event => { event.preventDefault(); drop.classList.remove('over'); take([...event.dataTransfer.files].filter(f => f.type.startsWith('image/'))); });
    const list = items();
    const grid = el('div', {class: 'grid'}, list.map(item => tile(item, {actions: [
        item.id !== ds.reference && act('Reference', async () => { setDataset(await studio('update', {dataset: ds.id, reference: item.id})); renderPictures(); }),
        act(item.excluded ? 'Include' : 'Exclude', async () => { setDataset(await studio('item-update', {dataset: ds.id, item: item.id, excluded: !item.excluded})); renderPictures(); }),
        ds.kind === 'object' && act('Remove background', () => removeBackground(item)),
        act('Delete', async () => { setDataset(await studio('item-delete', {dataset: ds.id, item: item.id})); renderPictures(); }, 'danger'),
    ]})));
    host.replaceChildren(
        el('div', {class: 'card'}, el('h2', {text: 'Import your own pictures'}), drop, input,
            el('div', {class: 'row'}, el('label', {}, crop, ' Auto-crop (trim borders, frame the person)'), el('label', {}, dedupe, ' Skip near-duplicates'), status)),
        el('div', {class: 'card'}, el('div', {class: 'row'}, el('h2', {class: 'grow', style: 'margin:0', text: `Pictures (${list.filter(i => !i.excluded).length} in use)`}),
            act('Find duplicates', async () => { const result = await studio('dedupe', {dataset: ds.id}); setDataset(result.dataset); toast(result.excluded.length ? `Excluded ${result.excluded.length} near-duplicates.` : 'No near-duplicates found.'); renderPictures(); }),
            ds.kind === 'character' && act('Add face close-ups', async () => { const result = await studio('face-crops', {dataset: ds.id}); setDataset(result.dataset); toast(result.added ? `Added ${result.added} close-ups cut from your pictures. Caption them on the Captions tab.` : 'No new close-ups: no faces big enough, or they are already added.'); renderPictures(); }),
            ds.kind === 'object' && act('Remove all backgrounds', async () => { for (const item of items().filter(i => !i.excluded && !(i.notes || []).includes('background_removed'))) await removeBackground(item); })),
            el('p', {class: 'muted', text: 'The reference picture (outlined) is what every generated picture is matched to. Pick the clearest, most typical one. For characters, Add face close-ups cuts head-and-shoulders and half-length crops from full-body pictures so the LoRA learns the face, not only the outfit.'}),
            list.length ? grid : el('div', {class: 'muted', text: 'No pictures yet. Import photos above, or go to Coverage to start from a description.'})));
}

async function removeBackground(item) {
    return enqueue('Removing a background', async () => {
        const staged = (await studio('stage', {dataset: state.ds.id, item: item.id})).file;
        let result;
        try { result = await imageGen.post('segment', {image: staged, points: [[Math.floor(item.width / 2), Math.floor(item.height / 2), 1]]}); }
        catch (error) {
            if (error.code !== 'segment_not_installed') throw error;
            if (!confirm('Background removal uses the small SAM 2 selection model (about 180 MB). Download it now?')) throw new Failure('cancelled');
            await imageGen.post('segment-install', {});
            for (let i = 0; i < 240; i++) { await sleep(2500); try { result = await imageGen.post('segment', {image: staged, points: [[Math.floor(item.width / 2), Math.floor(item.height / 2), 1]]}); break; } catch (again) { if (again.code !== 'segment_not_installed') throw again; } }
            if (!result) throw new Failure('segment_not_installed');
        }
        setDataset(await studio('cutout', {dataset: state.ds.id, item: item.id, mask: result.mask}));
        if (state.tab === 'pictures') renderPictures();
    });
}

// ── 2 · Coverage ───────────────────────────────────────────────────────────────────────────────────────────────
function renderCoverage() {
    const ds = state.ds, host = $('#tab-coverage'), plan = state.info.plans[ds.kind];
    const ref = reference(), model = imageGen.editModel();
    const bySlot = new Map(ds.items.filter(i => i.slot && !i.excluded && i.source === 'generated').map(i => [i.slot, i]));
    const filled = plan.filter(slot => bySlot.has(slot.id)).length;
    const description = el('textarea', {rows: 2, placeholder: ds.kind === 'character' ? 'e.g. a woman in her thirties with short curly red hair and freckles, green parka, jeans, white trainers' : ds.kind === 'object' ? 'e.g. a matte black ceramic coffee mug with a gold rim and a small fox logo' : 'e.g. loose watercolour with ink outlines, muted earth tones, visible paper texture'}, ds.description || '');
    const header = el('div', {class: 'card'}, el('h2', {text: 'Start point'}),
        ref ? el('div', {class: 'row'}, el('img', {src: view(ref.input), style: 'width:72px;height:96px;object-fit:cover;border-radius:6px'}),
            el('div', {class: 'grow'}, el('div', {text: 'Every slot is an edit of this reference picture, so the subject stays the same.'}),
                el('div', {class: 'muted', text: 'Change it on the Pictures tab. A description below helps the model keep details.'})))
            : el('p', {class: 'muted', text: 'No reference yet. Import a photo on the Pictures tab, or describe the subject and make one.'}),
        el('label', {class: 'f', style: 'margin-top:10px'}, 'Description (optional with a photo, needed without one)', description),
        el('div', {class: 'row'},
            act('Save description', async () => { setDataset(await studio('update', {dataset: ds.id, description: description.value})); state.questions = null; toast('Saved.'); }),
            act(ref ? 'Make a new reference from the description' : 'Make the reference from the description', async () => {
                setDataset(await studio('update', {dataset: ds.id, description: description.value}));
                await generateSlot('anchor'); setDataset(await studio('update', {dataset: ds.id, reference: state.ds.items.at(-1).id})); renderCoverage();
            }),
            el('span', {class: 'muted', text: model ? `Picture model: ${model === 'minimax-h3-image' ? 'H3 Image' : 'Qwen-Image'}` : 'No picture model that edits from a reference is installed (H3 Image or Qwen-Image).'})));
    const groups = {};
    for (const slot of plan) (groups[slot.group] ||= []).push(slot);
    const controls = el('div', {class: 'card'}, el('div', {class: 'row'},
        el('h2', {class: 'grow', style: 'margin:0', text: `Coverage: ${filled} of ${plan.length} slots filled`}),
        el('button', {class: 'btn primary', text: 'Fill empty slots', disabled: !ref || !model, onclick: () => fillSlots(plan.filter(s => !bySlot.has(s.id)).map(s => s.id))}),
        el('button', {class: 'btn', text: 'Regenerate drifted', disabled: !ref || !model, onclick: () => fillSlots(plan.filter(s => (bySlot.get(s.id)?.drift?.score ?? 10) < state.info.drift_limit).map(s => s.id))}),
        el('button', {class: 'btn ghost', text: 'Stop', onclick: stopQueue})),
        el('p', {class: 'muted', text: 'Each slot takes about half a minute to a minute with H3 Image. Pictures you don\'t like: Regenerate makes a new one in the same slot (the old one is kept, excluded).'}),
        (() => { const own = ds.items.filter(i => i.source === 'import' && !i.excluded && !(i.notes || []).includes('face_crop')).length, choice = ds.settings?.pictures || 'auto';
            const ownOnly = choice === 'mine' || (choice === 'auto' && own >= 8);
            return ownOnly ? el('p', {style: 'color:var(--warn)', text: `You gave ${own} pictures of your own, so training uses those (and close-ups cut from them) and leaves generated pictures out. Real pictures teach the face better. To train on generated ones too, set Train on to "My pictures and generated ones" on the Train tab.`}) : null; })(),
        Object.entries(groups).map(([group, slots]) => el('div', {class: 'slot-group'}, el('h3', {text: group}),
            el('div', {class: 'grid'}, slots.map(slot => slotTile(slot, bySlot.get(slot.id), Boolean(ref && model)))))));
    host.replaceChildren(header, controls, regularisationCard());
}
function slotTile(slot, item, possible) {
    const job = state.jobs.get(slot.id);
    const body = item ? el('img', {src: view(item.input), alt: '', loading: 'lazy', onclick: () => viewer(item.input)})
        : el('div', {class: 'empty', text: job ? (job.waiting ? 'Waiting for the GPU…' : `Making… ${Math.round((job.fraction || 0) * 100)}%`) : 'Empty'});
    const badges = [el('span', {class: 'label', text: slot.label})];
    if (item?.drift?.score != null) badges.push(el('span', {class: 'badge ' + (item.drift.score >= state.info.drift_limit ? 'good' : 'bad'), title: item.drift.differences || '', text: `${item.drift.score}/10`}));
    return el('div', {class: 'tile slot', 'data-slot': slot.id}, body,
        job ? el('div', {class: 'bar-p', style: `transform:scaleX(${job.fraction || 0.02})`}) : null,
        el('div', {class: 'meta'}, badges),
        el('div', {class: 'acts'}, el('button', {class: 'btn small', text: item ? 'Regenerate' : 'Generate', disabled: !possible || Boolean(job), onclick: () => fillSlots([slot.id])}),
            item && act('Exclude', async () => { setDataset(await studio('item-update', {dataset: state.ds.id, item: item.id, excluded: true})); renderCoverage(); })));
}
function fillSlots(ids) {
    if (!ids.length) { toast('Nothing to make.'); return; }
    for (const id of ids) {
        state.jobs.set(id, {fraction: 0, waiting: true});
        enqueue(`Making pictures (${id.replace(/_/g, ' ')})`, () => generateSlot(id)).catch(error => { if (error.code !== 'cancelled') toast(`${id}: ${message(error)}`, true); })
            .finally(() => { state.jobs.delete(id); if (state.tab === 'coverage') renderCoverage(); });
    }
    renderCoverage();
}
async function generateSlot(slotId, datasetId = state.ds.id) {
    const plan = await studio('slot-prompt', {dataset: datasetId, slot: slotId});
    const ref = reference();
    const model = plan.references ? imageGen.editModel() : (imageGen.editModel() || imageGen.textModel());
    if (!model) throw new Failure('no_model', {message: 'No picture model is installed. Install H3 Image in FreeVideo\'s Images view.'});
    const seed = randomSeed();
    state.abort = new AbortController();
    let file;
    for (let attempt = 0; ; attempt++) {
        // The service removes a staged reference when its job ends, so each attempt stages its own.
        const images = plan.references ? [(await studio('stage', {dataset: datasetId, item: ref.id})).file] : [];
        try {
            file = await imageGen.run({model, prompt: plan.prompt, width: plan.width, height: plan.height, images, seed}, progress => {
                state.jobs.set(slotId, {...state.jobs.get(slotId), ...progress});
                const cell = document.querySelector(`.slot[data-slot="${slotId}"]`);
                if (cell && state.tab === 'coverage') cell.replaceWith(slotTile(state.info.plans[state.ds.kind].find(s => s.id === slotId), null, true));
            }, state.abort.signal);
            break;
        } catch (error) {
            // Out of graphics memory (the video model still held the card): free it once and try again.
            if (attempt > 0 || !/memory/.test(error.code || '')) throw error;
            await studio('free-gpu').catch(() => {});
        }
    }
    const result = await studio('adopt', {dataset: datasetId, file, slot: slotId === 'anchor' ? null : slotId, prompt: plan.prompt, seed, model, source: 'generated'});
    setDataset(result.dataset);
    return result.item;
}
function regularisationCard() {
    const ds = state.ds, reg = ds.items.filter(i => i.source === 'reg');
    const count = el('input', {type: 'number', min: 1, max: 100, value: Math.max(8, Math.min(40, items().length || 8))});
    const use = el('input', {type: 'checkbox', checked: Boolean(ds.settings?.use_regularisation)});
    use.addEventListener('change', async () => { setDataset(await studio('update', {dataset: ds.id, settings: {use_regularisation: use.checked}})); });
    return el('div', {class: 'card'}, el('h2', {text: `Regularisation pictures (${reg.length})`}),
        el('p', {class: 'muted', text: ds.kind === 'style' ? 'Plain photos of everyday subjects, so the style LoRA does not forget what normal pictures look like.' : `Generic pictures of a "${ds.class_word || 'subject'}" that are not your subject. They stop the LoRA turning every ${ds.class_word || 'subject'} into yours. Optional; most useful for people.`}),
        el('div', {class: 'row'}, el('label', {}, use, ' Use them in training'), el('label', {class: 'f'}, 'How many', count),
            act('Make regularisation pictures', async () => {
                const prompts = (await studio('reg-prompts', {dataset: ds.id, count: Number(count.value)})).prompts;
                const model = imageGen.textModel();
                if (!model) throw new Failure('no_model', {message: 'No picture model is installed.'});
                let done = 0;
                for (const prompt of prompts) enqueue(`Regularisation pictures (${++done}/${prompts.length})`, async () => {
                    state.abort = new AbortController();
                    const seed = randomSeed();
                    const file = await imageGen.run({model, prompt, width: 1024, height: 1024, seed}, () => {}, state.abort.signal);
                    setDataset((await studio('adopt', {dataset: ds.id, file, prompt, seed, model, source: 'reg'})).dataset);
                    if (state.tab === 'coverage') renderCoverage();
                }).catch(error => { if (error.code !== 'cancelled') toast(message(error), true); });
            })),
        reg.length ? el('div', {class: 'grid', style: 'margin-top:10px'}, reg.map(item => tile(item, {actions: [act('Delete', async () => { setDataset(await studio('item-delete', {dataset: ds.id, item: item.id})); renderCoverage(); }, 'danger')]}))) : null);
}

// ── 3 · Captions & check ───────────────────────────────────────────────────────────────────────────────────────
function renderCaptions() {
    const ds = state.ds, host = $('#tab-captions');
    const list = items().filter(i => !i.excluded);
    const trigger = el('input', {type: 'text', value: ds.trigger, size: 14});
    const cls = el('input', {type: 'text', value: ds.class_word || '', size: 12, placeholder: ds.kind === 'style' ? '(none)' : 'woman, man, dog, mug…'});
    const mode = el('select', {}, el('option', {value: 'subject', text: 'Subject: learn who/what it is', selected: ds.mode !== 'style'}), el('option', {value: 'style', text: 'Style: learn how it looks', selected: ds.mode === 'style'}));
    const vary = el('input', {type: 'checkbox', checked: Boolean(ds.settings?.vary_clothes)});
    const lead = ds.kind === 'style' || ds.mode === 'style' || !ds.class_word ? ds.trigger : `${ds.trigger} ${ds.class_word}`;
    const status = el('div', {class: 'muted'});
    const rows = list.map(item => {
        const box = el('textarea', {rows: 2, placeholder: 'No caption yet'}, item.caption || '');
        let timer = null;
        box.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(async () => { try { setDataset(await studio('item-update', {dataset: ds.id, item: item.id, caption: box.value})); } catch (error) { toast(message(error), true); } }, 700); });
        return el('div', {class: 'cap-row'}, el('img', {src: view(item.input), alt: '', loading: 'lazy', onclick: () => viewer(item.input)}),
            el('div', {}, el('div', {class: 'lead'}, `Written as: "${lead}, …"`, item.caption_edited ? ' · edited by you' : '',
                item.drift?.score != null ? el('span', {class: 'badge ' + (item.drift.score >= state.info.drift_limit ? 'good' : 'bad'), style: 'margin-left:8px', title: item.drift.differences, text: `likeness ${item.drift.score}/10${item.drift.differences && item.drift.differences !== 'none' ? ': ' + item.drift.differences : ''}`}) : null), box));
    });
    host.replaceChildren(
        el('div', {class: 'card'}, el('h2', {text: 'Trigger and caption style'}),
            el('div', {class: 'row'}, el('label', {class: 'f'}, 'Trigger word', trigger), el('label', {class: 'f'}, 'Class word', cls), el('label', {class: 'f'}, 'What to learn', mode),
                el('label', {}, vary, ' Clothes change between pictures'),
                act('Save', async () => { setDataset(await studio('update', {dataset: ds.id, trigger: trigger.value.trim(), class_word: cls.value.trim(), mode: mode.value, settings: {vary_clothes: vary.checked}})); state.questions = null; renderCaptions(); })),
            el('p', {class: 'muted', text: 'Use a rare trigger word (not a real word) so it starts with no meaning. In subject mode captions describe everything except the subject itself, so the trigger word learns the subject; in style mode they describe the content and never the style.'})),
        el('div', {class: 'card'}, el('div', {class: 'row'}, el('h2', {class: 'grow', style: 'margin:0', text: `Captions (${list.filter(i => i.caption).length} of ${list.length})`}),
            act('Caption missing', () => captionItems(list.filter(i => !i.caption), status)),
            act('Re-caption all (keeps your edits)', () => captionItems(list, status)),
            act('Check likeness', () => checkDrift(list.filter(i => i.id !== ds.reference), status), 'primary')),
            el('p', {class: 'muted', text: 'Captions and the likeness check use FreeVideo\'s prompt model. The check compares each picture with the reference and flags drifted ones (below ' + state.info.drift_limit + '/10); regenerate those on the Coverage tab or exclude them.'}),
            status, rows.length ? rows : el('div', {class: 'muted', text: 'No pictures yet.'})));
}
async function captionItems(list, status) {
    if (!list.length) { toast('Nothing to caption.'); return; }
    const q = await questions(), ds = state.ds;
    for (let start = 0; start < list.length; start += 8) {
        const batch = list.slice(start, start + 8);
        await enqueue(`Captioning ${start + 1}–${start + batch.length} of ${list.length}`, async () => {
            status.textContent = `Captioning ${start + 1}–${start + batch.length} of ${list.length}…`;
            const answers = await vlm.ask(batch.map(item => ({text: q.caption, images: [{file: item.input}]})), {system: q.caption_system, max_tokens: 160}, () => { status.textContent = 'Waiting for the prompt model…'; });
            for (const [index, item] of batch.entries()) if (answers[index]?.text) setDataset(await studio('caption', {dataset: ds.id, item: item.id, text: answers[index].text}));
        });
    }
    status.textContent = 'Captions done.';
    if (state.tab === 'captions') renderCaptions();
}
async function checkDrift(list, status) {
    const ref = reference();
    if (!ref) { toast('Choose a reference picture first.', true); return; }
    if (!list.length) { toast('Nothing to check.'); return; }
    const q = await questions(), ds = state.ds;
    for (let start = 0; start < list.length; start += 8) {
        const batch = list.slice(start, start + 8);
        await enqueue(`Checking likeness ${start + 1}–${start + batch.length} of ${list.length}`, async () => {
            status.textContent = `Checking likeness ${start + 1}–${start + batch.length} of ${list.length}…`;
            const answers = await vlm.ask(batch.map(item => ({text: q.drift, images: [{file: ref.input, label: 'reference'}, {file: item.input, label: 'candidate'}]})), {system: q.drift_system, format: 'json', max_tokens: 200});
            for (const [index, item] of batch.entries()) if (answers[index]) setDataset(await studio('drift', {dataset: ds.id, item: item.id, answer: answers[index]}));
        });
    }
    const drifted = state.ds.items.filter(i => !i.excluded && i.drift?.score != null && i.drift.score < state.info.drift_limit).length;
    status.textContent = drifted ? `${drifted} picture${drifted > 1 ? 's' : ''} drifted from the reference.` : 'All checked pictures match the reference.';
    if (state.tab === 'captions') renderCaptions();
}

// ── 4 · Train ──────────────────────────────────────────────────────────────────────────────────────────────────
async function renderTrain() {
    const ds = state.ds, host = $('#tab-train'), s = {...state.info.defaults, ...(ds.settings || {})};
    const field = (key, label, attrs = {}) => { const input = el('input', {type: 'number', value: s[key], step: 'any', ...attrs}); input.dataset.key = key; return el('label', {class: 'f'}, label, input); };
    const form = el('div', {class: 'settings-grid'},
        field('steps', 'Steps', {min: 100, max: 20000, step: 50}), field('rank', 'Rank (size)', {min: 4, max: 128, step: 4}),
        field('alpha', 'Alpha', {min: 1, max: 128}), field('learning_rate', 'Learning rate', {min: 0.000001, max: 0.01}),
        field('resolution', 'Resolution', {min: 256, max: 1536, step: 64}), field('save_every', 'Save every (steps)', {min: 25, max: 5000, step: 25}),
        field('repeats', 'Repeats (kohya/Musubi)', {min: 1, max: 100}), field('caption_dropout', 'Caption dropout', {min: 0, max: 0.5}),
        field('offload', 'Offload to RAM (0-1, blank = auto)', {min: 0, max: 1, step: 0.05, value: s.offload === 'auto' || s.offload == null ? '' : s.offload, placeholder: 'auto'}));
    const pictures = el('select', {}, [['auto', 'Auto: my pictures only when I gave 8 or more'], ['mine', 'My pictures only (and crops of them)'], ['all', 'My pictures and generated ones']]
        .map(([value, label]) => el('option', {value, text: label, selected: (s.pictures || 'auto') === value})));
    form.append(el('label', {class: 'f'}, 'Train on', pictures));
    const save = async () => {
        const settings = {pictures: pictures.value};
        for (const input of form.querySelectorAll('input')) settings[input.dataset.key] = input.dataset.key === 'offload' && input.value.trim() === '' ? 'auto' : Number(input.value);
        setDataset(await studio('update', {dataset: ds.id, settings}));
    };
    const planBox = el('div', {class: 'muted', text: 'Checking…'});
    const progressBox = el('div');
    const trainer = state.info.trainer;
    const folder = el('input', {type: 'text', value: trainer.folder, placeholder: 'C:\\Projects\\ai-toolkit', style: 'width:100%'});
    const python = el('input', {type: 'text', value: trainer.python, placeholder: '(found in the folder\'s venv)', style: 'width:100%'});
    const models = el('input', {type: 'text', value: trainer.models_dir, placeholder: 'Where the H3 training files go (about 40 GB), e.g. D:\\h3-train-models', style: 'width:100%'});
    host.replaceChildren(
        el('div', {class: 'card'}, el('h2', {text: 'Training settings'}), form,
            el('div', {class: 'row'}, act('Save settings', async () => { await save(); toast('Saved.'); renderTrain(); }),
                el('span', {class: 'muted', text: 'Defaults suit a character or object on a 16 GB card: 1500 steps, rank 16, 512 px, a checkpoint every 125 steps. The best one is often well before the end (a 768 px run peaked at step 1000 of 2500), so test a few checkpoints below and use the best.'}))),
        el('div', {class: 'card'}, el('h2', {text: 'Train on this PC (MiniMax H3 video LoRA)'}), planBox, progressBox),
        el('div', {class: 'card'}, el('h2', {text: 'Export instead'}),
            el('p', {class: 'muted', text: 'Pictures with matching .txt captions in the layout each trainer expects, with a ready config for ai-toolkit and Musubi.'}),
            el('div', {class: 'row'}, state.info.layouts.map(layout => el('div', {class: 'row', style: 'gap:4px'},
                act(`Save ${layout} folder`, async () => { await save(); const result = await studio('export', {dataset: ds.id, layout}); toast(`Saved to ${result.path}`); }),
                el('a', {class: 'btn small ghost', href: `/freevideo/lora-studio/api/zip?${new URLSearchParams({dataset: ds.id, layout, token: studioToken})}`, text: 'zip'}))))),
        el('div', {class: 'card'}, el('h2', {text: 'Trainer setup'}),
            el('p', {class: 'muted'}, 'LoRA Studio runs ', el('a', {href: 'https://github.com/ostris/ai-toolkit', target: '_blank', rel: 'noopener', text: 'ai-toolkit'}), ' in its own Python. Install it once with ', el('code', {text: 'scripts\\install-lora-trainer.ps1'}), ' in your FreeVideo folder, then accept the MiniMax H3 licence on Hugging Face (', el('a', {href: 'https://huggingface.co/Comfy-Org/MiniMax-H3', target: '_blank', rel: 'noopener', text: 'Comfy-Org/MiniMax-H3'}), ').'),
            el('label', {class: 'f'}, 'ai-toolkit folder', folder), el('label', {class: 'f', style: 'margin-top:8px'}, 'Its Python (optional)', python),
            el('label', {class: 'f', style: 'margin-top:8px'}, 'Models folder (MODELS_PATH, optional)', models),
            el('div', {class: 'row'}, act('Save trainer setup', async () => {
                const result = await studio('settings', {ai_toolkit_dir: folder.value.trim(), ai_toolkit_python: python.value.trim(), models_dir: models.value.trim()});
                state.info.trainer = result.trainer; toast(result.trainer.ready ? 'Trainer found.' : 'Saved, but run.py or its Python was not found there.', !result.trainer.ready); renderTrain();
            }), el('span', {class: 'pill ' + (trainer.ready ? 'good' : 'warn'), text: trainer.ready ? (trainer.h3 ? 'ai-toolkit ready, H3 support found' : 'ai-toolkit found; update it for H3 support') : 'Trainer not found'}))));
    try {
        const plan = await studio('train-plan', {dataset: ds.id});
        planBox.replaceChildren(
            el('div', {text: `${plan.pictures} pictures${plan.reg ? ` + ${plan.reg} regularisation` : ''}. ${plan.own_only && plan.generated_left_out ? `Training on your own pictures and crops of them; ${plan.generated_left_out} generated pictures are left out (change under Train on). ` : ''}Estimated ${plan.estimate_hours < 1.5 ? `${Math.round(plan.estimate_hours * 60)} minutes` : `${plan.estimate_hours} hours`}${plan.vram_gb ? ` on this ${Math.round(plan.vram_gb)} GB card` : ''}${plan.offload ? `, with ${Math.round(plan.offload * 100)}% of the model waiting in system memory` : ''} (first run also downloads about 40 GB of H3 files).`}),
            plan.warnings.length ? el('ul', {class: 'warns'}, plan.warnings.map(w => el('li', {text: w}))) : null,
            el('div', {class: 'row'},
                el('button', {class: 'btn primary', text: 'Start training', disabled: !plan.trainer.ready || plan.training.phase === 'running', onclick: async event => {
                    event.currentTarget.disabled = true;
                    try { await save(); await studio('train-start', {dataset: ds.id}); toast('Training started. FreeVideo can\'t make videos while it runs.'); pollTraining(progressBox); }
                    catch (error) { toast(message(error), true); event.currentTarget.disabled = false; }
                }}),
                el('details', {}, el('summary', {class: 'muted', text: 'Show the trainer config'}), el('pre', {class: 'log', text: plan.config}))));
        pollTraining(progressBox, true);
    } catch (error) { planBox.textContent = message(error); }
}
async function pollTraining(box, once = false) {
    clearTimeout(state.trainTimer);
    let status;
    try { status = await studio('train-status'); } catch { return; }
    if (!box.isConnected) return;
    const mine = status.dataset === state.ds?.id;
    if (status.phase === 'idle' || !mine) {
        box.replaceChildren(status.phase === 'running' ? el('p', {class: 'muted', text: 'Another LoRA set is training right now.'}) : '', await checkpointList());
        if (status.phase === 'running') state.trainTimer = setTimeout(() => pollTraining(box), 5000);
        return;
    }
    const fraction = status.total ? status.step / status.total : 0;
    const eta = status.eta_seconds ? ` · about ${Math.round(status.eta_seconds / 60)} min left` : '';
    box.replaceChildren(
        el('div', {class: 'row'}, el('b', {text: {running: 'Training…', complete: 'Training finished', failed: 'Training failed', cancelled: 'Training stopped'}[status.phase] || status.phase}),
            el('span', {class: 'muted', text: `step ${status.step || 0} of ${status.total}${status.loss != null ? ` · loss ${status.loss.toFixed(4)}` : ''}${eta}`}),
            status.phase === 'running' ? act('Stop', async () => { await studio('train-cancel'); }, 'danger') : null),
        el('div', {class: 'progress', style: 'margin:8px 0'}, el('div', {style: `width:${Math.round(fraction * 100)}%`})),
        status.message ? el('p', {class: 'warns', text: status.message}) : null,
        status.lora ? el('p', {}, 'Installed as ', el('code', {text: status.lora.name}), '. Link it to a cast card on the Use & test tab.') : null,
        el('details', {open: status.phase === 'failed'}, el('summary', {class: 'muted', text: 'Trainer log'}), el('pre', {class: 'log', text: status.tail || '(nothing yet)'})),
        await checkpointList());
    if (status.phase === 'running') state.trainTimer = setTimeout(() => pollTraining(box), 5000);
    else if (status.lora && state.ds?.lora?.name !== status.lora.name) setDataset(await studio('get', {dataset: state.ds.id}));
}
async function checkpointList() {
    const list = (await studio('checkpoints', {dataset: state.ds.id}).catch(() => ({checkpoints: []}))).checkpoints;
    if (!list.length) return '';
    return el('div', {style: 'margin-top:10px'}, el('div', {class: 'muted', text: 'Checkpoints from the last run (an earlier one is often more flexible):'}),
        el('div', {class: 'row'}, list.map(c => act(c.final ? 'Use final' : `Use step ${c.step}`, async () => {
            const result = await studio('promote', {dataset: state.ds.id, checkpoint: c.name}); setDataset(result.dataset); toast(`Now using ${result.lora.name}`);
        }, state.ds.lora?.step === c.step || (c.final && state.ds.lora && state.ds.lora.step == null) ? 'primary' : ''))));
}

// ── 5 · Use & test ─────────────────────────────────────────────────────────────────────────────────────────────
async function renderUse() {
    const ds = state.ds, host = $('#tab-use'), lora = ds.lora;
    const strength = el('input', {type: 'number', min: 0, max: 2, step: 0.05, value: lora?.strength ?? 1});
    const cardBox = el('div', {class: 'muted', text: ds.kind === 'style' ? 'A style LoRA has no cast card: pick it in the creation panel\'s LoRA list; its trigger word is added for you.' : 'Loading cast cards…'});
    const otherLora = el('input', {type: 'text', placeholder: 'lora-studio/name.safetensors', size: 34});
    const testPrompt = el('textarea', {rows: 2}, ds.settings?.test_prompt || '');
    const strengths = el('input', {type: 'text', value: (ds.settings?.test_strengths || [0.6, 0.8, 1, 1.2]).join(', '), size: 18});
    const seconds = el('input', {type: 'number', min: 1.625, max: 10, step: 0.5, value: ds.settings?.test_seconds || 3});
    const seed = el('input', {type: 'number', value: 1234});
    const portrait = el('input', {type: 'checkbox'});
    const grid = el('div', {class: 'tests'});
    host.replaceChildren(
        el('div', {class: 'card'}, el('h2', {text: 'Your LoRA'}),
            lora ? el('div', {class: 'row'}, el('code', {text: lora.name}), lora.step != null ? el('span', {class: 'muted', text: `(step ${lora.step})`}) : null,
                el('label', {class: 'f'}, 'Strength when loaded by @Name', strength),
                act('Save strength', async () => { setDataset(await studio('link', {dataset: ds.id, cast_id: lora.cast_id || null, strength: Number(strength.value)})); toast('Saved.'); }))
                : el('p', {class: 'muted', text: 'No LoRA yet: train one on the Train tab, or use one you trained elsewhere:'}),
            el('div', {class: 'row'}, otherLora, act('Use this LoRA file', async () => { setDataset(await studio('use-lora', {dataset: ds.id, name: otherLora.value.trim()})); renderUse(); }))),
        el('div', {class: 'card'}, el('h2', {text: 'Cast card: @Name loads the LoRA'}), cardBox),
        el('div', {class: 'card'}, el('h2', {text: 'Strength test grid'}),
            el('p', {class: 'muted', text: 'Short test videos with the same prompt and seed at several strengths, so you can see where the likeness holds and where it starts to stiffen. Each takes a few minutes.'}),
            el('label', {class: 'f'}, 'Prompt (leave empty for a default that uses your trigger word)', testPrompt),
            el('div', {class: 'row'}, el('label', {class: 'f'}, 'Strengths', strengths), el('label', {class: 'f'}, 'Seconds', seconds), el('label', {class: 'f'}, 'Seed', seed),
                ds.kind === 'character' ? el('label', {}, portrait, ' Portrait') : null,
                el('button', {class: 'btn primary', text: 'Make test grid', disabled: !lora, onclick: () => runTests()})),
            grid));
    const values = () => strengths.value.split(/[ ,;]+/).map(Number).filter(v => Number.isFinite(v) && v >= 0 && v <= 2).slice(0, 6);
    async function runTests() {
        const list = values();
        if (!list.length) { toast('Enter strengths between 0 and 2.', true); return; }
        setDataset(await studio('update', {dataset: ds.id, settings: {test_prompt: testPrompt.value.trim(), test_strengths: list, test_seconds: Number(seconds.value)}}));
        const test = {id: '', prompt: testPrompt.value.trim(), seed: Number(seed.value), lora: lora.name, cells: []};
        grid.replaceChildren();
        const cells = [];
        for (const value of [0, ...list.filter(v => v !== 0)]) {
            const holder = el('div', {class: 'card', style: 'margin:0'}, el('b', {text: value === 0 ? 'Without LoRA' : `Strength ${value}`}), el('div', {class: 'muted', text: 'Queued…'}));
            grid.append(holder); cells.push({value, holder});
        }
        for (const cell of cells) {
            try {
                const graph = await studio('test-graph', {dataset: ds.id, strength: cell.value, seed: Number(seed.value), seconds: Number(seconds.value), prompt: testPrompt.value.trim(), portrait: portrait.checked});
                cell.id = await comfy.queue(graph.prompt); test.prompt = graph.text;
            } catch (error) { cell.holder.lastChild.textContent = message(error); }
        }
        for (;;) {
            let pending = 0;
            for (const cell of cells.filter(c => c.id && !c.done)) {
                const result = await comfy.result(cell.id).catch(() => null);
                if (!result?.done) { pending++; cell.holder.lastChild.textContent = 'Making…'; continue; }
                cell.done = true;
                if (result.video) {
                    const src = '/view?' + new URLSearchParams({filename: result.video.filename, subfolder: result.video.subfolder || '', type: result.video.type || 'output'});
                    cell.holder.lastChild.replaceWith(el('video', {src, controls: true, loop: true, muted: true, autoplay: true, playsInline: true}));
                    test.cells.push({strength: cell.value, video: result.video});
                } else cell.holder.lastChild.textContent = result.message || 'No video came back.';
            }
            if (!pending) break;
            await sleep(4000);
        }
        if (test.cells.length) setDataset(await studio('test-record', {dataset: ds.id, test}));
    }
    // Earlier test grids
    for (const test of (ds.tests || []).slice().reverse().slice(0, 1)) {
        for (const cell of test.cells || []) {
            const src = '/view?' + new URLSearchParams({filename: cell.video.filename, subfolder: cell.video.subfolder || '', type: cell.video.type || 'output'});
            grid.append(el('div', {class: 'card', style: 'margin:0'}, el('b', {text: cell.strength === 0 ? 'Without LoRA' : `Strength ${cell.strength}`}), el('video', {src, controls: true, loop: true, muted: true, playsInline: true})));
        }
    }
    if (ds.kind !== 'style') renderCast(cardBox);
}
async function renderCast(box) {
    const ds = state.ds, kind = ds.kind === 'object' ? 'prop' : 'character', lora = ds.lora;
    let cards;
    try { cards = await agent.cards(kind); } catch (error) { box.textContent = message(error); return; }
    const linked = cards.find(card => card.id === lora?.cast_id);
    const pick = el('select', {}, el('option', {value: '', text: '(not linked)'}), cards.map(card => el('option', {value: card.id, text: card.name, selected: card.id === lora?.cast_id})));
    box.replaceChildren(
        el('p', {class: 'muted', text: `When a ${kind === 'prop' ? 'prop' : 'character'} card linked here is mentioned with @ in the creation panel, its LoRA is added at the strength above (and taken off when the mention goes).`}),
        el('div', {class: 'row'}, el('label', {class: 'f'}, 'Linked card', pick),
            act('Link', async () => { setDataset(await studio('link', {dataset: ds.id, cast_id: pick.value || null})); toast(pick.value ? 'Linked.' : 'Unlinked.'); renderUse(); }),
            act(`Make a new ${kind === 'prop' ? 'prop' : 'character'} card from this set`, async () => {
                if (!lora) throw new Failure('no_lora');
                const ref = reference();
                const pictures = [ref, ...items().filter(i => !i.excluded && i.id !== ref?.id && ['face_front', 'three_quarter_left', 'profile_left', 'back', 'front', 'three_quarter', 'left'].includes(i.slot))].filter(Boolean).slice(0, 4);
                const labels = {front: 'front', face_front: 'face', three_quarter_left: 'three_quarter', three_quarter: 'three_quarter', profile_left: 'side', left: 'side', back: 'back'};
                const card = await agent.create({kind, name: ds.name, description: ds.description || null,
                    images: pictures.map((p, i) => ({media: p.input, label: i === 0 ? 'front' : labels[p.slot] || 'other'}))});
                setDataset(await studio('link', {dataset: ds.id, cast_id: card.id}));
                toast(`Made the card "${card.name}" and linked it. Write @${card.name} in a prompt.`); renderUse();
            }, 'primary')),
        linked ? el('p', {}, 'Linked to ', el('b', {text: linked.name}), `. Write @${linked.name} in the creation panel's prompt.`) : null);
}

// ── New / edit / delete ────────────────────────────────────────────────────────────────────────────────────────
function datasetDialog(existing = null) {
    const dialog = $('#dialog'); dialog.className = '';
    const name = el('input', {type: 'text', value: existing?.name || '', placeholder: 'e.g. Maya, Red mug, Ink wash', style: 'width:100%'});
    const trigger = el('input', {type: 'text', value: existing?.trigger || '', placeholder: '(made from the name)'});
    const cls = el('input', {type: 'text', value: existing?.class_word || '', placeholder: 'woman, man, cat, mug…'});
    const description = el('textarea', {rows: 3, placeholder: 'What it looks like (optional when you have photos)'}, existing?.description || '');
    const kinds = el('div', {class: 'kind-pick'}, [['character', 'Character', 'A person or creature: angles, expressions, poses.'], ['object', 'Object', 'A product, prop or vehicle: sides, details, in use.'], ['style', 'Style', 'A look: medium, palette, rendering.']]
        .map(([value, title, text]) => el('label', {}, el('input', {type: 'radio', name: 'kind', value, checked: (existing?.kind || 'character') === value, disabled: Boolean(existing)}), el('b', {text: title}), text)));
    const save = el('button', {class: 'btn primary', text: existing ? 'Save' : 'Create'});
    dialog.replaceChildren(el('h2', {style: 'margin-top:0', text: existing ? 'LoRA details' : 'New LoRA'}), kinds,
        el('label', {class: 'f', style: 'margin-top:12px'}, 'Name', name),
        el('div', {class: 'row'}, el('label', {class: 'f'}, 'Trigger word', trigger), el('label', {class: 'f'}, 'Class word', cls)),
        el('label', {class: 'f', style: 'margin-top:10px'}, 'Description', description),
        el('div', {class: 'row', style: 'justify-content:flex-end;margin-top:14px'}, el('button', {class: 'btn ghost', text: 'Cancel', onclick: () => dialog.close()}), save));
    save.addEventListener('click', async () => {
        save.disabled = true;
        try {
            const body = {name: name.value.trim(), trigger: trigger.value.trim() || undefined, class_word: cls.value.trim() || undefined, description: description.value.trim()};
            const result = existing ? await studio('update', {dataset: existing.id, ...body}) : await studio('create', {...body, kind: dialog.querySelector('input[name=kind]:checked').value});
            dialog.close(); await openDataset(result.id);
        } catch (error) { toast(message(error), true); save.disabled = false; }
    });
    dialog.showModal(); name.focus();
}

// ── Start ──────────────────────────────────────────────────────────────────────────────────────────────────────
async function start() {
    try { state.info = await studioInfo(); } catch (error) { document.body.prepend(el('div', {class: 'card', style: 'margin:20px', text: 'LoRA Studio could not reach FreeVideo: ' + message(error)})); return; }
    imageGen.load().catch(() => {});
    $('#new-btn').addEventListener('click', () => datasetDialog());
    $('#welcome-new').addEventListener('click', () => datasetDialog());
    $('#ds-edit').addEventListener('click', () => datasetDialog(state.ds));
    $('#ds-delete').addEventListener('click', async () => {
        if (!confirm(`Delete "${state.ds.name}" and all its pictures? A LoRA already installed stays in your LoRA folder.`)) return;
        await studio('delete', {dataset: state.ds.id}); state.ds = null; $('#editor').hidden = true; $('#welcome').hidden = false; refreshList();
    });
    for (const button of document.querySelectorAll('#tabs button')) button.addEventListener('click', () => { state.tab = button.dataset.tab; clearTimeout(state.trainTimer); renderTab(); });
    await refreshList();
    let last = null; try { last = localStorage.getItem('loraStudio.last'); } catch {}
    const wanted = new URLSearchParams(location.search).get('dataset') || last;
    if (wanted && state.datasets.some(row => row.id === wanted)) openDataset(wanted);
    if (state.info.training?.phase === 'running') { const note = $('#gpu-note'); note.hidden = false; note.className = 'pill warn'; note.textContent = 'A LoRA is training: the GPU is busy'; }
}
start();
