// Take Report Card (fork addition): what the automatic checks found on one saved take, and a button to check again.
import { api } from '../../scripts/api.js';

const el = (tag, text, cls) => { const e = document.createElement(tag); if (text != null) e.textContent = text; if (cls) e.className = cls; return e; };
const css = `
.fv-report-card{max-width:560px;width:calc(100vw - 32px);border:1px solid var(--border-color,#444);border-radius:12px;padding:20px;
  background:var(--comfy-menu-bg,#1e1e1e);color:var(--fg-color,#eee);font:14px/1.45 system-ui,sans-serif}
.fv-report-card::backdrop{background:rgba(0,0,0,.55)}
.fv-report-card h2{margin:0 0 4px;font-size:18px}
.fv-report-card .fv-rc-verdict{font-weight:600;margin:0 0 12px}
.fv-report-card .fv-rc-pass{color:#5cc98a}.fv-report-card .fv-rc-fail{color:#f0a04b}
.fv-report-card ul{margin:0 0 12px;padding-left:20px}.fv-report-card li{margin:4px 0}
.fv-report-card .fv-rc-note{opacity:.7;font-size:13px}
.fv-report-card .fv-rc-actions{display:flex;gap:8px;justify-content:flex-end;margin-top:8px}
.fv-report-card button{padding:6px 14px;border-radius:8px;border:1px solid var(--border-color,#555);background:transparent;color:inherit;cursor:pointer}`;
let styled = false;

async function fetchCard(id, fresh, look = false) {
    const post = {method: 'POST', headers: {'Content-Type': 'application/json', 'X-FreeVideo-Library': '1'}, body: JSON.stringify({id})};
    const response = look ? await api.fetchApi('/freevideo/report-card/look', post)
        : fresh ? await api.fetchApi('/freevideo/report-card', post)
        : await api.fetchApi('/freevideo/report-card?' + new URLSearchParams({id}));
    const body = await response.json().catch(() => null);
    if (!response.ok) throw new Error(body?.error || 'The check could not run.');
    return body;
}

export function openReportCard(id, t = (en) => en) {
    if (!styled) { const s = el('style', css); document.head.append(s); styled = true; }
    const dialog = el('dialog', null, 'fv-report-card');
    const body = el('div');
    const again = el('button', t('Check again', '重新检查'));
    // Wardrobe, set, hands and people: the picture reader uses the graphics card, so it runs only when asked.
    const lookButton = el('button', t('Ask the picture reader', '让看图模型检查'));
    const close = el('button', t('Close', '关闭'));
    close.onclick = () => dialog.close();
    dialog.addEventListener('close', () => dialog.remove());
    dialog.append(el('h2', t('Report card', '成绩单')), body, el('div', null, 'fv-rc-actions'));
    dialog.lastElementChild.append(lookButton, again, close);
    document.body.append(dialog);
    dialog.showModal();

    const show = (card) => {
        body.replaceChildren();
        if (!card) { body.append(el('p', t('Not checked yet.', '还没有检查。'))); return; }
        body.append(el('p', card.passed ? t('Passed every check.', '全部通过。') : t('Needs another look.', '有问题需要处理。'),
            'fv-rc-verdict ' + (card.passed ? 'fv-rc-pass' : 'fv-rc-fail')));
        const list = el('ul');
        for (const row of [...card.faces, ...card.flags, ...(card.look?.flags || [])]) list.append(el('li', row.text));
        if (list.children.length) body.append(list);
        for (const note of [...(card.notes || []), ...(card.look?.notes || [])]) body.append(el('p', note, 'fv-rc-note'));
        if (!card.look) body.append(el('p', t('Wardrobe, set, hands and people not checked yet: use "Ask the picture reader".', '服装、场景、手和人数尚未检查。'), 'fv-rc-note'));
    };
    const run = async (fresh, look = false) => {
        again.disabled = lookButton.disabled = true;
        body.replaceChildren(el('p', look ? t('Asking the picture reader about four frames…', '正在让看图模型检查…')
            : fresh ? t('Checking faces, pops, cuts and motion…', '正在检查…') : t('Loading…', '加载中…')));
        try {
            const card = await fetchCard(id, fresh, look);
            if (!card && !fresh) return run(true);   // never checked: check it now
            show(card);
        } catch (error) { body.replaceChildren(el('p', error.message, 'fv-rc-fail')); }
        finally { again.disabled = lookButton.disabled = false; }
    };
    again.onclick = () => run(true);
    lookButton.onclick = () => run(false, true);
    run(false);
}
