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
.fv-report-card .fv-rc-take{display:flex;gap:12px;align-items:center;margin:8px 0}.fv-report-card .fv-rc-take img{width:54px;height:96px;object-fit:cover;border-radius:6px}
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
            : fresh ? t('Checking faces, pops, cuts, motion and speech…', '正在检查…') : t('Loading…', '加载中…')));
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

// Best take first among several (plan step 1.2): each take's Report Card is made where missing, then ranked.
export function openRanking(ids, t = (en) => en) {
    if (!styled) { const s = el('style', css); document.head.append(s); styled = true; }
    const dialog = el('dialog', null, 'fv-report-card');
    const body = el('div');
    const close = el('button', t('Close', '关闭'));
    close.onclick = () => dialog.close();
    dialog.addEventListener('close', () => dialog.remove());
    dialog.append(el('h2', t('Best take', '最佳版本')), body, el('div', null, 'fv-rc-actions'));
    dialog.lastElementChild.append(close);
    document.body.append(dialog);
    dialog.showModal();
    body.append(el('p', t('Checking {count} takes…', '正在检查 {count} 个版本…', { count: ids.length })));
    (async () => {
        try {
            const response = await api.fetchApi('/freevideo/report-card/rank', {method: 'POST',
                headers: {'Content-Type': 'application/json', 'X-FreeVideo-Library': '1'}, body: JSON.stringify({ids})});
            const cards = await response.json().catch(() => null);
            if (!response.ok) throw new Error(cards?.error || 'The check could not run.');
            body.replaceChildren();
            cards.forEach((card, place) => {
                const row = el('div', null, 'fv-rc-take');
                const image = el('img'); image.src = api.apiURL('/freevideo/library/thumbnail?' + new URLSearchParams({id: card.id})); image.alt = '';
                const words = el('div');
                words.append(el('div', (place === 0 ? t('Best', '最佳') + ' · ' : '#' + (place + 1) + ' · ') + (card.passed ? t('Passed', '通过') : t('Needs another look', '有问题')),
                    card.passed ? 'fv-rc-pass' : 'fv-rc-fail'));
                for (const face of card.faces) words.append(el('div', face.text, 'fv-rc-note'));
                const problems = [...card.flags, ...(card.look?.flags || [])].length;
                if (problems) words.append(el('div', t('{count} flags', '{count} 个问题', { count: problems }), 'fv-rc-note'));
                const open = el('button', t('Report card', '成绩单')); open.onclick = () => openReportCard(card.id, t);
                row.append(image, words, open);
                body.append(row);
            });
        } catch (error) { body.replaceChildren(el('p', error.message, 'fv-rc-fail')); }
    })();
}

// Film continuity (plan step 1.3): colour shot to shot, and on request set and wardrobe at every cut.
export function openContinuity(ids, t = (en) => en) {
    if (!styled) { const s = el('style', css); document.head.append(s); styled = true; }
    const dialog = el('dialog', null, 'fv-report-card');
    const body = el('div');
    const lookButton = el('button', t('Ask the picture reader', '让看图模型检查'));
    const close = el('button', t('Close', '关闭'));
    close.onclick = () => dialog.close();
    dialog.addEventListener('close', () => dialog.remove());
    dialog.append(el('h2', t('Continuity', '连贯性')), body, el('div', null, 'fv-rc-actions'));
    dialog.lastElementChild.append(lookButton, close);
    document.body.append(dialog);
    dialog.showModal();
    const run = async (look) => {
        lookButton.disabled = true;
        body.replaceChildren(el('p', look ? t('Comparing every cut…', '正在比较每个剪接点…') : t('Comparing colour across {count} shots…', '正在比较 {count} 个镜头的颜色…', { count: ids.length })));
        try {
            const response = await api.fetchApi('/freevideo/report-card/continuity', {method: 'POST',
                headers: {'Content-Type': 'application/json', 'X-FreeVideo-Library': '1'}, body: JSON.stringify({ids, look})});
            const result = await response.json().catch(() => null);
            if (!response.ok) throw new Error(result?.error || 'The check could not run.');
            body.replaceChildren(el('p', result.passed ? t('The shots match.', '镜头之间一致。') : t('Some shots do not match.', '有镜头不一致。'),
                'fv-rc-verdict ' + (result.passed ? 'fv-rc-pass' : 'fv-rc-fail')));
            const list = el('ul');
            for (const flag of result.flags) list.append(el('li', flag.text));
            if (list.children.length) body.append(list);
            for (const note of result.notes) body.append(el('p', note, 'fv-rc-note'));
        } catch (error) { body.replaceChildren(el('p', error.message, 'fv-rc-fail')); }
        finally { lookButton.disabled = false; }
    };
    lookButton.onclick = () => run(true);
    run(false);
}
