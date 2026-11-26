// Fork addition: a "LoRA" entry in the top bar that opens LoRA Studio (freevideo_engine/lora_studio) in a new tab.
// The top bar is rebuilt by topbar.js, so the entry is added to every bar as it appears instead of editing that file.
const browser = typeof document !== 'undefined' && typeof MutationObserver !== 'undefined';
const ICON = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.4 6.8 19.1l1-5.8L3.5 9.2l5.9-.9z"/></svg>';

function zh() { try { return (localStorage.getItem('freevideo.language') || navigator.language || '').toLowerCase().startsWith('zh'); } catch { return false; } }

function entry() {
    const link = document.createElement('a');
    link.className = 'fv-tb-link fv-lora-studio-entry';
    link.href = '/freevideo/lora-studio/';
    link.target = '_blank'; link.rel = 'noopener';
    const label = zh() ? 'LoRA 工作室' : 'LoRA Studio';
    link.setAttribute('aria-label', label);
    link.title = zh() ? '制作人物、物体或风格 LoRA' : 'Make a character, object or style LoRA';
    link.style.display = 'inline-flex'; link.style.alignItems = 'center'; link.style.gap = '6px'; link.style.textDecoration = 'none';
    link.innerHTML = ICON;
    const text = document.createElement('span'); text.className = 'fv-tb-lbl'; text.textContent = 'LoRA';
    link.append(text);
    return link;
}

function place(root = document) {
    for (const links of root.querySelectorAll?.('.fv-tb-links') || []) {
        if (links.querySelector('.fv-lora-studio-entry')) continue;
        const settings = links.querySelector('.fv-settings-entry');
        links.insertBefore(entry(), settings || null);
    }
}

if (browser) {
    const start = () => {
        place();
        new MutationObserver(records => {
            for (const record of records) for (const node of record.addedNodes) if (node.nodeType === 1 && (node.matches?.('.fv-tb-links') || node.querySelector?.('.fv-tb-links'))) { place(); return; }
        }).observe(document.body, {childList: true, subtree: true});
    };
    if (document.body) start(); else document.addEventListener('DOMContentLoaded', start, {once: true});
}

export {};
