// The three-column shell shared by the picture view and the creation panel: a left panel (settings), the stage in the
// middle, a right panel (works). Both panels fold to a rail; under 1280 wide they rest as rails and open as drawers over
// the stage, one at a time; on a tall window (portrait, wider than a phone) the settings open as a drawer over the works
// and the works open below them. Wide, they dock and fold, and what is chosen is kept.
//
// createPanelShell({body, left, right, stage, panels, save, onPaint, onOpenLeft, resizable}) → {setPanel, paint, closeDrawer, drawer, below, tall, narrow, tallQuery}
//   resizable: docked panels get a drag handle on their inner edge; widths are kept in panels.leftWidth/rightWidth.
//   body, left, right, stage: the elements (the panels carry data-collapsed, body carries data-left/right/narrow/tall/drawer/below)
//   panels: {left, right} wide-window choices (false = folded); save(): writes them; onPaint(): after every repaint;
//   onOpenLeft(): after the settings were opened.
export function createPanelShell({body, left, right, stage, panels, save = () => {}, onPaint = () => {}, onOpenLeft = () => {}, resizable = false}) {
    // Under 1280 wide both panels rest as rails and open as drawers over the stage, one at a time, so the picture
    // keeps the room; the stage still measures from the two rails, so it stays centred.
    const narrowQuery = matchMedia('(min-width: 720px) and (max-width: 1279px)');
    // A tall window: the settings open as a drawer over the works, and the works panel opens below them, taking the
    // height they leave (at least 320 px), so it never narrows them from the side.
    const tallQuery = matchMedia('(orientation: portrait) and (min-width: 720px)');
    let drawer = '', below = false;
    function paint() {
        if (tallQuery.matches) {
            if (drawer === 'right') drawer = '';
            left.dataset.collapsed = String(drawer !== 'left'); right.dataset.collapsed = String(!below);
            body.dataset.tall = 'true'; body.dataset.below = String(below); body.dataset.narrow = 'true'; body.dataset.drawer = drawer;
            body.dataset.left = 'rail'; body.dataset.right = 'rail';
            onPaint();
            return;
        }
        delete body.dataset.tall; delete body.dataset.below;
        const narrow = narrowQuery.matches;
        const leftShown = narrow ? drawer === 'left' : panels.left !== false, rightShown = narrow ? drawer === 'right' : panels.right !== false;
        left.dataset.collapsed = String(!leftShown); right.dataset.collapsed = String(!rightShown);
        body.dataset.narrow = String(narrow); body.dataset.drawer = narrow ? drawer : '';
        body.dataset.left = leftShown && !narrow ? 'open' : 'rail'; body.dataset.right = rightShown && !narrow ? 'open' : 'rail';
        onPaint();
    }
    function setPanel(side, open) {
        if (tallQuery.matches && side === 'right') below = open;
        else if (tallQuery.matches || narrowQuery.matches) drawer = open ? side : drawer === side ? '' : drawer;
        else { panels[side] = open; save(); }
        paint(); if (open && side === 'left') onOpenLeft();
    }
    const closeDrawer = () => { if (drawer) { drawer = ''; paint(); } };
    narrowQuery.addEventListener('change', () => { drawer = ''; paint(); onOpenLeft(); });
    tallQuery.addEventListener('change', () => { drawer = ''; below = false; paint(); onOpenLeft(); });
    // A drawer closes when the stage is touched; on a tall window, when anything outside it is.
    stage.addEventListener('pointerdown', closeDrawer);
    body.addEventListener('pointerdown', event => { if (tallQuery.matches && drawer === 'left' && !left.contains(event.target)) closeDrawer(); });
    if (resizable) resizeHandles({body, panels, save, paint});
    return {setPanel, paint, closeDrawer, drawer: () => drawer, below: () => below, tall: () => tallQuery.matches, narrow: () => narrowQuery.matches, tallQuery};
}

// Drag handles on the inner edges of the docked panels. A width is kept per side; double-click (or Home) goes back to
// the default; the arrow keys move the edge by 16 px. Shown only where the CSS docks the panels (wide windows).
const MIN_WIDTH = 240, MAX_WIDTH = 720;
function resizeHandles({body, panels, save, paint}) {
    const clamp = width => Math.round(Math.min(Math.max(width, MIN_WIDTH), Math.min(MAX_WIDTH, body.getBoundingClientRect().width * .4)));
    const apply = side => {
        const width = panels[side + 'Width'];
        if (Number.isFinite(width)) body.style.setProperty(`--fv-col-${side[0]}`, `${width}px`);
        else body.style.removeProperty(`--fv-col-${side[0]}`);
    };
    for (const side of ['left', 'right']) {
        apply(side);
        const handle = document.createElement('div');
        handle.className = 'fv-ps-resize'; handle.dataset.side = side; handle.tabIndex = 0;
        handle.setAttribute('role', 'separator'); handle.setAttribute('aria-orientation', 'vertical');
        handle.setAttribute('aria-label', side === 'left' ? 'Resize the settings column' : 'Resize the creations column');
        handle.title = 'Drag to resize; double-click to reset';
        const set = (width, keep = true) => {
            if (width === null) delete panels[side + 'Width']; else panels[side + 'Width'] = clamp(width);
            apply(side); if (keep) save(); paint();
        };
        // The column's width as drawn: the handle sits centred on its inner edge.
        const current = () => {
            const rect = body.getBoundingClientRect(), edge = handle.getBoundingClientRect();
            const middle = edge.left + edge.width / 2;
            return side === 'left' ? middle - rect.left : rect.right - middle;
        };
        handle.addEventListener('pointerdown', event => {
            if (event.button !== 0) return;
            event.preventDefault(); handle.setPointerCapture(event.pointerId); body.dataset.resizing = side;
            const move = moved => {
                const rect = body.getBoundingClientRect();
                set(side === 'left' ? moved.clientX - rect.left : rect.right - moved.clientX, false);
            };
            const end = () => {
                handle.removeEventListener('pointermove', move); delete body.dataset.resizing; save();
            };
            handle.addEventListener('pointermove', move);
            handle.addEventListener('pointerup', end, {once: true}); handle.addEventListener('pointercancel', end, {once: true});
        });
        handle.addEventListener('dblclick', () => set(null));
        handle.addEventListener('keydown', event => {
            const step = event.shiftKey ? 64 : 16, grow = side === 'left' ? 'ArrowRight' : 'ArrowLeft', shrink = side === 'left' ? 'ArrowLeft' : 'ArrowRight';
            if (event.key === grow) set(current() + step);
            else if (event.key === shrink) set(current() - step);
            else if (event.key === 'Home') set(null);
            else return;
            event.preventDefault();
        });
        body.append(handle);
    }
}
