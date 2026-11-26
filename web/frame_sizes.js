// Shared creation/editor aspect names and the creation panel's 32-pixel sizing.
export const frameShapes = t => [
    ['16:9', t('Landscape', '横屏'), 16 / 9],
    ['9:16', t('Portrait', '竖屏'), 9 / 16],
    ['1:1', t('Square', '方形'), 1],
];
// The further shapes Create offers; Edit's frame-size menu offers the same list.
export const allFrameShapes = t => [...frameShapes(t), ...[[4, 3], [3, 4], [3, 2], [2, 3], [21, 9]].map(([a, b]) => [`${a}:${b}`, `${a}:${b}`, a / b])];
export function frameSize(pixels, ratio) {
    const total = pixels * 1024 ** 2;
    return {width: Math.round(Math.sqrt(total * ratio) / 32) * 32,
        height: Math.round(Math.sqrt(total / ratio) / 32) * 32};
}
