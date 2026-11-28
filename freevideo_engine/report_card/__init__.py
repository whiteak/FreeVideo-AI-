"""Take Report Card (fork addition, plan step 1.1): checks a saved take and writes video.report-card.json beside it.

Checks, all on the CPU and none of them downloading anything:
- faces: every cast character in the take scored once a second against its card pictures (insightface buffalo_l,
  run by faces.py in its own process), with a pass bar per card (card "face_bar", default 0.72);
- frame-0 pop: the first frames jump far more than the rest of the take;
- flicker: brightness pumps up and down for a frame or two;
- cuts: a hard jump to another picture in the middle of a take (a restart);
- back to the start: the picture drifts away from the opening frame and then returns to it;
- walking back: a character moves one way and then comes back (from the face boxes).
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_BAR = 0.72
MIN_FACE_PX = 80          # Smaller faces are listed but not scored, as in the promo checks.
SAME_PERSON = 0.3         # Under this a face is someone else (another person, or a face with no card).
SMALL_WIDTH = 64          # Frames are compared at this width.
FILE = 'video.report-card.json'   # video.* is a name the library owns, so deleting the take still works
FACE_TIMEOUT = 600
FACE_PYTHON_FALLBACK = Path(r'C:\LocalAI\facecheck\Scripts\python.exe')
CARD_LABELS = ('face', 'front', 'other')   # The card pictures that show the face straight on.


class Unavailable(RuntimeError):
    """A check could not run (no face checker, unreadable video); the message says why in plain words."""


# --- Frame checks -------------------------------------------------------------------------------------------------

def small_frames(video):
    """(fps, frames as an N x h x w float32 grey array at SMALL_WIDTH wide)."""
    import av
    import numpy as np
    rows = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 24)
        height = None
        for frame in container.decode(stream):
            if height is None:
                height = max(2, round(frame.height * SMALL_WIDTH / frame.width))
            rows.append(frame.reformat(width=SMALL_WIDTH, height=height, format='gray').to_ndarray().astype(np.float32))
    if len(rows) < 4:
        raise Unavailable('The video has fewer than four frames.')
    return fps, np.stack(rows)


def frame_flags(frames, fps):
    """Flags from the small grey frames: frame-0 pop, flicker, cuts and a return to the opening picture."""
    import numpy as np
    flags = []
    n = len(frames)
    diff = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))          # diff[i]: frame i -> i + 1
    base = float(np.median(diff)) + 0.5
    head = diff[:3]
    if head.max() > max(3 * base, base + 4):
        k = int(head.argmax())
        jump = head.max() / base
        text = ('Opening frame dropped: after frame %d the take leaves the opening picture (%.0fx jump).' if jump > 12 else
                'Frame-0 pop: frame %d jumps %.0fx more than the rest of the take.') % (k if jump > 12 else k + 1, jump)
        flags.append(dict(check='pop', frame=k + 1, t=round((k + 1) / fps, 2),
                          text=text + ' Trim the first %d frames.' % (k + 2)))
    if diff[-1] > max(3 * base, base + 4):
        flags.append(dict(check='end_pop', frame=n - 1, t=round((n - 1) / fps, 2),
                          text='Last-frame pop: the final frame jumps %.0fx more than the rest. Trim it.' % (diff[-1] / base)))
    cut_at = [i + 1 for i in range(3, n - 2) if diff[i] > max(8 * base, 18)]
    for frame in cut_at:
        flags.append(dict(check='cut', frame=frame, t=round(frame / fps, 2),
                          text='Hard cut at %.1f s: the picture jumps to another shot (a restart).' % (frame / fps)))
    luma = frames.mean(axis=(1, 2))
    if n >= 5:
        smooth = np.array([np.median(luma[max(0, i - 2):i + 3]) for i in range(n)])
        quiet = set(cut_at) | {c - 1 for c in cut_at} | {0, 1, 2, n - 1}   # pops and cuts are flagged on their own
        bumps = [i for i in range(n) if abs(luma[i] - smooth[i]) > 3 and i not in quiet]
        if bumps:
            flags.append(dict(check='flicker', frames=bumps[:20], t=round(bumps[0] / fps, 2),
                              text='Flicker: brightness pumps on %d frame%s, first at %.1f s.'
                                   % (len(bumps), '' if len(bumps) == 1 else 's', bumps[0] / fps)))
    back = return_to_start(frames)
    if back is not None and not any(abs(back - c) <= 2 for c in cut_at):
        flags.append(dict(check='back_to_start', frame=back, t=round(back / fps, 2),
                          text='Back to the start: at %.1f s the picture returns to the opening shot.' % (back / fps)))
    return flags


def return_to_start(frames, away=0.55, near=0.85):
    """The first frame that looks like the opening picture again after the take has moved well away from it."""
    import numpy as np
    flat = frames.reshape(len(frames), -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-6
    similarity = flat @ flat[0]
    gone = False
    for index, value in enumerate(similarity):
        gone = gone or value < away
        if gone and value > near:
            return index
    return None


# --- Faces --------------------------------------------------------------------------------------------------------

def face_python():
    """A Python that can import insightface: Eclipse's own, FREEVIDEO_FACE_PYTHON, or the facecheck env."""
    try:
        import importlib.util
        if importlib.util.find_spec('insightface') and importlib.util.find_spec('cv2'):
            return sys.executable
    except (ImportError, ValueError):
        pass
    for value in (os.environ.get('FREEVIDEO_FACE_PYTHON'), FACE_PYTHON_FALLBACK):
        if value and Path(value).is_file():
            return str(value)
    return None


def cast_cards(output_root, input_root, folder, wanted=()):
    """The character cards a take used: ids marked on its references, and @Name in its prompt."""
    store = Path(output_root) / 'FreeVideo' / '.agent' / 'jobs' / 'cast'
    cards = {}
    for path in store.glob('*.json'):
        try:
            card = json.loads(path.read_text(encoding='utf-8')).get('card') or {}
        except (OSError, ValueError, AttributeError):
            continue
        if card.get('kind') == 'character' and isinstance(card.get('name'), str):
            cards[card.get('id')] = card
    try:   # references carry the card id, or a card picture under input/freevideo/agent/<card id>/
        media = (folder / 'media.json').read_text(encoding='utf-8')
    except OSError:
        media = ''
    try:
        prompt = (folder / 'prompt.txt').read_text(encoding='utf-8')
    except OSError:
        prompt = ''
    from ..lora_studio.cast_loras import read_links
    plain = media.replace('\\\\', '/').replace('\\', '/').lower()
    linked = {identity for identity, link in read_links(input_root).items() if isinstance(link, dict)
              and isinstance(link.get('lora'), str) and link['lora'].replace('\\', '/').lower() in plain}
    ids = list(wanted or ()) + [identity for identity, card in cards.items() if identity in linked
           or isinstance(identity, str) and identity in media
           or re.search(r'(@|<Subject \d+> is )%s\b' % re.escape(card['name']), prompt, re.I)]
    chosen = []
    root = Path(input_root).resolve()
    for identity in dict.fromkeys(i for i in ids if i in cards):
        card = cards[identity]
        pictures = []
        for image in card.get('images') or ():
            name = ((image or {}).get('media') or {}).get('file')
            if image.get('label') in CARD_LABELS and isinstance(name, str):
                path = (root / name).resolve()
                if path.is_relative_to(root) and path.is_file():
                    pictures.append(str(path))
        bar = card.get('face_bar')
        chosen.append(dict(id=identity, name=card['name'], pictures=pictures,
                           bar=float(bar) if isinstance(bar, (int, float)) and 0 < bar < 1 else DEFAULT_BAR))
    return chosen


def run_faces(video, cards, step=1.):
    python = face_python()
    if python is None:
        raise Unavailable('No face checker: install insightface in Eclipse\'s Python or set FREEVIDEO_FACE_PYTHON.')
    job = dict(video=str(video), refs={card['name']: card['pictures'] for card in cards if card['pictures']}, step=step)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    result = subprocess.run([python, '-I', str(Path(__file__).with_name('faces.py'))], input=json.dumps(job),
                            capture_output=True, text=True, timeout=FACE_TIMEOUT, creationflags=flags)
    if result.returncode:
        raise Unavailable('The face checker stopped: ' + (result.stderr.strip().splitlines() or ['no message'])[-1])
    return json.loads(result.stdout)


def assign(samples, names):
    """Per second, each card keeps at most its best-matching face of at least MIN_FACE_PX; smaller faces are counted."""
    tracks = {name: [] for name in names}
    small = 0
    for sample in samples:
        taken = set()
        faces = sorted(sample['faces'], key=lambda f: -(f.get('score') or 0))
        for face in faces:
            x0, y0, x1, y1 = face['box']
            if y1 - y0 < MIN_FACE_PX:
                small += 1
                continue
            name = face.get('name')
            if name in tracks and name not in taken and (face.get('score') or 0) >= SAME_PERSON:
                taken.add(name)
                tracks[name].append(dict(t=sample['t'], score=face['score'], x=(x0 + x1) / 2, size=y1 - y0))
    return tracks, small


def reversal(values, threshold):
    """Index where a series that moved at least `threshold` one way turns and comes back at least half of that."""
    if len(values) < 3:
        return None
    low = high = values[0]
    low_at = high_at = 0
    for index, value in enumerate(values):
        if value > high:
            high, high_at = value, index
        if value < low:
            low, low_at = value, index
        if high - values[0] >= threshold and high - value >= max(threshold, (high - values[0])) * 0.5:
            return high_at
        if values[0] - low >= threshold and value - low >= max(threshold, (values[0] - low)) * 0.5:
            return low_at
    return None


def motion_flags(tracks, width, cuts=()):
    """Walking back per shot: a hard cut is flagged on its own, so each side of it is looked at separately."""
    import math
    flags = []
    edges = [0., *sorted(cuts), float('inf')]
    shots = [(name, [p for p in track if start <= p['t'] < end])
             for name, track in tracks.items() for start, end in zip(edges, edges[1:])]
    for name, track in shots:
        if len(track) < 3:
            continue
        across = reversal([p['x'] / max(width, 1) for p in track], 0.15)   # framing drifts up to about 0.1
        depth = reversal([math.log(p['size']) for p in track], 0.15)
        for index, what in ((across, 'turns back across the frame'), (depth, 'walks back the way %s came')):
            if index is not None:
                flags.append(dict(check='walk_back', name=name, t=track[index]['t'],
                                  text='%s %s at %.0f s.' % (name, what.replace('%s', 'they'), track[index]['t'])))
    return flags


def face_summary(card, track):
    if not track:
        return dict(name=card['name'], scored=0, passed=None,
                    text='%s: no face big enough to score (%d px or more).' % (card['name'], MIN_FACE_PX))
    values = sorted(p['score'] for p in track)
    median = values[len(values) // 2]
    share = sum(v >= card['bar'] for v in values) / len(values)
    passed = median >= card['bar']
    return dict(name=card['name'], scored=len(values), median=round(median, 3), worst=round(values[0], 3),
                share=round(share, 2), bar=card['bar'], passed=passed,
                text='%s: median %.2f, worst %.2f, %d%% of seconds at %.2f or more (%s).'
                     % (card['name'], median, values[0], round(100 * share), card['bar'], 'pass' if passed else 'below the bar'))


# --- The card -----------------------------------------------------------------------------------------------------

def take_folder(output_root, identity):
    from ..comfy_library import _video
    return _video(output_root, identity).parent


def build(output_root, input_root, identity, cards=()):
    """Run every check on one saved take, save video.report-card.json beside it and return it."""
    folder = take_folder(output_root, identity)
    video = folder / 'video.mp4'
    if not video.is_file():
        raise Unavailable('This take has no video file.')
    fps, frames = small_frames(video)
    flags = frame_flags(frames, fps)
    cards = cast_cards(output_root, input_root, folder, cards)
    faces, notes = [], []
    try:
        result = run_faces(video, cards)
        tracks, small = assign(result['samples'], [c['name'] for c in cards])
        faces = [face_summary(card, tracks[card['name']]) for card in cards]
        flags += motion_flags(tracks, result.get('width') or 0, [f['t'] for f in flags if f['check'] in ('cut', 'pop')])
        for card in cards:
            if not card['pictures'] or card['name'] not in result.get('refs', {}):
                notes.append('%s\'s card has no face picture the checker could read; add a face or front picture.' % card['name'])
        if not cards:
            counted = max((len(s['faces']) for s in result['samples']), default=0)
            notes.append('No cast cards in this take, so faces were counted (up to %d at once) but not scored.' % counted)
        if small:
            notes.append('%d face sightings were under %d px and not scored.' % (small, MIN_FACE_PX))
    except (Unavailable, OSError, ValueError, subprocess.TimeoutExpired) as error:
        notes.append('Faces not checked: %s' % error)
    passed = all(f['passed'] is not False for f in faces) and not flags and not (cards and not faces)
    report = dict(version=1, id=identity, passed=passed, checks_passed=passed, frames=len(frames), fps=fps, faces=faces, flags=flags, notes=notes)
    (folder / FILE).write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report


def saved(output_root, identity):
    path = take_folder(output_root, identity) / FILE
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def schedule(video):
    """Check a take that just finished, on a background thread; a failed check never touches the take."""
    import logging
    import threading
    video = Path(video)
    identity = '%s/%s' % (video.parent.parent.name, video.parent.name)

    def run():
        try:
            import folder_paths
            report = build(folder_paths.get_output_directory(), folder_paths.get_input_directory(), identity)
            logging.info('[Eclipse] Report Card %s: %s', identity, 'pass' if report['passed'] else
                         '; '.join(f['text'] for f in report['faces'] + report['flags'] if f.get('passed') is not True)[:400])
        except Exception as error:
            logging.warning('[Eclipse] Report Card not made for %s: %s', identity, error)
    threading.Thread(target=run, name='eclipse-report-card', daemon=True).start()


def register():
    from aiohttp import web
    import asyncio
    import folder_paths
    from server import PromptServer
    server = PromptServer.instance
    if server is None or getattr(server, '_freevideo_report_card', False):
        return
    server._freevideo_report_card = True
    busy = asyncio.Semaphore(1)   # one check at a time: each holds a few CPU cores for a few seconds

    @server.routes.get('/freevideo/report-card')
    async def read(request):
        try:
            report = await asyncio.to_thread(saved, folder_paths.get_output_directory(), request.query.get('id'))
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(report, headers={'Cache-Control': 'no-store'})

    @server.routes.post('/freevideo/report-card')
    async def make(request):
        # The custom header keeps other websites from starting checks without the browser's pre-check.
        if request.headers.get('X-FreeVideo-Library') != '1':
            raise web.HTTPForbidden(text='Missing header')
        try:
            body = await request.json()
            cards = body.get('cards') or ()
            if not isinstance(cards, list) or not all(isinstance(c, str) for c in cards):
                raise ValueError('cards must be a list of card ids')
            async with busy:
                report = await asyncio.to_thread(build, folder_paths.get_output_directory(),
                                                 folder_paths.get_input_directory(), body.get('id'), cards)
        except (ValueError, TypeError, AttributeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        except Unavailable as error:
            return web.json_response(dict(error=str(error)), status=409)
        return web.json_response(report, headers={'Cache-Control': 'no-store'})

    @server.routes.post('/freevideo/report-card/look')
    async def look(request):
        """The picture reader's questions (wardrobe, set, hands, people) on one take, added to its Report Card."""
        from .look import look as ask_pictures
        if request.headers.get('X-FreeVideo-Library') != '1':
            raise web.HTTPForbidden(text='Missing header')
        service = getattr(server, '_freevideo_prompt_vlm', None)
        running, pending = server.prompt_queue.get_current_queue()
        if service is None:
            return web.json_response(dict(error='The picture reader is not available in this installation.'), status=409)
        if running or pending:
            return web.json_response(dict(error='The picture reader uses the graphics card: ask again when no video is rendering.'), status=409)
        try:
            body = await request.json()
            identity = body.get('id')
            output, inputs = folder_paths.get_output_directory(), folder_paths.get_input_directory()
            async with busy:
                report = await asyncio.to_thread(saved, output, identity) or await asyncio.to_thread(build, output, inputs, identity)
                report = await asyncio.to_thread(ask_pictures, service, output, inputs, identity, report)
        except (TypeError, AttributeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        except ValueError as error:
            words = {'busy': 'The picture reader is busy with another job; try again in a moment.',
                     'model_missing': 'The picture reader is not installed yet (Settings, prompt helper).'}
            return web.json_response(dict(error=words.get(str(error), str(error))), status=409)
        except (Unavailable, RuntimeError, OSError) as error:
            return web.json_response(dict(error='Picture questions failed: %s' % error), status=409)
        return web.json_response(report, headers={'Cache-Control': 'no-store'})
