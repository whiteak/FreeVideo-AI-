"""Take Report Card (fork addition, plan step 1.1): checks a saved take and writes video.report-card.json beside it.

Checks, all on the CPU and none of them downloading anything:
- faces: every cast character in the take scored once a second against its card pictures (insightface buffalo_l,
  run by faces.py in its own process), with a pass bar per card (card "face_bar", default 0.72);
- frame-0 pop: the first frames jump far more than the rest of the take;
- flicker: brightness pumps up and down for a frame or two;
- cuts: a hard jump to another picture in the middle of a take (a restart);
- back to the start: the picture drifts away from the opening frame and then returns to it;
- walking back: a character moves one way and then comes back (from the face boxes).
- speech: the prompt's spoken lines heard (faster-whisper), digits in lines, and the lip-sync offset (speech.py).
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
        quiet = set(cut_at) | {c - 1 for c in cut_at} | {0, 1, 2, 3, 4, n - 1}   # pops and cuts are flagged on their own
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


def run_faces(video, cards, step=1., mouth=None):
    python = face_python()
    if python is None:
        raise Unavailable('No face checker: install insightface in Eclipse\'s Python or set FREEVIDEO_FACE_PYTHON.')
    job = dict(video=str(video), refs={card['name']: card['pictures'] for card in cards if card['pictures']}, step=step, mouth=mouth)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    result = subprocess.run([python, '-I', str(Path(__file__).with_name('faces.py'))], input=json.dumps(job),
                            capture_output=True, text=True, timeout=FACE_TIMEOUT, creationflags=flags)
    if result.returncode:
        raise Unavailable('The face checker stopped: ' + (result.stderr.strip().splitlines() or ['no message'])[-1])
    return json.loads(result.stdout)


def assign(samples, names, min_px=MIN_FACE_PX):
    """Per second, each card keeps at most its best-matching face of at least min_px; smaller faces are counted."""
    tracks = {name: [] for name in names}
    small = 0
    for sample in samples:
        taken = set()
        faces = sorted(sample['faces'], key=lambda f: -(f.get('score') or 0))
        for face in faces:
            x0, y0, x1, y1 = face['box']
            if y1 - y0 < min_px:
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


def face_summary(card, track, min_px=MIN_FACE_PX):
    if not track:
        return dict(name=card['name'], scored=0, passed=None,
                    text='%s: no face big enough to score (%d px or more).' % (card['name'], min_px))
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
    faces, notes, words, span = [], [], [], None
    from . import speech
    lines, audio = speech.spoken_lines(folder), None
    if lines:
        try:
            from ..subtitle_ears import decode_mono, EarsError
            audio = decode_mono(video)
            words = speech.heard_words(audio)
            flags += speech.line_flags(lines, words)
            span = speech.voice_span(words, fps, len(frames))
            notes.append('Heard: "%s"' % ' '.join(w['text'] for w in words)[:300] if words else 'No speech heard.')
        except (Unavailable, EarsError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
            notes.append('Speech not checked: %s' % error)
    try:
        result = run_faces(video, cards, mouth=span)
        if span:
            sync, said = speech.sync_flags(speech.lip_lag(result.get('mouth') or [], speech.loudness(audio, fps, *span)), fps)
            flags += sync
            notes += said
        # A half-size draft (preview) has half-size faces: the smallest scored face scales with the picture.
        min_px = min(MIN_FACE_PX, max(32, round(MIN_FACE_PX * (result.get('height') or 1472) / 1472)))
        tracks, small = assign(result['samples'], [c['name'] for c in cards], min_px)
        faces = [face_summary(card, tracks[card['name']], min_px) for card in cards]
        flags += motion_flags(tracks, result.get('width') or 0, [f['t'] for f in flags if f['check'] in ('cut', 'pop')])
        for card in cards:
            if not card['pictures'] or card['name'] not in result.get('refs', {}):
                notes.append('%s\'s card has no face picture the checker could read; add a face or front picture.' % card['name'])
        if not cards:
            counted = max((len(s['faces']) for s in result['samples']), default=0)
            notes.append('No cast cards in this take, so faces were counted (up to %d at once) but not scored.' % counted)
        if small:
            notes.append('%d face sightings were under %d px and not scored.' % (small, min_px))
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

    @server.routes.post('/freevideo/report-card/rank')
    async def ranked(request):
        """Best take first among several saved takes (plan step 1.2: best-take picker)."""
        if request.headers.get('X-FreeVideo-Library') != '1':
            raise web.HTTPForbidden(text='Missing header')
        try:
            body = await request.json()
            ids = body.get('ids')
            if not isinstance(ids, list) or not 2 <= len(ids) <= 16 or not all(isinstance(i, str) for i in ids):
                raise ValueError('ids must list 2 to 16 saved takes')
            async with busy:
                reports = await asyncio.to_thread(rank, folder_paths.get_output_directory(),
                                                  folder_paths.get_input_directory(), list(dict.fromkeys(ids)))
        except (ValueError, TypeError, AttributeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        except Unavailable as error:
            return web.json_response(dict(error=str(error)), status=409)
        return web.json_response(reports, headers={'Cache-Control': 'no-store'})

    @server.routes.post('/freevideo/report-card/continuity')
    async def continuity(request):
        """Colour across several takes in film order; with look=true also set and wardrobe at each cut (plan 1.3)."""
        from .continuity import check
        if request.headers.get('X-FreeVideo-Library') != '1':
            raise web.HTTPForbidden(text='Missing header')
        try:
            body = await request.json()
            ids = body.get('ids')
            if not isinstance(ids, list) or not 2 <= len(ids) <= 16 or not all(isinstance(i, str) for i in ids):
                raise ValueError('ids must list 2 to 16 saved takes')
            service = None
            if body.get('look') is True:
                service = getattr(server, '_freevideo_prompt_vlm', None)
                running, pending = server.prompt_queue.get_current_queue()
                if service is None or running or pending:
                    return web.json_response(dict(error='The picture reader is free only when no video is rendering; ask again then.'), status=409)
            async with busy:
                result = await asyncio.to_thread(check, folder_paths.get_output_directory(), folder_paths.get_input_directory(),
                                                 list(dict.fromkeys(ids)), service)
        except (TypeError, AttributeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        except (ValueError, RuntimeError, OSError) as error:
            return web.json_response(dict(error='Continuity not checked: %s' % error), status=409)
        return web.json_response(result, headers={'Cache-Control': 'no-store'})

    from .drafts import Runs
    runs = Runs()

    @server.routes.post('/freevideo/report-card/drafts')
    async def drafts(request):
        """Draft several seeds small, check each, finish only the best (plan step 1.2). Body: {spec, drafts}, where
        spec is an agent generate job without takes, seed or preview."""
        if request.headers.get('X-FreeVideo-Library') != '1':
            raise web.HTTPForbidden(text='Missing header')
        jobs = getattr(server, '_freevideo_agent', None)
        if jobs is None:
            return web.json_response(dict(error='Drafting needs the agent API, which is off in this installation.'), status=409)
        try:
            body = await request.json()
            spec, count = body.get('spec'), body.get('drafts', 3)
            if not isinstance(spec, dict) or not isinstance(spec.get('prompt'), str) or type(count) is not int or not 2 <= count <= 6:
                raise ValueError('spec needs a prompt, and drafts is 2 to 6')
            state = runs.start(jobs, spec, count, folder_paths.get_output_directory(), folder_paths.get_input_directory())
        except (ValueError, TypeError, AttributeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(state)

    @server.routes.get('/freevideo/report-card/drafts')
    async def draft_state(request):
        state = runs.get(request.query.get('id'))
        if state is None:
            raise web.HTTPNotFound(text='Unknown draft run')
        return web.json_response(state, headers={'Cache-Control': 'no-store'})

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


# Flags that make a take unusable for the director's judge; the rest are worth a look (a pop is trimmed, a lip-sync
# offset is shifted).
PROBLEMS = ('back_to_start', 'walk_back', 'line', 'people', 'changes', 'differences', 'problems')


def take_qa_part(report):
    """(problems, notes) in take_qa's words, so the director drops takes whose faces or motion fail (plan step 1.2)."""
    problems, notes = [], []
    for face in report.get('faces') or ():
        if face.get('passed') is False:
            problems.append("%s's face %.2f, under %.2f" % (face['name'], face['median'], face['bar']))
        elif face.get('passed'):
            notes.append("%s's face %.2f" % (face['name'], face['median']))
    for flag in (report.get('flags') or []) + ((report.get('look') or {}).get('flags') or []):
        (problems if flag.get('check') in PROBLEMS else notes).append(flag['text'].rstrip('.'))
    return problems, notes


def for_take(output_root, input_root, identity):
    """The take's Report Card (made now unless the background check already saved one), as take_qa problems/notes."""
    report = saved(output_root, identity) or build(output_root, input_root, identity)
    return take_qa_part(report)


def score(report):
    """Sort key for picking the best of several takes: passing first, then the weakest face, then fewest flags."""
    medians = [f['median'] for f in report.get('faces') or () if f.get('median') is not None]
    flags = len(report.get('flags') or ()) + len((report.get('look') or {}).get('flags') or ())
    problems = len(take_qa_part(report)[0])
    return (bool(report.get('passed')), -problems, min(medians) if medians else 0., -flags)


def rank(output_root, input_root, identities):
    """The takes best first, each with its Report Card (made now where missing)."""
    reports = [saved(output_root, i) or build(output_root, input_root, i) for i in identities]
    return sorted(reports, key=score, reverse=True)
