"""Report Card picture questions (plan step 1.1): the installed picture reader (Qwen3-VL, prompt_vlm) is shown four
frames of a take and asked about people count, hands, wardrobe and set. It uses the graphics card, so it runs only
when asked and only while no video renders; the answers are added to the take's Report Card.
"""
import json
import time
from pathlib import Path

FRAMES = 4
WAIT_SECONDS = 600
SYSTEM = ('You check frames of an AI-generated video for continuity mistakes. Look closely and answer only from what '
          'is visible. Never guess: if something cannot be seen, say so.')


def frame_times(count, fps):
    """Four frames spread over the take, skipping the first three frames and the last one (their pops are flagged
    by the frame checks)."""
    first, last = min(3, count - 1), max(0, count - 2)
    return [round(first + (last - first) * i / (FRAMES - 1)) for i in range(FRAMES)]


def save_frames(video, folder, count):
    """Saves the frames as JPEGs in `folder` and returns [(frame index, path)]."""
    import av
    folder.mkdir(parents=True, exist_ok=True)
    wanted = frame_times(count, None)
    saved = []
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                path = folder / ('frame-%05d.jpg' % index)
                frame.to_image().save(path, quality=90)
                saved.append((index, path))
            if index >= wanted[-1]:
                break
    return saved


def wardrobe_of(card, links):
    """The words that describe what a character wears: the card's LoRA recipe words, else its description."""
    link = links.get(card['id']) if isinstance(links.get(card['id']), dict) else {}
    words = link.get('prompt_suffix') or card.get('description') or ''
    return ' '.join(str(words).split())[:700]


def questions(images, cards, links):
    """The ask request: images are ComfyUI input names, cards the take's character cards (with 'description')."""
    shown = [dict(file=name, label='frame %d' % (i + 1)) for i, name in enumerate(images)]
    n = len(images)
    rows = [
        dict(text='How many people are visible in each of the %d frames? Count partly visible people too. '
                  'Return JSON: {"counts": [one integer per frame]}.' % n, images=shown),
        dict(text='Look at every visible hand. Are there extra, missing, fused or bent-wrong fingers, extra hands or '
                  'limbs, or hands melting into objects? Return JSON: {"problems": [{"frame": number, "what": short '
                  'text}]}, an empty list when the hands look right.', images=shown),
        dict(text='Compare only the set across the frames: the floor, walls and lighting. Ignore the people, what they '
                  'do, and doors opening or closing. Report only a clear change of material or colour (for example a '
                  'black marble floor turning white, or night turning to day). Return JSON: {"changes": [{"frame": '
                  'number, "what": short text}]}, an empty list when the set holds.', images=shown),
    ]
    for card in cards[:5]:
        wear = wardrobe_of(card, links)
        if wear:
            rows.append(dict(text='%s should be wearing: %s. In each frame where %s is visible, check only these: a wrong '
                                  'colour, fabric that is sheer or see-through where it should be solid, a missing item, '
                                  'or an accessory on the wrong side (such as a watch on the wrong wrist). Do not report '
                                  'whether a coat is open or closed, seams, folds, creases or lighting. Return JSON: '
                                  '{"differences": [{"frame": number, "what": short text}]}, an empty list when it matches.'
                                  % (card['name'], wear, card['name']), images=shown))
    return dict(mode='ask', format='json', max_tokens=500, system=SYSTEM, questions=rows)


def read_answers(request, answers, frames, fps, cards):
    """Flags and notes from the reader's answers; a reply that is not the asked JSON becomes a note."""
    from ..prompt_vlm.rules import parse_json
    flags, notes, at = [], [], lambda k: '%.1f s' % (frames[k - 1] / fps) if 1 <= k <= len(frames) else 'frame %s' % k
    named = [None, None, None] + [c['name'] for c in cards[:5] if c.get('_asked')]
    for question, answer, name in zip(request['questions'], answers, named):
        try:
            value = parse_json(answer.get('text', ''))[1]
        except ValueError:
            notes.append('The picture reader gave no usable answer to one question.')
            continue
        if not isinstance(value, dict):
            continue
        if 'counts' in value:
            counts = [c for c in value['counts'] if isinstance(c, int)] if isinstance(value['counts'], list) else []
            # People entering is often the story; more people than both the cast and the opening frame is not.
            allowed = max([len(cards)] + counts[:1])
            extra = [i + 1 for i, c in enumerate(counts) if c > allowed]
            if extra:
                flags.append(dict(check='people', t=frames[extra[0] - 1] / fps,
                                  text='People count: %s seen, more than the cast cards found and the opening frame (at %s).'
                                       % ('/'.join(map(str, counts)), ', '.join(at(k) for k in extra))))
            elif counts:
                notes.append('People seen in the four frames: %s.' % '/'.join(map(str, counts)))
            continue
        for key, label in (('problems', 'Hands'), ('changes', 'Set'), ('differences', 'Wardrobe')):
            rows = value.get(key)
            if isinstance(rows, list):
                for row in rows[:4]:
                    if isinstance(row, dict) and isinstance(row.get('what'), str) and row['what'].strip():
                        frame = row.get('frame') if isinstance(row.get('frame'), int) else 0
                        who = (' (%s)' % name) if key == 'differences' and name else ''
                        flags.append(dict(check=key, t=frames[frame - 1] / fps if 1 <= frame <= len(frames) else 0,
                                          text='%s%s at %s: %s' % (label, who, at(frame), row['what'].strip()[:160])))
    return flags, notes


def ask(service, request, cancelled=lambda: False):
    """Runs one ask job on the picture reader service and returns its answers."""
    job = service.start('ask', request)['job']
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        state = service.status(job)
        if state.get('phase') == 'complete':
            return state['answers']
        if state.get('phase') in ('failed', 'cancelled'):
            raise RuntimeError('the picture reader stopped (%s)' % state.get('error', state['phase']))
        if cancelled():
            service.cancel(job)
        time.sleep(.5)
    service.cancel(job)
    raise RuntimeError('the picture reader took too long')


def look(service, output_root, input_root, identity, report):
    """Asks the picture questions about one take and folds the answers into its Report Card."""
    from . import FILE, cast_cards, take_folder
    from ..lora_studio.cast_loras import read_links
    folder = take_folder(output_root, identity)
    root = Path(input_root).resolve()
    shots = root / 'freevideo' / 'report-card' / identity.replace('/', '-')
    frames = save_frames(folder / 'video.mp4', shots, report['frames'])
    cards = cast_cards(output_root, input_root, folder)
    store = Path(output_root) / 'FreeVideo' / '.agent' / 'jobs' / 'cast'
    for card in cards:   # the description lives on the card itself
        try:
            card['description'] = json.loads((store / (card['id'] + '.json')).read_text(encoding='utf-8'))['card'].get('description')
        except (OSError, ValueError, KeyError, TypeError):
            card['description'] = ''
    links = read_links(input_root)
    request = questions([p.relative_to(root).as_posix() for _, p in frames], cards, links)
    for card in cards[:5]:
        card['_asked'] = bool(wardrobe_of(card, links))
    answers = ask(service, request)
    fps = report.get('fps') or 24.
    flags, notes = read_answers(request, answers, [i for i, _ in frames], fps, cards)
    report = dict(report, look=dict(flags=flags, notes=notes, frames=[i for i, _ in frames], at=time.time()))
    report['passed'] = report.get('checks_passed', report['passed']) and not flags
    (folder / FILE).write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report
