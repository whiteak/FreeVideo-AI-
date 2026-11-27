"""Film-level continuity (plan step 1.3): several takes in film order, compared shot to shot.

- colour: skin saturation and brightness of each shot against the film's median. Saturation is measured on skin
  pixels (YCrCb skin range), not the whole frame, because noisy blacks skew a whole-frame measure;
- set and wardrobe (only when asked, it uses the graphics card): the picture reader compares the last frame of each
  shot with the first frame of the next one.
"""
import json
import time
from pathlib import Path

SAMPLES = 5            # frames per shot for the colour measure
SAT_LIMIT = .12        # skin saturation more than 12% away from the film median is flagged
LUMA_LIMIT = 18.       # 0-255 brightness further than this from the film median is flagged
MIN_SKIN = 400         # pixels of skin needed for a saturation reading


def skin_colour(frames):
    """(skin saturation 0-1 or None, mean brightness 0-255) over RGB uint8 frames."""
    import numpy as np
    sats, lumas = [], []
    for rgb in frames:
        rgb = rgb.astype(np.float32)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        y = .299 * r + .587 * g + .114 * b
        cr, cb = (r - y) * .713 + 128, (b - y) * .564 + 128
        skin = (cr > 133) & (cr < 173) & (cb > 77) & (cb < 127) & (y > 40)
        lumas.append(float(y.mean()))
        if skin.sum() >= MIN_SKIN:
            high, low = rgb.max(axis=-1), rgb.min(axis=-1)
            sats.append(float(((high - low) / np.maximum(high, 1))[skin].mean()))
    return (float(np.median(sats)) if sats else None), float(np.median(lumas))


def sample_frames(video, count, width=256):
    """SAMPLES frames spread over the take (skipping the pops), plus the first and last of them, as RGB arrays."""
    import av
    wanted = {round(3 + (count - 5) * i / (SAMPLES - 1)) for i in range(SAMPLES)}
    rows = []
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                height = max(2, round(frame.height * width / frame.width))
                rows.append((index, frame.reformat(width=width, height=height, format='rgb24').to_ndarray(), frame))
            if index >= max(wanted):
                break
    return rows


def colour_flags(shots):
    """shots: [{name, sat, luma}] in film order; flags the ones off the film's median."""
    import numpy as np
    flags = []
    sats = [s['sat'] for s in shots if s['sat'] is not None]
    sat_mid = float(np.median(sats)) if sats else None
    luma_mid = float(np.median([s['luma'] for s in shots]))
    for number, shot in enumerate(shots, 1):
        if sat_mid and shot['sat'] is not None and abs(shot['sat'] / sat_mid - 1) > SAT_LIMIT:
            change = shot['sat'] / sat_mid - 1
            flags.append(dict(check='colour', shot=number, text='Shot %d: skin %d%% %s saturated than the rest of the film.'
                              % (number, round(abs(change) * 100), 'more' if change > 0 else 'less')))
        if abs(shot['luma'] - luma_mid) > LUMA_LIMIT:
            flags.append(dict(check='brightness', shot=number, text='Shot %d is %s than the rest of the film (%.0f vs %.0f).'
                              % (number, 'brighter' if shot['luma'] > luma_mid else 'darker', shot['luma'], luma_mid)))
    return flags


def order(output_root, identities):
    """Film order: the order the takes were made in."""
    from . import take_folder
    return sorted(identities, key=lambda i: (take_folder(output_root, i) / 'video.mp4').stat().st_mtime)


def check(output_root, input_root, identities, service=None):
    """Continuity across the takes; with `service` (the picture reader) also set and wardrobe at each cut."""
    import av
    from . import take_folder
    identities = order(output_root, identities)
    shots, ends = [], []
    for identity in identities:
        video = take_folder(output_root, identity) / 'video.mp4'
        with av.open(str(video)) as container:
            count = container.streams.video[0].frames or 0
        if count < 8:
            raise ValueError('A take is too short or its frame count is unknown.')
        rows = sample_frames(video, count)
        sat, luma = skin_colour([rgb for _, rgb, _ in rows])
        shots.append(dict(id=identity, sat=None if sat is None else round(sat, 3), luma=round(luma, 1)))
        ends.append((rows[0][2], rows[-1][2]))
    flags = colour_flags(shots)
    notes = []
    if service is not None and len(identities) > 1:
        folder = Path(input_root).resolve() / 'freevideo' / 'report-card' / ('film-%d' % int(time.time()))
        folder.mkdir(parents=True, exist_ok=True)
        questions = []
        for number in range(1, len(identities)):
            names = []
            for side, frame in (('end', ends[number - 1][1]), ('start', ends[number][0])):
                path = folder / ('shot%02d-%s.jpg' % (number + (side == 'start'), side))
                frame.to_image().save(path, quality=90)
                names.append(path.relative_to(Path(input_root).resolve()).as_posix())
            questions.append(dict(images=[dict(file=names[0], label='end of shot %d' % number),
                                          dict(file=names[1], label='start of shot %d' % (number + 1))],
                                  text='These are two shots of one film, cut together. Ignore camera angle and framing. '
                                       'Do the set (floor, walls, light) and each person\'s clothes, hair and accessories '
                                       'match across the cut? Return JSON: {"differences": [short text]}, an empty list '
                                       'when they match.'))
        from .look import ask, SYSTEM
        from ..prompt_vlm.rules import parse_json
        for start in range(0, len(questions), 8):   # the reader takes eight questions per job
            batch = questions[start:start + 8]
            answers = ask(service, dict(mode='ask', format='json', max_tokens=300, system=SYSTEM, questions=batch))
            for offset, answer in enumerate(answers):
                number = start + offset + 1
                try:
                    found = parse_json(answer.get('text', ''))[1].get('differences') or []
                except (ValueError, AttributeError):
                    notes.append('No usable answer for the cut after shot %d.' % number)
                    continue
                for what in [w for w in found if isinstance(w, str) and w.strip()][:3]:
                    flags.append(dict(check='cut_match', shot=number + 1,
                                      text='Cut %d to %d: %s' % (number, number + 1, what.strip()[:160])))
    elif len(identities) > 1:
        notes.append('Set and wardrobe across the cuts not checked: use "Ask the picture reader" for that.')
    if not any(s['sat'] is not None for s in shots):
        notes.append('No skin big enough in any shot, so saturation was not compared.')
    return dict(shots=shots, flags=flags, notes=notes, passed=not flags)
