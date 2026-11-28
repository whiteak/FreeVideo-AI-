"""The Take Inspector: checks on a finished take that the basic picture and sound checks of take_qa miss.

- Opening frame held: the given first frame is in the take but the picture jumps away from it within the first
  second (a character LoRA re-drawing the scene is the usual cause). The basic check compares frame 0 alone and
  calls such a jump a single odd frame, which would be trimmed away and hide the failure.
- Back-and-forth: the camera moves one way and then back the other way, or the action returns to where it already
  was (a person walks off and back, a take restarts its action half way).
- Faces: each cast member's face in the take against the face on their cast card (needs a face model, see face_id).
- Lip sync: the mouth movement against the speech, with the offset that would line them up.

Everything here works on frames already decoded by take_qa, except the face and lip checks, which read the take
again at a larger size. Every check is plain data; take_qa.check folds the findings into problems and notes.
"""
import numpy as np

OPEN_SECONDS = 1.0        # the given first frame must still be the picture this long
OPEN_DROP = 30.0          # 0-255 mean difference from the given frame that is a different picture
OPEN_HELD = 12.0          # ...and frame 0 at least this close to it (else the first frame was never given)
OPEN_JUMP = 18.0          # 0-255 mean change between two frames that is a jump
OPEN_RATIO = 3.0          # ...and this many times the motion that follows it
OPEN_LOOK = 6             # frames after the jump that set the motion that follows
MOTION_STRIDE = 4         # frames between the two pictures whose shift gives the camera motion
MOTION_SPEED = 0.05       # picture widths per second that count as a camera move
MOTION_SECONDS = 0.6      # a move held at least this long
MOTION_GAP = 1.5          # s, a move and the move back within this are one back-and-forth
RESTART_STEP = 0.25       # s between the frames compared for a return
RESTART_AWAY = 24.0       # 0-255 mean difference: the action has moved on from a moment
RESTART_BACK = 9.0        # ...and a later frame this close to it is the same moment again
RESTART_GAP = 2.0         # s between the moment and its return
FACE_EVERY = 0.5          # s between the frames whose faces are compared
FACE_FRAME_BAR = 0.68     # similarity a single frame should reach
FACE_MEDIAN_BAR = 0.72    # median similarity a take must reach for a cast member
FACE_SHARE = 0.8          # ...with at least this share of frames above the frame bar
FACE_MIN_SEEN = 3         # frames a face must be seen in before it is judged
FACE_SIDE = 960           # long side of the frames read for faces
LIP_SIDE = 480            # long side of the frames read for the mouth
LIP_LAG = 0.5             # s, the largest offset looked for
LIP_OFFSET = 0.1          # s, an offset this large is worth fixing
LIP_TRUST = 0.25          # correlation below which the offset is not trusted


def _diff(a, b):
    return float(np.abs(a - b).mean())


# Opening frame -----------------------------------------------------------------------------------------------

def opening(still, small, times, planned=None):
    """Where the take leaves the given first frame inside the first second: {'held', 'left_at', 'diff'}.
    `still` is the given frame at the analysis size; None when the frame was not given or cannot be compared.
    A planned cut inside the first second is allowed."""
    if still is None or not small:
        return None
    first = _diff(still, small[0])
    if first > OPEN_HELD:
        return dict(held=False, left_at=0.0, diff=round(first, 1))
    allowed = [p for p in (planned or []) if 0 < p <= OPEN_SECONDS + 0.5]
    for index in range(1, len(small)):
        if times[index] - times[0] > OPEN_SECONDS:
            break
        if any(abs(times[index] - p) <= 0.6 for p in allowed):
            return dict(held=True, left_at=None, diff=round(first, 1))
        away = _diff(still, small[index])
        step = _diff(small[index - 1], small[index])
        after = [_diff(small[n - 1], small[n]) for n in range(index + 1, min(len(small), index + 1 + OPEN_LOOK))]
        # a jump, not a camera move or an action carrying the picture away from it
        if away > OPEN_DROP and step >= OPEN_JUMP and step >= OPEN_RATIO * max(np.median(after) if after else 0.0, 1e-3):
            return dict(held=False, left_at=round(times[index], 2), diff=round(away, 1))
    return dict(held=True, left_at=None, diff=round(first, 1))


# Camera motion -----------------------------------------------------------------------------------------------

def _shift(a, b):
    """(dx, dy) in pixels that moves picture a onto picture b, by phase correlation."""
    height, width = a.shape
    window = np.outer(np.hanning(height), np.hanning(width))
    fa = np.fft.fft2((a - a.mean()) * window)
    fb = np.fft.fft2((b - b.mean()) * window)
    cross = fb * np.conj(fa)
    cross /= np.abs(cross) + 1e-9
    surface = np.real(np.fft.ifft2(cross))
    y, x = np.unravel_index(int(np.argmax(surface)), surface.shape)

    def refine(index, size, values):
        left, right = values[(index - 1) % size], values[(index + 1) % size]
        middle = values[index]
        bend = left - 2 * middle + right
        offset = 0.5 * (left - right) / bend if bend < 0 else 0.0
        position = index + offset
        return position - size if position > size / 2 else position

    return refine(x, width, surface[y, :]), refine(y, height, surface[:, x])


def camera(small, times, cuts=()):
    """Camera speed per moment, in picture widths (x) and heights (y) per second: [(t, vx, vy)].
    Moments across a cut are left out."""
    path = []
    for index in range(0, len(small) - MOTION_STRIDE):
        later = index + MOTION_STRIDE
        if any(times[index] < c <= times[later] for c in cuts):
            continue
        span = times[later] - times[index]
        if span <= 0:
            continue
        dx, dy = _shift(small[index], small[later])
        height, width = small[index].shape
        path.append(((times[index] + times[later]) / 2, dx / width / span, dy / height / span))
    return path


def _moves(path, axis):
    """Held moves along one axis: [(start, end, sign)]."""
    moves, current = [], None
    for row in path:
        t, speed = row[0], row[axis]
        sign = 1 if speed > MOTION_SPEED else -1 if speed < -MOTION_SPEED else 0
        if current and (sign != current[2] or t - current[1] > 0.5):
            moves.append(current)
            current = None
        if sign:
            current = [current[0], t, sign] if current else [t, t, sign]
    if current:
        moves.append(current)
    return [(round(a, 2), round(b, 2), s) for a, b, s in moves if b - a >= MOTION_SECONDS]


def reversals(path, cuts=()):
    """Camera moves followed by the opposite move: [{'axis', 'out', 'back'}] with the seconds of each."""
    found = []
    for axis, name in ((1, 'sideways'), (2, 'up and down')):
        moves = _moves(path, axis)
        for one, two in zip(moves, moves[1:]):
            if one[2] != two[2] and two[0] - one[1] <= MOTION_GAP and not any(one[1] <= c <= two[0] for c in cuts):
                found.append(dict(axis=name, out=[one[0], one[1]], back=[two[0], two[1]]))
    return found


# Action restart ----------------------------------------------------------------------------------------------

def returns(small, times, planned=None):
    """Moments the picture comes back to after it had clearly moved on: [{'at', 'back_to'}], earliest first.
    With planned cuts, only frames inside the same planned shot are compared (a shot-reverse-shot returns
    to an angle on purpose)."""
    if len(small) < 3:
        return []
    picked, last = [], None
    for index, t in enumerate(times):
        if last is None or t - last >= RESTART_STEP:
            picked.append(index)
            last = t
    edges = sorted(p for p in (planned or []) if p > 0)

    def segment(t):
        return sum(1 for edge in edges if t >= edge - 0.3)

    found = []
    for a, i in enumerate(picked):
        away = 0.0
        for j in picked[a + 1:]:
            if segment(times[j]) != segment(times[i]):
                break
            gap = _diff(small[i], small[j])
            away = max(away, gap)
            if times[j] - times[i] >= RESTART_GAP and away >= RESTART_AWAY and gap <= RESTART_BACK:
                found.append(dict(at=round(times[j], 2), back_to=round(times[i], 2)))
                break
        if found:
            break
    return found


# Faces -------------------------------------------------------------------------------------------------------

def _frames(path, every, side):
    """(time, RGB array) every `every` seconds, the long side at most `side`."""
    from . import media
    out, next_at = [], 0.0
    with media._reading(media._file(path)) as container:
        stream = media._video_stream(container)
        for count, frame in enumerate(container.decode(stream)):
            t = float(frame.time) if frame.time is not None else count / 24.0
            if t + 1e-6 < next_at:
                continue
            k = min(1.0, side / max(frame.width, frame.height))
            image = frame.to_image()
            if k < 1:
                image = image.resize((max(1, round(frame.width * k)), max(1, round(frame.height * k))))
            out.append((t, np.asarray(image.convert('RGB'))))
            next_at = t + every
    return out


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def faces(path, references, embed, frames=None):
    """How well each cast member's face holds through the take.

    `references` maps a name to the face vectors from their cast card; `embed(rgb)` returns the face vectors
    found in one RGB frame. In each frame every face goes to at most one cast member, best match first.
    Returns {name: {'seen', 'median', 'worst', 'share', 'scores'}}."""
    frames = frames if frames is not None else _frames(path, FACE_EVERY, FACE_SIDE)
    scores = {name: [] for name in references}
    for _, rgb in frames:
        found = embed(rgb)
        pairs = sorted(((max(_cosine(vector, ref) for ref in refs), name, n)
                        for name, refs in references.items() if refs for n, vector in enumerate(found)), reverse=True)
        used_names, used_faces = set(), set()
        for score, name, n in pairs:
            if name in used_names or n in used_faces:
                continue
            used_names.add(name)
            used_faces.add(n)
            scores[name].append(score)
    result = {}
    for name, values in scores.items():
        values = [v for v in values if v > 0.3]  # a lower match is a different person, not this one badly drawn
        result[name] = dict(seen=len(values), median=round(float(np.median(values)), 3) if values else None,
                            worst=round(min(values), 3) if values else None,
                            share=round(sum(v >= FACE_FRAME_BAR for v in values) / len(values), 2) if values else None,
                            scores=[round(v, 3) for v in values])
    return result


def face_verdicts(found):
    """(problems, notes) from faces()."""
    problems, notes = [], []
    for name, row in found.items():
        if row['seen'] < FACE_MIN_SEEN:
            notes.append('%s seen in only %d checked frames' % (name, row['seen']))
        elif row['median'] < FACE_MEDIAN_BAR:
            problems.append("%s's face does not match the cast card (median %.2f, bar %.2f)" % (name, row['median'], FACE_MEDIAN_BAR))
        elif row['share'] < FACE_SHARE:
            notes.append("%s's face drifts in places (worst %.2f)" % (name, row['worst']))
    return problems, notes


# Lip sync ----------------------------------------------------------------------------------------------------

def mouth_motion(path, find_faces):
    """(times, motion) of the mouth of the largest face per frame. `find_faces(rgb)` returns faces with
    'landmarks' as face_box gives them: eyes, nose and the two mouth corners."""
    times, motion, previous = [], [], None
    for t, rgb in _frames(path, 0.0, LIP_SIDE):
        found = find_faces(rgb)
        crop = None
        if found:
            marks = found[0]['landmarks']
            nose, left, right = marks[2], marks[3], marks[4]
            width = max(4.0, right[0] - left[0])
            top = nose[1] + 0.2 * (min(left[1], right[1]) - nose[1])
            bottom = max(left[1], right[1]) + 0.6 * width
            x0, x1 = int(left[0] - 0.25 * width), int(right[0] + 0.25 * width)
            region = rgb[max(0, int(top)):max(0, int(bottom)), max(0, x0):max(0, x1)]
            if region.size:
                from PIL import Image
                crop = np.asarray(Image.fromarray(region).convert('L').resize((32, 20)), np.float32)
        motion.append(_diff(crop, previous) if crop is not None and previous is not None else 0.0)
        previous = crop
        times.append(t)
    return times, motion


def lip_offset(times, motion, samples, rate):
    """Seconds the sound should move to line up with the mouth (positive: later), and how sure that is:
    {'offset', 'trust'}. None when there is too little of either."""
    if len(times) < 24 or samples.size < rate:
        return None
    fps = (len(times) - 1) / max(1e-6, times[-1] - times[0])
    envelope = []
    for t in times:
        a, b = int(t * rate), int((t + 1 / fps) * rate)
        chunk = samples[a:b]
        envelope.append(float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0)
    sound = np.abs(np.diff(np.asarray(envelope)))
    mouth = np.asarray(motion[1:], np.float64)
    if sound.std() < 1e-6 or mouth.std() < 1e-6:
        return None
    sound = (sound - sound.mean()) / sound.std()
    mouth = (mouth - mouth.mean()) / mouth.std()
    best, best_lag = -1.0, 0
    limit = int(round(LIP_LAG * fps))
    for lag in range(-limit, limit + 1):
        if lag >= 0:
            a, b = mouth[lag:], sound[:len(sound) - lag]
        else:
            a, b = mouth[:lag], sound[-lag:]
        if len(a) < 12:
            continue
        value = float(np.mean(a * b))
        if value > best:
            best, best_lag = value, lag
    return dict(offset=round(best_lag / fps, 3), trust=round(best, 2))


# Fixes -------------------------------------------------------------------------------------------------------

def repair(path, out, *, drop_first=0, drop_last=0, audio_shift=0.0):
    """Write a fixed copy of a take: `drop_first`/`drop_last` frames cut off (a single odd edge frame) and the
    sound moved by `audio_shift` seconds (positive: later), kept to the picture's length. Returns the seconds
    written. The picture is re-encoded at high quality; the original file is left as it is."""
    import av
    from fractions import Fraction
    from pathlib import Path
    from . import media
    source = media._file(path)
    with media._reading(source) as container:
        stream = media._video_stream(container)
        rate = media._rate(stream)
        pictures = [frame.to_ndarray(format='rgb24') for frame in container.decode(stream)]
    with media._reading(source) as container:
        audio = next(iter(container.streams.audio), None)
        sound, sound_rate = None, 48000
        if audio is not None:
            sound_rate = audio.codec_context.sample_rate or 48000
            resampler = av.AudioResampler(format='fltp', layout='stereo', rate=sound_rate)
            chunks = [o.to_ndarray() for f in container.decode(audio) for o in resampler.resample(f)]
            chunks += [o.to_ndarray() for o in resampler.resample(None)]
            sound = np.concatenate(chunks, axis=1) if chunks else None
    end = len(pictures) - drop_last if drop_last else len(pictures)
    pictures = pictures[drop_first:end]
    if not pictures:
        raise ValueError('nothing left to write')
    seconds = len(pictures) / float(rate)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(out), 'w') as target:
        video = target.add_stream('libx264', rate=Fraction(rate))
        video.width, video.height = pictures[0].shape[1], pictures[0].shape[0]
        video.pix_fmt = 'yuv420p'
        video.options = {'crf': '14', 'preset': 'medium'}
        voice = None
        if sound is not None:
            voice = target.add_stream('aac', rate=sound_rate, layout='stereo')
            start = int(round((drop_first / float(rate) - audio_shift) * sound_rate))
            length = int(round(seconds * sound_rate))
            padded = np.zeros((2, length), np.float32)
            a, b = max(0, start), min(sound.shape[1], start + length)
            if b > a:
                padded[:, a - start:b - start] = sound[:, a:b]
            sound = padded
        for picture in pictures:
            for packet in video.encode(av.VideoFrame.from_ndarray(picture, format='rgb24')):
                target.mux(packet)
        for packet in video.encode():
            target.mux(packet)
        if voice is not None:
            step = 1024
            for at in range(0, sound.shape[1], step):
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(sound[:, at:at + step]), format='fltp', layout='stereo')
                frame.sample_rate = sound_rate
                frame.pts = at
                for packet in voice.encode(frame):
                    target.mux(packet)
            for packet in voice.encode():
                target.mux(packet)
    return seconds
