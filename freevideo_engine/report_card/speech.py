"""Report Card speech check (plan step 1.1): are the prompt's spoken lines heard, and do the lips match the voice?

The words are heard by faster-whisper in its own process (hear.py); the lines are laid over them with the subtitle
aligner (subtitle_sherpa.match_lines), so a line counts as heard only when most of its words were really said. The
lip-sync offset compares the mouth opening of the biggest face (faces.py, insightface 106 landmarks) with the loudness
of the voice frame by frame, over the spoken part only.
"""
import json
import os
import re
import subprocess
from pathlib import Path

SPEECH_PYTHON_FALLBACK = Path(r'C:\Users\JASON\my-agent\backtalk\.venv\Scripts\python.exe')
TIMEOUT = 600
MAX_LAG = 4          # frames each way
OFF_FRAMES = 2       # an offset of this many frames (83 ms at 24 fps) or more is flagged
DIGIT = re.compile(r'\d')


def speech_python():
    for value in (os.environ.get('FREEVIDEO_SPEECH_PYTHON'), SPEECH_PYTHON_FALLBACK):
        if value and Path(value).is_file():
            return str(value)
    return None


def heard_words(audio):
    """Timed words in 16 kHz mono audio (subtitle_ears.decode_mono)."""
    import tempfile
    import numpy as np
    from . import Unavailable
    python = speech_python()
    if python is None:
        raise Unavailable('No speech checker: set FREEVIDEO_SPEECH_PYTHON to a Python with faster-whisper.')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'audio.npy'
        np.save(path, audio)
        result = subprocess.run([python, '-I', str(Path(__file__).with_name('hear.py'))], input=json.dumps(dict(audio=str(path))),
                                capture_output=True, text=True, timeout=TIMEOUT, creationflags=flags)
    if result.returncode:
        raise Unavailable('The speech checker stopped: ' + (result.stderr.strip().splitlines() or ['no message'])[-1])
    return json.loads(result.stdout)['words']


def spoken_lines(folder):
    from ..prompt_vlm.rules import spoken_lines as lines
    try:
        return lines((folder / 'prompt.txt').read_text(encoding='utf-8'))
    except OSError:
        return []


def line_flags(lines, words):
    """Flags for lines not heard, and for digits in a line (the model garbles spoken numbers)."""
    from ..subtitle_sherpa import match_lines
    flags = []
    heard = [dict(text=w['text'].strip(' .,!?;:"'), start=w['start'], end=w['end']) for w in words]
    heard = [w for w in heard if w['text']]
    for line, match in zip(lines, match_lines(lines, heard) if heard else [dict(matched=False)] * len(lines)):
        if DIGIT.search(line):
            flags.append(dict(check='digits', t=match.get('start', 0),
                              text='Numbers in "%s": spell them out (for example "Seven Sins"); spoken digits come out garbled.' % line[:80]))
        if not match['matched']:
            flags.append(dict(check='line', t=0, text='Line not heard clearly: "%s".' % line[:120]))
    return flags


def voice_span(words, fps, count):
    """[first, last] frame of the speech, padded a little, or None without words."""
    if not words:
        return None
    first = max(0, int((words[0]['start'] - .3) * fps))
    last = min(count - 1, int((words[-1]['end'] + .3) * fps))
    return [first, last] if last - first >= 2 * fps else None


def loudness(audio, fps, first, last):
    """Voice loudness (RMS) per video frame over [first, last] of 16 kHz mono audio."""
    import numpy as np
    step = 16000 / fps
    pieces = [audio[int(i * step):int((i + 1) * step)] for i in range(first, last + 1)]
    return np.array([np.sqrt(np.mean(np.square(p))) if p.size else 0. for p in pieces], dtype=np.float32)


def lip_lag(mouth, voice):
    """(lag in frames, correlation): positive means the voice comes after the lips. None when it cannot be told."""
    import numpy as np
    n = min(len(mouth), len(voice))
    m = np.array([np.nan if v is None else v for v in mouth[:n]], dtype=np.float64)
    if n < 24 or np.isnan(m).mean() > .5:
        return None
    m = np.nan_to_num(m, nan=np.nanmean(m))
    e = np.asarray(voice[:n], dtype=np.float64)
    if m.std() == 0 or e.std() == 0:
        return None
    m, e = (m - m.mean()) / m.std(), (e - e.mean()) / e.std()
    scores = {lag: float(np.corrcoef(m[:n - lag], e[lag:])[0, 1]) if lag >= 0 else float(np.corrcoef(m[-lag:], e[:n + lag])[0, 1])
              for lag in range(-MAX_LAG, MAX_LAG + 1)}
    best = max(scores, key=scores.get)
    return (best, round(scores[best], 3)) if scores[best] > .15 else None


def sync_flags(lag, fps):
    if lag is None:
        return [], ['Lip sync not measured: no clear talking face during the speech.']
    frames, score = lag
    ms = round(frames * 1000 / fps)
    if abs(frames) < OFF_FRAMES:
        return [], ['Lip sync within %d ms (match %.2f).' % (abs(ms), score)]
    which = 'after' if frames > 0 else 'before'
    return [dict(check='lip_sync', t=0, frames=frames, ms=ms,
                 text='Lip sync off: the voice comes %d ms %s the lips. Move the voice %d ms %s.'
                      % (abs(ms), which, abs(ms), 'earlier' if frames > 0 else 'later'))], []
