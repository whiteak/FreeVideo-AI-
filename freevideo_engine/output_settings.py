"""Output choices: whether a finished video is also saved as a 10-bit copy (see video_10bit.py; off unless the user turns it
on), whether a finished picture is also copied into the output folder (on unless the user turns it off), and which
unwanted extras a video prompt asks the model to leave out (subtitles, background music, narration; all off unless the
user turns them on)."""
import json
from pathlib import Path

from .monitoring import save

FILE = 'output-settings.json'
# One format for the first batch; chosen from playback tests on Windows (H.264 High 10 or HEVC Main 10).
CODEC = 'h264'
# Extras a user can ask every video to leave out, and the words added to the prompt for each. H3 has no negative
# prompt, so these are plain instructions the model usually, but not always, follows.
AVOID = (
    ('avoid_subtitles', 'No subtitles, captions or on-screen text.'),
    ('avoid_music', 'No background music.'),
    ('avoid_narration', 'No narration or voice-over.'),
)


def available():
    """True when this installation's packaged encoder can write the 10-bit format."""
    try:
        import av
        from .video_10bit import CODECS
        return 'yuv420p10le' in {f.name for f in av.codec.Codec(CODECS[CODEC], 'w').video_formats or ()}
    except Exception:
        return False


def read(root):
    """{'ten_bit', 'image_copy', 'avoid_*': bool}; unreadable means 10-bit off, picture copy on, nothing avoided."""
    try:
        value = json.loads((Path(root) / FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        value = {}
    value = value if isinstance(value, dict) else {}
    return dict(ten_bit=value.get('ten_bit') is True, image_copy=value.get('image_copy') is not False,
                **{name: value.get(name) is True for name, _ in AVOID})


def write(root, ten_bit=None, image_copy=None, **avoid):
    """Change the choices given (None leaves one as it is) and return them all."""
    known = {name for name, _ in AVOID}
    if set(avoid) - known:
        raise ValueError('Unknown output setting')
    for name, value in (('The 10-bit copy', ten_bit), ('The picture copy in the output folder', image_copy),
                        *((name, value) for name, value in avoid.items())):
        if value is not None and not isinstance(value, bool):
            raise ValueError(name + ' is switched on or off')
    if ten_bit and not available():
        raise ValueError('This installation cannot write 10-bit video')
    result = read(root)
    if ten_bit is not None:
        result['ten_bit'] = ten_bit
    if image_copy is not None:
        result['image_copy'] = image_copy
    for name, value in avoid.items():
        if value is not None:
            result[name] = value
    save(Path(root) / FILE, result)
    return result


def command_option(root):
    """['--ten-bit', codec] when the copy is on and can be written, else []."""
    return ['--ten-bit', CODEC] if read(root)['ten_bit'] and available() else []


def avoid_prompt(root, text):
    """The video prompt with the chosen "leave out" instructions put first, on a line of their own (before any timed
    lines, so those keep their times). An empty prompt stays empty, and nothing chosen leaves the prompt as it is."""
    if not isinstance(text, str) or not text.strip():
        return text
    try:
        chosen = read(root)
    except Exception:
        return text
    words = ' '.join(sentence for name, sentence in AVOID if chosen.get(name))
    if not words or text.lstrip().startswith(words):
        return text
    return words + '\n' + text
