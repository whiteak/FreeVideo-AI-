"""@Name loads a LoRA: when a video request carries a cast card's pictures, the LoRA linked to that card joins the
request, and its trigger words go in front of the prompt.

The creation panel already sends each mentioned card's pictures as references marked with the card id
(comfy_assets: reference['cast']). LoRA Studio keeps card id -> LoRA in <input>/lora-studio/links.json. Done at the
node, so the creation panel, the editor and the Agent all get it, and the panel's own LoRA list is left as the user set it.
"""
import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


def cast_ids(media):
    if not isinstance(media, dict):
        return []
    seen = []
    for row in media.get('references') or ():
        identity = row.get('cast') if isinstance(row, dict) else None
        if isinstance(identity, str) and identity not in seen:
            seen.append(identity)
    return seen


def read_links(input_directory):
    try:
        value = json.loads((Path(input_directory) / 'lora-studio' / 'links.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


# Which still starts a shot: a face close-up for talking or pushing in, waist-up for walking or turning. As a first
# frame Scarlett scored 0.80-0.92 on Jason's face checker, against 0.69 with the same still as a reference.
_CLOSE = re.compile(r'\b(close[- ]?up|push(?:es|ing)? in|portrait|face|talk|talks|talking|speak|speaks|speaking|says?|'
                    r'saying|whisper|sing|sings|singing|laugh|laughs|smile|smiles|smiling|cries|crying|looks? at)\b', re.I)
_WAIST = re.compile(r'\b(walk|walks|walking|turn|turns|turning|dance|dances|dancing|run|runs|running|wave|waves|'
                    r'waving|gesture|gestures|stride|strides|steps?|moves?|moving|medium shot|waist)\b', re.I)


def shot_of(text):
    text = text if isinstance(text, str) else ''
    return 'close' if _CLOSE.search(text) else 'waist' if _WAIST.search(text) else 'default'


def recipe(text, media, width, height, *, folders=None):
    """(text, media, width, height) with the first linked card's recipe applied: its words after the prompt, its
    size, and a still of the card as the first frame when the request has none. Nothing changes without a recipe."""
    ids = cast_ids(media)
    if not ids:
        return text, media, width, height
    if folders is None:
        import folder_paths as folders
    links = read_links(folders.get_input_directory())
    link = next((links[i] for i in ids if isinstance(links.get(i), dict)
                 and (links[i].get('first_frames') or links[i].get('prompt_suffix') or links[i].get('size'))), None)
    if link is None:
        return text, media, width, height
    shot = shot_of(text)  # from the user's words, before the card's own are added
    suffix = link.get('prompt_suffix')
    if isinstance(suffix, str) and suffix and isinstance(text, str) and suffix.lower() not in text.lower():
        text = '%s %s' % (text.rstrip().rstrip('.') + '.', suffix)
    size = link.get('size')
    if isinstance(size, list) and len(size) == 2 and all(type(v) is int and 256 <= v <= 2048 for v in size):
        width, height = size
    frames = link.get('first_frames') or {}
    if isinstance(media, dict) and 'first' not in media and isinstance(frames, dict) and frames:
        name = frames.get(shot) or frames.get('default') or next(iter(frames.values()))
        root = Path(folders.get_input_directory()).resolve()
        path = (root / str(name)).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            path = None
        if path is not None and path.is_file():
            media = dict(media, first=str(path))
            log.info('[Eclipse] @mention starts the %s shot from the card still %s', shot, name)
        else:
            log.warning('[Eclipse] The first-frame still of a cast card is missing: %s', name)
    return text, media, width, height


def extend(text, media, loras, *, folders=None, check=None):
    """(text, loras) with the linked LoRAs of the cards in `media` added. Nothing changes when no card is linked,
    when a linked file is gone (logged), or when the same file is already in the request."""
    ids = cast_ids(media)
    if not ids:
        return text, loras
    if folders is None:
        import folder_paths as folders
    links = read_links(folders.get_input_directory())
    wanted = [links[identity] for identity in ids if isinstance(links.get(identity), dict)]
    if not wanted:
        return text, loras
    result = list(loras or [])
    have = {str(Path(row['path']).resolve()) for row in result if isinstance(row, dict) and row.get('path')}
    names = set(folders.get_filename_list('loras'))
    words = []
    for link in wanted:
        name, strength = link.get('lora'), link.get('strength', 1.0)
        if not isinstance(name, str) or name not in names or type(strength) not in (int, float) or not strength:
            if name:
                log.warning('[Eclipse] The LoRA linked to a cast card is not in the LoRA folder any more: %s', name)
            continue
        path = str(Path(folders.get_full_path_or_raise('loras', name)).resolve())
        if path in have:
            continue
        if check is not None:
            check(path)
        result.append({'path': path, 'strength': float(strength)})
        have.add(path)
        try:
            from ..lora_triggers import get_trigger_words
            trigger = get_trigger_words(name, folders=folders)
        except Exception:
            trigger = None
        if not trigger:
            # The words LoRA Studio linked with the card, for a file that carries none (an ai-toolkit step checkpoint).
            trigger = link.get('trigger_words') or link.get('trigger')
        if isinstance(trigger, str) and trigger.strip():
            words.append(trigger.strip())
        log.info('[Eclipse] @mention loads LoRA %s at %.2f', name, strength)
    if words and isinstance(text, str):
        from ..lora_triggers import prepend_words
        for phrase in reversed(words):
            text = prepend_words(text, phrase)
    return text, result
