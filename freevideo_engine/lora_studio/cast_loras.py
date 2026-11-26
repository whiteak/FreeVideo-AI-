"""@Name loads a LoRA: when a video request carries a cast card's pictures, the LoRA linked to that card joins the
request, and its trigger words go in front of the prompt.

The creation panel already sends each mentioned card's pictures as references marked with the card id
(comfy_assets: reference['cast']). LoRA Studio keeps card id -> LoRA in <input>/lora-studio/links.json. Done at the
node, so the creation panel, the editor and the Agent all get it, and the panel's own LoRA list is left as the user set it.
"""
import json
import logging
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
                log.warning('[FreeVideo] The LoRA linked to a cast card is not in the LoRA folder any more: %s', name)
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
        if trigger:
            words.append(trigger)
        log.info('[FreeVideo] @mention loads LoRA %s at %.2f', name, strength)
    if words and isinstance(text, str):
        from ..lora_triggers import prepend_words
        for phrase in reversed(words):
            text = prepend_words(text, phrase)
    return text, result
