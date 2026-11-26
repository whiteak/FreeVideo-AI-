"""Datasets on disk: one folder per dataset under <ComfyUI input>/lora-studio/, a dataset.json beside the pictures.

    lora-studio/
      settings.json             trainer folder, Python, models folder
      links.json                cast card id -> trained LoRA, so @Name loads it
      d1a2b3c4d5e6/
        dataset.json
        img/<item>.png          training pictures (imported and generated)
        reg/<item>.png          regularisation pictures (generic class pictures)
        runs/<run>/             trainer config and log
"""
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from pathlib import Path

from . import captions, images, plans

FOLDER = 'lora-studio'
MAX_ITEMS = 400
MAX_NAME = 80
_ID = re.compile(r'^[di][0-9a-f]{12}$')
_TRIGGER = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_ -]{0,39}$')
_LOCK = threading.RLock()

DEFAULT_SETTINGS = dict(
    trainer='ai-toolkit',      # or 'musubi'
    rank=16, alpha=16, steps=1500, learning_rate=1e-4, resolution=512, repeats=10, save_every=125,
    caption_dropout=0.05, use_regularisation=False, vary_clothes=False,
    offload='auto',            # share of the H3 transformer kept in system memory while training, or 'auto'
    test_prompt='', test_strengths=[0.6, 0.8, 1.0, 1.2], test_seconds=3.0,
)
SETTING_LIMITS = dict(rank=(4, 128), alpha=(1, 128), steps=(100, 20000), learning_rate=(1e-6, 1e-2),
                      resolution=(256, 1536), repeats=(1, 100), save_every=(25, 5000), caption_dropout=(0., .5),
                      test_seconds=(1.625, 10.))


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def new_id(prefix):
    return prefix + secrets.token_hex(6)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=1, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path, default=None):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default
    return value


def _text(value, limit, field):
    if value is None:
        return ''
    if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 and c not in '\n\t' for c in value):
        raise ValueError('invalid_' + field)
    return value.strip()


class Store:
    def __init__(self, input_directory):
        self.input = Path(input_directory).resolve()
        self.root = self.input / FOLDER

    # Paths ------------------------------------------------------------------------------------------------
    def folder(self, identity):
        if not isinstance(identity, str) or not _ID.match(identity) or not identity.startswith('d'):
            raise ValueError('unknown_dataset')
        path = self.root / identity
        if not (path / 'dataset.json').is_file():
            raise ValueError('unknown_dataset')
        return path

    def input_name(self, identity, item):
        """The picture's name relative to ComfyUI's input folder (what /view and the engine's routes take)."""
        return '%s/%s/%s' % (FOLDER, identity, item['file'])

    def item_path(self, identity, item):
        path = (self.folder(identity) / item['file']).resolve()
        if path.parent.parent != self.folder(identity).resolve():
            raise ValueError('invalid_item')
        return path

    # Datasets ---------------------------------------------------------------------------------------------
    def list(self):
        rows = []
        if not self.root.is_dir():
            return rows
        for child in sorted(self.root.iterdir()):
            value = read_json(child / 'dataset.json') if child.is_dir() and _ID.match(child.name) else None
            if not isinstance(value, dict):
                continue
            kept = [row for row in value.get('items', []) if row.get('source') != 'reg' and not row.get('excluded')]
            cover = next((row for row in kept if row['id'] == value.get('reference')), kept[0] if kept else None)
            rows.append(dict(id=value['id'], name=value.get('name', ''), kind=value.get('kind'),
                             trigger=value.get('trigger', ''), count=len(kept), updated_at=value.get('updated_at'),
                             cover=self.input_name(value['id'], cover) if cover else None,
                             lora=value.get('lora')))
        rows.sort(key=lambda row: row.get('updated_at') or '', reverse=True)
        return rows

    def get(self, identity):
        with _LOCK:
            value = read_json(self.folder(identity) / 'dataset.json')
        if not isinstance(value, dict):
            raise ValueError('unknown_dataset')
        return self.view(value)

    def view(self, value):
        result = dict(value)
        result['items'] = [dict(row, input=self.input_name(value['id'], row)) for row in value.get('items', [])]
        return result

    def _load(self, identity):
        value = read_json(self.folder(identity) / 'dataset.json')
        if not isinstance(value, dict):
            raise ValueError('unknown_dataset')
        return value

    def _save(self, value):
        value['updated_at'] = now()
        write_json(self.root / value['id'] / 'dataset.json', value)
        return self.view(value)

    def create(self, body):
        kind = body.get('kind', 'character')
        if kind not in plans.KINDS:
            raise ValueError('invalid_kind')
        name = _text(body.get('name'), MAX_NAME, 'name') or 'Untitled'
        trigger = _text(body.get('trigger'), 40, 'trigger') or captions.suggest_trigger(name)
        if not _TRIGGER.match(trigger):
            raise ValueError('invalid_trigger')
        class_word = _text(body.get('class_word'), 40, 'class_word') or ('person' if kind == 'character' else
                                                                        '' if kind == 'style' else 'object')
        identity = new_id('d')
        stamp = now()
        value = dict(id=identity, name=name, kind=kind, trigger=trigger, class_word=class_word,
                     description=_text(body.get('description'), 1000, 'description'),
                     mode='style' if kind == 'style' else 'subject', reference=None, items=[],
                     settings=dict(DEFAULT_SETTINGS), lora=None, runs=[], tests=[], created_at=stamp, updated_at=stamp)
        with _LOCK:
            (self.root / identity / 'img').mkdir(parents=True, exist_ok=True)
            return self._save(value)

    def update(self, identity, body):
        with _LOCK:
            value = self._load(identity)
            for key, limit in (('name', MAX_NAME), ('description', 1000), ('class_word', 40)):
                if key in body:
                    value[key] = _text(body[key], limit, key) or value[key]
            if 'trigger' in body:
                trigger = _text(body['trigger'], 40, 'trigger')
                if not _TRIGGER.match(trigger):
                    raise ValueError('invalid_trigger')
                value['trigger'] = trigger
            if 'mode' in body:
                if body['mode'] not in ('subject', 'style'):
                    raise ValueError('invalid_mode')
                value['mode'] = body['mode']
            if 'reference' in body:
                if body['reference'] is not None and not any(r['id'] == body['reference'] for r in value['items']):
                    raise ValueError('invalid_item')
                value['reference'] = body['reference']
            if 'settings' in body:
                value['settings'] = self._settings(value.get('settings') or {}, body['settings'])
            return self._save(value)

    @staticmethod
    def _settings(current, given):
        if not isinstance(given, dict):
            raise ValueError('invalid_settings')
        result = dict(DEFAULT_SETTINGS, **current)
        for key, item in given.items():
            if key not in DEFAULT_SETTINGS:
                raise ValueError('invalid_settings')
            if key in SETTING_LIMITS:
                low, high = SETTING_LIMITS[key]
                if type(item) not in (int, float) or not low <= item <= high:
                    raise ValueError('invalid_settings')
                item = int(item) if isinstance(DEFAULT_SETTINGS[key], int) and not isinstance(DEFAULT_SETTINGS[key], bool) else float(item)
            elif key in ('use_regularisation', 'vary_clothes'):
                if type(item) is not bool:
                    raise ValueError('invalid_settings')
            elif key == 'offload':
                if item != 'auto' and (type(item) not in (int, float) or not 0 <= item <= 1):
                    raise ValueError('invalid_settings')
                item = item if item == 'auto' else round(float(item), 2)
            elif key == 'trainer':
                if item not in ('ai-toolkit', 'musubi'):
                    raise ValueError('invalid_settings')
            elif key == 'test_prompt':
                item = _text(item, 2000, 'settings')
            elif key == 'test_strengths':
                if (not isinstance(item, list) or not 1 <= len(item) <= 6
                        or any(type(v) not in (int, float) or not 0 <= v <= 2 for v in item)):
                    raise ValueError('invalid_settings')
                item = [round(float(v), 2) for v in item]
            result[key] = item
        return result

    def delete(self, identity):
        with _LOCK:
            folder = self.folder(identity)
            trash = folder.with_name('.deleted-' + folder.name + '-' + secrets.token_hex(3))
            folder.rename(trash)
        shutil.rmtree(trash, ignore_errors=True)
        links = self.links()
        changed = {k: v for k, v in links.items() if v.get('dataset') != identity}
        if changed != links:
            write_json(self.root / 'links.json', changed)
        return dict(deleted=identity)

    # Items ------------------------------------------------------------------------------------------------
    def add_image(self, identity, image, *, source='import', slot=None, prompt='', seed=None, model='', notes=(),
                  dedupe=True):
        """Save a Pillow image as a new item. With dedupe, a near-copy of a kept picture is refused."""
        if source not in ('import', 'generated', 'reg'):
            raise ValueError('invalid_source')
        fingerprint = images.dhash(image)
        with _LOCK:
            value = self._load(identity)
            if len(value['items']) >= MAX_ITEMS:
                raise ValueError('too_many_items')
            if dedupe and source != 'reg':
                for row in value['items']:
                    if (row.get('source') != 'reg' and not row.get('excluded')
                            and images.distance(row.get('hash'), fingerprint) <= images.DUPLICATE_DISTANCE):
                        return dict(duplicate_of=row['id'], dataset=self.view(value))
            item_id = new_id('i')
            relative = '%s/%s.png' % ('reg' if source == 'reg' else 'img', item_id)
            images.save_png(image, self.root / identity / relative)
            item = dict(id=item_id, file=relative, source=source, slot=slot, prompt=prompt, seed=seed, model=model,
                        caption='', caption_edited=False, drift=None, excluded=False, hash=fingerprint,
                        width=image.width, height=image.height, quality=images.quality(image), notes=list(notes),
                        added_at=now())
            if slot:
                # A slot holds one picture: the one it held before steps aside (kept, but excluded).
                for row in value['items']:
                    if row.get('slot') == slot and not row.get('excluded') and row.get('source') == source:
                        row['excluded'] = True
                        row['replaced_by'] = item_id
            value['items'].append(item)
            if value.get('reference') is None and source != 'reg':
                value['reference'] = item_id
            return dict(item=dict(item, input=self.input_name(identity, item)), dataset=self._save(value))

    def update_item(self, identity, item_id, body):
        with _LOCK:
            value = self._load(identity)
            item = next((row for row in value['items'] if row['id'] == item_id), None)
            if item is None:
                raise ValueError('invalid_item')
            if 'caption' in body:
                item['caption'] = _text(body['caption'], 2000, 'caption')
                item['caption_edited'] = bool(body.get('edited', True))
            if 'excluded' in body:
                if type(body['excluded']) is not bool:
                    raise ValueError('invalid_request')
                item['excluded'] = body['excluded']
            if 'drift' in body:
                drift = body['drift']
                if drift is not None and (not isinstance(drift, dict) or set(drift) - {'score', 'differences'}):
                    raise ValueError('invalid_request')
                item['drift'] = None if drift is None else dict(
                    score=None if drift.get('score') is None else max(0., min(10., float(drift['score']))),
                    differences=_text(drift.get('differences', ''), 300, 'drift'))
            if 'slot' in body:
                item['slot'] = _text(body['slot'], 40, 'slot') or None
            return self._save(value)

    def replace_image(self, identity, item_id, image, note):
        with _LOCK:
            value = self._load(identity)
            item = next((row for row in value['items'] if row['id'] == item_id), None)
            if item is None:
                raise ValueError('invalid_item')
            images.save_png(image, self.item_path(identity, item))
            item.update(width=image.width, height=image.height, hash=images.dhash(image),
                        quality=images.quality(image), notes=sorted(set(item.get('notes', [])) | {note}))
            return self._save(value)

    def delete_item(self, identity, item_id):
        with _LOCK:
            value = self._load(identity)
            item = next((row for row in value['items'] if row['id'] == item_id), None)
            if item is None:
                raise ValueError('invalid_item')
            try:
                self.item_path(identity, item).unlink()
            except FileNotFoundError:
                pass
            value['items'] = [row for row in value['items'] if row['id'] != item_id]
            if value.get('reference') == item_id:
                value['reference'] = next((r['id'] for r in value['items'] if r.get('source') != 'reg'), None)
            return self._save(value)

    def dedupe(self, identity):
        """Exclude every kept picture that repeats an earlier one; returns the dataset and the ids excluded."""
        with _LOCK:
            value = self._load(identity)
            rows = [(r['id'], r.get('hash')) for r in value['items'] if r.get('source') != 'reg' and not r.get('excluded')]
            repeated = set(images.duplicates(rows))
            for row in value['items']:
                if row['id'] in repeated:
                    row['excluded'] = True
                    row['notes'] = sorted(set(row.get('notes', [])) | {'duplicate'})
            return dict(excluded=sorted(repeated), dataset=self._save(value))

    def record(self, identity, key, entry, limit=20):
        """Append a run or test entry (newest last), keeping the last `limit`."""
        with _LOCK:
            value = self._load(identity)
            rows = [row for row in value.get(key, []) if row.get('id') != entry.get('id')]
            rows.append(entry)
            value[key] = rows[-limit:]
            return self._save(value)

    def set_lora(self, identity, lora):
        with _LOCK:
            value = self._load(identity)
            value['lora'] = lora
            return self._save(value)

    def training_items(self, identity):
        value = self._load(identity)
        kept = [row for row in value['items'] if not row.get('excluded')]
        return value, [r for r in kept if r.get('source') != 'reg'], [r for r in kept if r.get('source') == 'reg']

    # Settings and links ----------------------------------------------------------------------------------
    def settings(self):
        value = read_json(self.root / 'settings.json', {})
        return value if isinstance(value, dict) else {}

    def save_settings(self, body):
        if not isinstance(body, dict):
            raise ValueError('invalid_settings')
        value = self.settings()
        for key in ('ai_toolkit_dir', 'ai_toolkit_python', 'musubi_dir', 'musubi_python', 'models_dir'):
            if key in body:
                value[key] = _text(body[key], 1000, 'settings')
        write_json(self.root / 'settings.json', value)
        return value

    def links(self):
        value = read_json(self.root / 'links.json', {})
        return value if isinstance(value, dict) else {}

    def link(self, cast_id, entry):
        if not isinstance(cast_id, str) or not 0 < len(cast_id) <= 128:
            raise ValueError('invalid_request')
        with _LOCK:
            value = self.links()
            if entry is None:
                value.pop(cast_id, None)
            else:
                value[cast_id] = entry
            write_json(self.root / 'links.json', value)
            return value
