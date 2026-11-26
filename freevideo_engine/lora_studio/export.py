"""Write a dataset in the folder layouts the common LoRA trainers read: kohya sd-scripts, ai-toolkit and Musubi Tuner.

Each picture is copied with a .txt caption of the same name. The caption starts with the trigger word (and the class
word for a subject set), so the trainer needs no trigger setting of its own.
"""
import io
import re
import shutil
import zipfile
from pathlib import Path

from . import captions, training

LAYOUTS = ('kohya', 'ai-toolkit', 'musubi')


def safe_name(text):
    name = re.sub(r'[^A-Za-z0-9_-]+', '_', text or '').strip('_')
    return name[:60] or 'lora'


def caption_for(dataset, item):
    text = item.get('caption') or ''
    return captions.with_trigger(text, dataset.get('trigger'), dataset.get('class_word'), dataset.get('kind'),
                                 dataset.get('mode'))


def reg_caption(dataset, item):
    """A regularisation picture's caption is its prompt without the trigger: plain class words."""
    return captions.clean(item.get('caption') or item.get('prompt') or dataset.get('class_word') or '')


def files(store, dataset_id, layout):
    """(relative path, bytes or source Path) pairs for a layout, plus the config text files it needs."""
    if layout not in LAYOUTS:
        raise ValueError('invalid_layout')
    dataset, kept, reg = store.training_items(dataset_id)
    if not kept:
        raise ValueError('empty_dataset')
    settings = dict(dataset.get('settings') or {})
    use_reg = bool(settings.get('use_regularisation')) and bool(reg)
    trigger, cls = dataset.get('trigger', ''), dataset.get('class_word') or ''
    rows = []
    if layout == 'kohya':
        repeats = int(settings.get('repeats', 10))
        image_dir = 'img/%d_%s' % (repeats, ('%s %s' % (trigger, cls)).strip())
        reg_dir = 'reg/1_%s' % (cls or 'subject')
    else:
        image_dir, reg_dir = 'images', 'reg'
    for index, item in enumerate(kept, 1):
        stem = '%03d_%s' % (index, item['id'])
        rows.append(('%s/%s.png' % (image_dir, stem), store.item_path(dataset_id, item)))
        rows.append(('%s/%s.txt' % (image_dir, stem), caption_for(dataset, item).encode('utf-8')))
    if use_reg:
        for index, item in enumerate(reg, 1):
            stem = 'reg_%03d_%s' % (index, item['id'])
            rows.append(('%s/%s.png' % (reg_dir, stem), store.item_path(dataset_id, item)))
            rows.append(('%s/%s.txt' % (reg_dir, stem), reg_caption(dataset, item).encode('utf-8')))
    rows.append(('README.txt', readme(dataset, layout, len(kept), len(reg) if use_reg else 0).encode('utf-8')))
    return rows, dataset, use_reg


def readme(dataset, layout, count, reg_count):
    lines = ['LoRA Studio export: %s' % dataset.get('name'),
             'Kind: %s   Trigger: %s   Class word: %s' % (dataset.get('kind'), dataset.get('trigger'), dataset.get('class_word')),
             'Pictures: %d   Regularisation pictures: %d   Layout: %s' % (count, reg_count, layout), '']
    if layout == 'kohya':
        lines += ['Point sd-scripts / kohya_ss at the "img" folder as train_data_dir (and "reg" as reg_data_dir).',
                  'The folder name prefix is the repeat count. Captions are the .txt files (caption_extension=.txt).']
    elif layout == 'ai-toolkit':
        lines += ['config.yaml trains a MiniMax H3 LoRA with ai-toolkit: run.py config.yaml from the ai-toolkit folder.',
                  'The folder paths in config.yaml are absolute; edit them if you move this folder.']
    else:
        lines += ['dataset.toml and train.cmd train a MiniMax H3 LoRA with Musubi Tuner (kohya-ss/musubi-tuner).',
                  'Edit the model paths at the top of train.cmd first.']
    return '\n'.join(lines) + '\n'


def write_folder(store, dataset_id, layout, destination, *, settings=None):
    """Write the layout into `destination` (replacing what an earlier export left there) with its trainer files."""
    rows, dataset, use_reg = files(store, dataset_id, layout)
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for relative, source in rows:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(source, Path):
            shutil.copyfile(source, target)
        else:
            target.write_bytes(source)
    extra = training.layout_files(dataset, layout, destination, use_reg, settings or {})
    for relative, text in extra.items():
        (destination / relative).write_text(text, encoding='utf-8', newline='\n' if not relative.endswith('.cmd') else '\r\n')
    return dict(path=str(destination), files=len(rows) + len(extra), layout=layout)


def zip_bytes(store, dataset_id, layout, *, settings=None, root_hint='.'):
    """The layout as a zip in memory, for the page's download button."""
    rows, dataset, use_reg = files(store, dataset_id, layout)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for relative, source in rows:
            if isinstance(source, Path):
                archive.write(source, relative, compress_type=zipfile.ZIP_STORED)
            else:
                archive.writestr(relative, source)
        for relative, text in training.layout_files(dataset, layout, Path(root_hint), use_reg, settings or {}).items():
            archive.writestr(relative, text)
    return buffer.getvalue(), '%s-%s.zip' % (safe_name(dataset.get('name')), layout)
